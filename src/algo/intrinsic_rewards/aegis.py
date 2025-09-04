import gym
from typing import Dict, Any

import numpy as np
from gym import spaces
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from stable_baselines3.common.utils import obs_as_tensor

from src.algo.intrinsic_rewards.base_model import IntrinsicRewardBaseModel
from src.algo.common_models.mlps import *
from src.utils.enum_types import NormType
from src.utils.running_mean_std import RunningMeanStd

class AEGIS(IntrinsicRewardBaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        max_grad_norm: float = 0.5,
        model_learning_rate: float = 3e-4,
        model_cnn_features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        model_cnn_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        model_features_dim: int = 256,
        model_latents_dim: int = 256,
        model_mlp_norm: NormType = NormType.BatchNorm,
        model_cnn_norm: NormType = NormType.BatchNorm,
        model_gru_norm: NormType = NormType.NoNorm,
        use_model_rnn: int = 0,
        model_mlp_layers: int = 1,
        gru_layers: int = 1,
        use_status_predictor: int = 0,
        # Method-specific params
        aegis_knn_k: int = 5,
        aegis_nem_capacity: int = 512,
        aegis_dst_momentum: float = 0.9,
        aegis_l_coef: float = 0.5,
        aegis_g_coef: float = 0.5,
        aegis_novelty_alpha: float = 0.5,
        aegis_novelty_beta: float = 0.0,
    ):
        super().__init__(observation_space, action_space, activation_fn, normalize_images,
                         optimizer_class, optimizer_kwargs, max_grad_norm, model_learning_rate,
                         model_cnn_features_extractor_class, model_cnn_features_extractor_kwargs,
                         model_features_dim, model_latents_dim, model_mlp_norm,
                         model_cnn_norm, model_gru_norm, use_model_rnn, model_mlp_layers,
                         gru_layers, use_status_predictor)
        
        self.globally_visited_obs = dict()  # A dict of set

        # The following constant values are from the original paper:
        # Results show that α = 0.5 and β = 0 works the best (page 7)
        self.aegis_knn_k = aegis_knn_k
        self.aegis_nem_capacity = aegis_nem_capacity
        self.aegis_dst_momentum = aegis_dst_momentum
        self.aegis_moving_avg_dists = RunningMeanStd(momentum=self.aegis_dst_momentum)
        self.l_coef = aegis_l_coef
        self.g_coef = aegis_g_coef
        self.novelty_alpha = aegis_novelty_alpha
        self.novelty_beta = aegis_novelty_beta

        self._init_novel_experience_memory()

        self._build()
        self._init_modules()
        self._init_optimizers()


    def _build(self) -> None:
        # Build CNN and RNN
        super()._build()

        # Build MLP
        self.model_mlp = InverseModelOutputHeads(
            features_dim=self.model_features_dim,
            latents_dim=self.model_latents_dim,
            activation_fn=self.activation_fn,
            action_num=self.action_num,
            mlp_norm=self.model_mlp_norm,
            mlp_layers=self.model_mlp_layers,
        )


    def _init_novel_experience_memory(self):
        self.obs_shape = get_obs_shape(self.observation_space)
        self.obs_queue_filled = 0
        self.obs_queue_pos = 0
        self.obs_queue = np.zeros((self.aegis_nem_capacity,) + self.obs_shape, dtype=float)
        self.obs_embs_queue = np.zeros((self.aegis_nem_capacity, self.model_features_dim), dtype=float)
        self.mem_queue = np.zeros((self.aegis_nem_capacity, self.gru_layers, self.model_features_dim), dtype=float)
        self.novelty_scores = np.zeros((self.aegis_nem_capacity,), dtype=float)


    def _encode_obs(self, obs, mems, device=None):
        if not isinstance(obs, Tensor):
            obs = obs_as_tensor(obs, device)

        # Get CNN embeddings
        cnn_embs = self._get_cnn_embeddings(obs)

        # If RNN enabled
        if self.use_model_rnn:
            mems = self._get_rnn_embeddings(mems, cnn_embs, self.model_rnns)
            rnn_embs = th.squeeze(mems[:, -1, :])
            return cnn_embs, rnn_embs, mems
        return cnn_embs, cnn_embs, None


    def _get_embeddings(self, curr_obs, next_obs, last_mems, device=None):
        if not isinstance(curr_obs, Tensor):
            curr_obs = obs_as_tensor(curr_obs, device)
            next_obs = obs_as_tensor(next_obs, device)

        # Get CNN embeddings
        curr_cnn_embs = self._get_cnn_embeddings(curr_obs)
        next_cnn_embs = self._get_cnn_embeddings(next_obs)

        # If RNN enabled
        if self.use_model_rnn:
            curr_mems = self._get_rnn_embeddings(last_mems, curr_cnn_embs, self.model_rnns)
            next_mems = self._get_rnn_embeddings(curr_mems, next_cnn_embs, self.model_rnns)
            curr_rnn_embs = th.squeeze(curr_mems[:, -1, :])
            next_rnn_embs = th.squeeze(next_mems[:, -1, :])
            return curr_cnn_embs, next_cnn_embs, curr_rnn_embs, next_rnn_embs, curr_mems

        # If RNN disabled
        return curr_cnn_embs, next_cnn_embs, curr_cnn_embs, next_cnn_embs, None


    def _add_obs(self, obs, embs, mems=None, add_pos=None, novelty_score=0.0):
        """
        Add one new element into the novel experience memory.
        """
        if add_pos is not None:
            assert 0 <= add_pos < self.aegis_nem_capacity
        else:
            add_pos = self.obs_queue_pos
        
        self.obs_queue[add_pos] = np.copy(obs)
        self.obs_embs_queue[add_pos] = np.copy(embs)
        self.mem_queue[add_pos] = np.copy(mems)
        self.novelty_scores[add_pos] = novelty_score
        if self.obs_queue_filled < self.aegis_nem_capacity:
            self.obs_queue_filled += 1
            self.obs_queue_pos += 1

        
    def calculate_novelty_score(self, obs_embs):
        if self.obs_queue_filled <= self.aegis_knn_k:
            novelty_score = 0.0
            similarity = 1.0
        else:
            # Compute the embedding distance between the new observation and all observations in the novel experience memory
            dists = self.calc_euclidean_dists(
                th.tensor(self.obs_embs_queue[:self.obs_queue_filled], dtype=th.float32, device=obs_embs.device),
                obs_embs[-1]
            ) ** 2
            dists = dists.clone().cpu().numpy()
            # Compute the k-nearest neighbours of the new observation in the memory
            knn_dists = np.sort(dists)[:self.aegis_knn_k]
            # Update the moving average of the k-nearest neighbour distances
            
            # print(f"===KNN DISTS: {knn_dists}")
            # self.aegis_moving_avg_dists.update(knn_dists)
            # print(f"===MOV AVG DISTS: {self.aegis_moving_avg_dists.mean}")
            # Normalize the distances with the moving average
            # norm_knn_dists = knn_dists / (self.aegis_moving_avg_dists.mean + 1e-5)
            norm_knn_dists = knn_dists
            # print(f"===NORM KNN DISTS: {norm_knn_dists}")
            # input()

            # Cluster the normalized distances. i.e. they become 0 if too small and 0k is a list of k zeros
            norm_knn_dists = np.maximum(norm_knn_dists - 0.01, np.zeros_like(knn_dists))
            # Compute the Kernel values between the embedding f (x_t) and its neighbours N_k
            similarity = 0.0001 / (norm_knn_dists.mean() + 0.0001)
            novelty_score = 1 - similarity
            # print(f"=== NOV {novelty_score} ==== SIM {similarity}")
        return novelty_score, similarity
        

    def update_experience_memory(self, iteration, new_obs, new_embs, last_mems, stats_logger):
        """
        Update the novel experience memory after generating the intrinsic rewards for
        the current RL rollout. Tries to add one new observation per environment.
        """
        for env_id in range(new_obs.shape[0]):
            next_obs = new_obs[env_id]
            next_embs = new_embs[env_id]
            mems = last_mems[env_id] if self.use_model_rnn else None
            
            if iteration == 0:
                self._add_obs(next_obs, next_embs, mems if self.use_model_rnn else None)
                stats_logger.add(obs_insertions=1)
            else:
                # Calculate the novelty score of the new observation    
                novelty_score, similarity = self.calculate_novelty_score(next_embs.view(1, -1))
                
                # Eviction strategy: Remove the least novel observation if the memory is full
                if self.obs_queue_filled >= self.aegis_nem_capacity:
                    obs_queue_add_pos = np.argmin(self.novelty_scores)
                    stats_logger.add(obs_evictions=1)
                else:
                    obs_queue_add_pos = self.obs_queue_pos
                    stats_logger.add(obs_evictions=0)
                # Addition strategy: Add the new observation if the novelty score is positive
                if novelty_score > 0.0:
                    self._add_obs(new_obs[env_id], new_embs[env_id], last_mems[env_id] if self.use_model_rnn else None,
                                  add_pos=obs_queue_add_pos, novelty_score=novelty_score)
                    stats_logger.add(obs_insertions=1)
                else:
                    stats_logger.add(obs_insertions=0)


    def init_novel_experience_memory(self, initial_obs, curr_key_status, curr_door_status, curr_agent_pos):
        """
        In order to ensure the novel experience memory is not empty on training start
        by adding all observations received at time step 0.
        """
        n_envs = initial_obs.shape[0]
        curr_act = th.zeros((n_envs, get_action_dim(self.action_space)), dtype=th.long if isinstance(self.action_space, spaces.Discrete) else th.float32)

        with th.no_grad():
            _, _, _, _, _, _, _, _, _, _, initial_embs, _, init_mems = \
                self.forward(
                    curr_obs=initial_obs, 
                    next_obs=initial_obs, 
                    last_mems=th.zeros((n_envs, self.gru_layers, self.model_features_dim) if self.use_model_rnn else None),
                    curr_act=curr_act.squeeze(-1),
                    curr_dones=th.zeros((n_envs,), dtype=th.float32), 
                    curr_key_status=curr_key_status, 
                    curr_door_status=curr_door_status, 
                    curr_agent_pos=curr_agent_pos
                )
        
        for i in range(n_envs):
            self._add_obs(initial_obs[i], initial_embs[i], init_mems[i] if self.use_model_rnn else None)
    

    def _update_novelty_scores(self):
        """Refresh all novelty scores"""
        if self.obs_queue_filled == 0:
            return
        else:
            with th.no_grad():
                queue_obs_embs = th.tensor(self.obs_embs_queue[:self.obs_queue_filled],
                                    dtype=th.float32)
                for i in range(self.obs_queue_filled):
                    novelty_score, _ = self.calculate_novelty_score(queue_obs_embs[i].view(1, -1))
                    self.novelty_scores[i] = novelty_score


    def update_embeddings(self):
        """
        Refresh the embeddings and memory states of all observations in the novel experience memory 
        and recalculate their novelty scores
        """
        if self.obs_queue_filled == 0:
            return
        else:
            with th.no_grad():
                obs_tensor = th.tensor(self.obs_queue[:self.obs_queue_filled], dtype=th.float32)
                mems_tensor = th.tensor(self.mem_queue[:self.obs_queue_filled], dtype=th.float32)
                
                _, new_embs, new_mems = self._encode_obs(obs_tensor, mems_tensor, device=obs_tensor.device)                
                
            self.obs_embs_queue[:self.obs_queue_filled] = new_embs.clone().cpu().numpy()
            self.mem_queue[:self.obs_queue_filled] = new_mems.clone().cpu().numpy()
            self._update_novelty_scores()


    def forward(self,
        curr_obs: Tensor, next_obs: Tensor, last_mems: Tensor,
        curr_act: Tensor, curr_dones: Tensor,
        curr_key_status: Optional[Tensor],
        curr_door_status: Optional[Tensor],
        curr_agent_pos: Optional[Tensor],
    ):
        # CNN Extractor
        curr_cnn_embs = self._get_cnn_embeddings(curr_obs)
        next_cnn_embs = self._get_cnn_embeddings(next_obs)
        
        if self.use_model_rnn:
            curr_mems = self._get_rnn_embeddings(last_mems, curr_cnn_embs, self.model_rnns)
            next_mems = self._get_rnn_embeddings(curr_mems, next_cnn_embs, self.model_rnns)
            curr_rnn_embs = th.squeeze(curr_mems[:, -1, :])
            next_rnn_embs = th.squeeze(next_mems[:, -1, :])
            curr_embs = curr_rnn_embs
            next_embs = next_rnn_embs
        else:
            curr_embs = curr_cnn_embs
            next_embs = next_cnn_embs
            curr_mems = None

        # Inverse model
        pred_act = self.model_mlp(curr_embs, next_embs)
        
        # Inverse loss
        curr_dones = curr_dones.view(-1)
        n_samples = (1 - curr_dones).sum()
        inv_losses = F.cross_entropy(pred_act, curr_act, reduction='none') * (1 - curr_dones)
        inv_loss = inv_losses.sum() * (1 / n_samples if n_samples > 0 else 0.0)

        contrastive_loss = self.contrastive_loss(curr_embs.unsqueeze(0), temperature=0.1)

        lmdp_loss = inv_loss + 0.05 * contrastive_loss

        key_loss, door_loss, pos_loss, \
        key_dist, door_dist, goal_dist = \
        self._get_status_prediction_losses(
            curr_embs, curr_key_status, curr_door_status, curr_agent_pos
        )
        return lmdp_loss, inv_losses, \
            key_loss, door_loss, pos_loss, \
            key_dist, door_dist, goal_dist, \
            curr_cnn_embs, next_cnn_embs, \
            curr_embs, next_embs, curr_mems


    def get_intrinsic_rewards(self,
        curr_obs, next_obs, last_mems, curr_act, curr_dones, obs_history,
        key_status, door_status, target_dists, stats_logger
    ):
        with th.no_grad():
            lmdp_loss, inv_losses, \
            key_loss, door_loss, pos_loss, \
            key_dist, door_dist, goal_dist, \
            _, _, \
            curr_embs, next_embs, model_mems = \
                self.forward(
                    curr_obs, next_obs, last_mems,
                    curr_act, curr_dones,
                    key_status, door_status, target_dists
                )
        local_reward = inv_losses.clone().cpu().numpy()
        batch_size = curr_obs.shape[0]
        
        int_rews = np.zeros(batch_size, dtype=np.float32)
        global_rewards = np.zeros(batch_size, dtype=np.float32)
        for env_id in range(batch_size):
            # Update historical observation embeddings
            curr_emb = curr_embs[env_id].view(1, -1)
            next_emb = next_embs[env_id].view(1, -1)
            obs_embs = obs_history[env_id]
            new_embs = [curr_emb, next_emb] if obs_embs is None else [obs_embs, next_emb]
            obs_embs = th.cat(new_embs, dim=0)
            obs_history[env_id] = obs_embs

            if obs_embs.shape[0] > 1:
                curr_obs_global_novelty, _ = self.calculate_novelty_score(curr_emb)
                next_obs_global_novelty, _ = self.calculate_novelty_score(next_emb)
                curr_novelty = curr_obs_global_novelty
                next_novelty = next_obs_global_novelty
                novelty = max(next_novelty - curr_novelty * self.novelty_alpha, self.novelty_beta)

                # Restriction on IRs
                if env_id not in self.globally_visited_obs:
                    self.globally_visited_obs[env_id] = set()

                obs_hash = tuple(next_obs[env_id].reshape(-1).tolist())
                if obs_hash in self.globally_visited_obs[env_id]:
                    novelty *= 0.0
                else:
                    self.globally_visited_obs[env_id].add(obs_hash)
                global_rewards[env_id] += novelty
            
            int_rews[env_id] += self.l_coef*local_reward[env_id] + self.g_coef*global_rewards[env_id]
            # print(f"\nEnv {env_id}:\n Int Rew = {int_rews[env_id]:.4f},\n Local Reward = {local_reward[env_id]:.4f},\n Global Reward = {global_rewards[env_id]:.4f}\n")

        # Logging
        stats_logger.add(
            inv_loss=lmdp_loss,
            local_reward=local_reward,
            global_rewards=global_rewards,
            key_loss=key_loss,
            door_loss=door_loss,
            pos_loss=pos_loss,
            key_dist=key_dist,
            door_dist=door_dist,
            goal_dist=goal_dist,
        )
        return int_rews, next_embs, model_mems, 


    def optimize(self, rollout_data, stats_logger):
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions = rollout_data.actions.long().flatten()

        if self.use_status_predictor:
            curr_key_status = rollout_data.curr_key_status
            curr_door_status = rollout_data.curr_door_status
            curr_target_dists = rollout_data.curr_target_dists
        else:
            curr_key_status = None
            curr_door_status = None
            curr_target_dists = None

        loss, _, \
        key_loss, door_loss, pos_loss, \
        key_dist, door_dist, goal_dist, \
        _, _, _, _, _ = \
            self.forward(
                rollout_data.observations,
                rollout_data.new_observations,
                rollout_data.last_model_mems,
                actions,
                rollout_data.episode_dones,
                curr_key_status,
                curr_door_status,
                curr_target_dists,
            )

        stats_logger.add(
            inv_loss=loss,
            key_loss=key_loss,
            door_loss=door_loss,
            pos_loss=pos_loss,
            key_dist=key_dist,
            door_dist=door_dist,
            goal_dist=goal_dist,
        )
        
        self.model_optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.model_params, self.max_grad_norm)
        self.model_optimizer.step()

        if self.use_status_predictor:
            predictor_loss = key_loss + door_loss + pos_loss
            self.predictor_optimizer.zero_grad()
            predictor_loss.backward()
            self.predictor_optimizer.step()


    def contrastive_loss(self, embeddings, temporal_window=1, temperature=0.1):
        """
        embeddings: Tensor of shape (T, N, D)
            T = time horizon
            N = number of envs
            D = embedding dimension
        """
        T, N, D = embeddings.shape
        z = F.normalize(embeddings, dim=-1)

        # Flatten to (T*N, D)
        z_flat = z.reshape(T * N, D)

        # Similarity matrix (T*N, T*N)
        sim = th.matmul(z_flat, z_flat.T) / temperature

        # Build index grids
        idxs_t = th.arange(T).repeat_interleave(N)   # [0,0,1,1,2,2,...]
        idxs_e = th.arange(N).repeat(T)              # [0,1,0,1,0,1,...]

        t1 = idxs_t.unsqueeze(1).expand(-1, T*N)
        t2 = idxs_t.unsqueeze(0).expand(T*N, -1)
        e1 = idxs_e.unsqueeze(1).expand(-1, T*N)
        e2 = idxs_e.unsqueeze(0).expand(T*N, -1)

        # Same env
        same_env = (e1 == e2)

        # Time distance
        delta_t = (t1 - t2).abs()

        # Positive mask: same env, within temporal_window, not self
        pos_mask = same_env & (delta_t > 0) & (delta_t <= temporal_window)

        # Weighting: closer pairs = higher weight (1/delta_t)
        pos_weights = th.zeros_like(sim)
        pos_weights[pos_mask] = 1.0 / delta_t[pos_mask].float()

        # Negatives = everything else except self
        neg_mask = (~pos_mask) & (~th.eye(T*N, dtype=th.bool, device=z.device))

        # Compute numerator/denominator
        exp_sim = th.exp(sim)

        numerator = (exp_sim * pos_weights).sum(dim=-1)
        denominator = numerator + exp_sim[neg_mask].reshape(T*N, -1).sum(dim=-1)

        # Avoid division by zero if no positives
        valid = numerator > 0
        loss = th.zeros(T*N, device=z.device)
        loss[valid] = -th.log(numerator[valid] / denominator[valid])

        return loss.mean()
