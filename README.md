# About
This repository implements __Adversarial Exploration for Generalized Improved State representations (AEGIS)__, an intrinsic motivation method for reinforcement learning that has been found to be highly effective in reward free pretraining settings for environments with stochasticity and partial observability.

Our PPO implementation is based on [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3/tree/master). In case you are mainly interested in the implementation of AEGIS, its major components can be found at [src/algo/intrinsic_rewards/aegis.py]. The other methods we developed based on [DEIR's repo](https://github.com/swan-utokyo/deir).

# Usage
### Installation
```commandline
conda create -n aegis python=3.9
conda activate aegis 
cd aegis_implementation
python3 -m pip install -r requirements.txt
```

### Train AEGIS on MiniGrid
Run the below command in the root directory of this repository to train a DEIR agent in the standard _DoorKey-8x8_ (MiniGrid) environment. Specify the pretraining duration by setting up the `--pretrain_percentage` parameter (default is no pretraining).
```commandline
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=AEGIS \
  --env_source=minigrid \
  --game_name=DoorKey-8x8 \
  --pretrain_percentage=0.5 \
```

### Train AEGIS on MiniGrid with advanced settings
```commandline
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=AEGIS \
  --env_source=minigrid \
  --game_name=DoorKey-8x8-ViewSize-3x3 \
  --can_see_walls=0 \
  --image_noise_scale=0.1
```

### Train AEGIS on ProcGen
```commandline
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=AEGIS \
  --env_source=procgen --game_name=ninja --total_steps=100_000_000 \
  --num_processes=64 --n_steps=256 --batch_size=2048 \
  --n_epochs=3 --model_n_epochs=3 \
  --learning_rate=1e-4 --model_learning_rate=1e-4 \
  --policy_cnn_type=2 --features_dim=256 --latents_dim=256 \
  --model_cnn_type=1 --model_features_dim=64 --model_latents_dim=256 \
  --policy_cnn_norm=LayerNorm --policy_mlp_norm=NoNorm \
  --model_cnn_norm=LayerNorm --model_mlp_norm=NoNorm \
  --adv_norm=0 --adv_eps=1e-5 --adv_momentum=0.9
```

### Example of Training Baselines
Please note that the default value of each option in `src/train.py` is optimized for AEGIS. For now, when training other methods, please use the corresponding hyperparameter values specified in Table 3 (in the Appendix B. An example is `--int_rew_coef=3e-2` and `--rnd_err_norm=0` in the below command.
```commandline
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=NovelD \
  --env_source=minigrid \
  --game_name=DoorKey-8x8 \
  --int_rew_coef=3e-2 \
  --rnd_err_norm=0
```
