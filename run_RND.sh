#!/bin/bash
declare -a test_cases=("NoPreTrain" "QuarterPreTrain" "HalfPreTrain" "ThreeQuarterPreTrain")
declare -a methods=("RND")
declare -a arr=("DoorKey-8x8" "DoorKey-16x16" "FourRooms" "MultiRoom-N4-S5" "MultiRoom-N6" "KeyCorridorS4R3" "KeyCorridorS6R3") # "ObstructedMaze-Full-V3"

for group_name in "${test_cases[@]}"; do
  for env in "${arr[@]}"; do
    for int_rew_source in "${methods[@]}"; do
      for seed in 0 1 2 3 4 5 6 7 8 9; do
        # Set total training steps based on the environment
        if [ "$env" == "DoorKey-8x8" ]; then
          total_steps=500_000        
        elif [ "$env" == "MultiRoom-N4-S5" ]; then
          total_steps=1_000_000
        elif [ "$env" == "KeyCorridorS4R3" ]; then
          total_steps=2_500_000
        elif [ "$env" == "KeyCorridorS6R3" ]; then
          total_steps=5_000_000
        elif [ "$env" == "ObstructedMaze-Full-V3" ]; then
          total_steps=10_000_000
        else 
          total_steps=2_000_000
        fi

        # Logging explored states options
        # 0 - Not to log
        # 1 - Log both episodic and lifelong states
        # 2 - Log episodic visited states only
        log_explored_states=0

        # Default hyperparameters for intrinsic rewards
        int_rew_momentum=0.9
        rnd_err_norm=1
        int_rew_coef=1e-2
        
        # Adjust hyperparameters based on the intrinsic reward method
        if [ "$int_rew_source" == "NGU" ]; then
          int_rew_coef=1e-3
          int_rew_momentum=0.0
        elif [ "$int_rew_source" == "NovelD" ]; then
          int_rew_coef=3e-2
          rnd_err_norm=0
        elif [ "$int_rew_source" == "RND" ]; then
          int_rew_coef=3e-3
          rnd_err_norm=0
        fi

        # Set pretraining percentage based on the group name
        if [ "$group_name" == "ThreeQuarterPreTrain" ]; then
          pretrain_percentage=0.75
        elif [ "$group_name" == "HalfPreTrain" ]; then
          pretrain_percentage=0.5
        elif [ "$group_name" == "QuarterPreTrain" ]; then
          pretrain_percentage=0.25
        else
          pretrain_percentage=0.0
        fi

        PYTHONPATH=./ python3 src/train.py \
          --group_name=$group_name \
          --run_id=$seed \
          --total_steps=$total_steps \
          --pretrain_percentage=$pretrain_percentage \
          --int_rew_source=$int_rew_source \
          --env_source=minigrid \
          --game_name=$env \
          --features_dim=64 \
          --model_features_dim=64 \
          --latents_dim=128 \
          --model_latents_dim=128 \
          --int_rew_coef=$int_rew_coef \
          --int_rew_momentum=$int_rew_momentum \
          --rnd_err_norm=$rnd_err_norm
        
        # redirect terminal output to null
      done
    done
  done
done
echo "All experiments done!"