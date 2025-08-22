for seed in 0 1 2 3 4 5 6 7 8 9; do
  PYTHONPATH=./ python3 src/train.py \
    --run_id=$seed \
    --total_steps=1000000 \
    --int_rew_source=NGU \
    --env_source=minigrid \
    --game_name=DoorKey-8x8 \
    --int_rew_coef=1e-3 \
    --int_rew_momentum=0.0
  done

declare -a arr=("DoorKey-16x16" 
                "FourRooms"
                "MultiRoom-N6"
                )

for env in "${arr[@]}"; do
  for seed in 0 1 2 3 4 5 6 7 8 9; do
    PYTHONPATH=./ python3 src/train.py \
      --run_id=$seed \
      --total_steps=2000000 \
      --int_rew_source=NGU \
      --env_source=minigrid \
      --game_name=$env \
      --int_rew_coef=1e-3 \
      --int_rew_momentum=0.0
    done
  done

for seed in 0 1 2 3 4 5 6 7 8 9; do
  PYTHONPATH=./ python3 src/train.py \
    --run_id=$seed \
    --total_steps=5000000 \
    --int_rew_source=NGU \
    --env_source=minigrid \
    --game_name=KeyCorridorS6R3 \
    --int_rew_coef=1e-3 \
    --int_rew_momentum=0.0
  done