
for seed in 0; do
  PYTHONPATH=./ python src/train.py \
    --run_id=$seed \
    --num_processes=2\
    --total_steps=500_000 \
    --int_rew_source=AEGIS \
    --env_source=minigrid \
    --game_name=DoorKey-8x8 \
    --features_dim=64 \
    --model_features_dim=64 \
    --latents_dim=64 \
    --model_latents_dim=64
done