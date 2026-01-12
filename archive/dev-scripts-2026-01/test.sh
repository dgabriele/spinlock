#  LR Scheduling:
#  - --lr-schedule cosine (default) or constant
#  - --warmup-steps N for linear warmup
#
#  Checkpointing:
#  - --checkpoint-dir checkpoints/noa
#  - --save-every 1000 (save every N batches, 0 = epoch-end only)
#  - Auto-saves best_model.pt and epoch_N.pt
#
#  Early Stopping:
#  - --early-stop-patience 2 (stop after N epochs without improvement)

# Example for 100K:
poetry run python scripts/dev/train_noa_state_supervised.py \
--dataset datasets/100k_full_features.h5 \
--vqvae-path checkpoints/production/100k_3family_v1 \
--n-samples 100000 \
--epochs 3 \
--batch-size 4 \
--lr 3e-4 \
--warmup-steps 500 \
--timesteps 256 \
--save-every 2000 \
--early-stop-patience 1

# Test Example
poetry run python scripts/dev/train_noa_state_supervised.py \
      --timesteps 256 --bptt-window 32 \
      --vqvae-path checkpoints/production/100k_3family_v1 \
      --lambda-latent 0.1 --lambda-commit 0.5

