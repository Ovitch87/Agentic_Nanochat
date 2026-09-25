#!/usr/bin/env bash
set -euo pipefail

# Select cpu or cuda; default is cpu.
device="${1:-cpu}"
case "$device" in
    cpu|cuda) ;;
    *) echo "Usage: bash scripts/task_3_eval.sh [cpu|cuda]" >&2; exit 1 ;;
esac

batch_size="${2:-8}"

if ! [[ "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "Batch size must be a positive integer." >&2
    exit 1
fi

# Create a separate results folder for each run.
mkdir -p task_3_results/evaluations
output_dir=$(mktemp -d "task_3_results/evaluations/$(date +%Y%m%d_%H%M%S)_XXXXXX")
echo "Saving evaluation logs to: $output_dir"

# 1. Base model
export NANOCHAT_BASE_DIR="$PWD/task_2_results"
python -u -m scripts.chat_eval \
    --source base --model-tag d4 --step 528 \
    --device-type "$device" \
    --batch-size "$batch_size" \
    --task-name 'ARC-Easy|ARC-Challenge|GSM8K' \
    2>&1 | tee "$output_dir/base_eval.log"

# 2. Mid-trained model
export NANOCHAT_BASE_DIR="$PWD/task_3_results"
python -u -m scripts.chat_eval \
    --source sft --model-tag d4_mid \
    --device-type "$device" \
    --batch-size "$batch_size" \
    --task-name 'ARC-Easy|ARC-Challenge|GSM8K' \
    2>&1 | tee "$output_dir/mid_eval.log"

# 3. SFT model
python -u -m scripts.chat_eval \
    --source sft --model-tag d4_sft \
    --device-type "$device" \
    --batch-size "$batch_size" \
    --task-name 'ARC-Easy|ARC-Challenge|GSM8K' \
    2>&1 | tee "$output_dir/sft_eval.log"