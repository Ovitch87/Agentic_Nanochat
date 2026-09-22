import os
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from dataclasses import asdict

DEPTH = 4                    
BATCH_SIZE = 8
SEQ_LEN = 2048
EVAL_EVERY = 25
EVAL_TOKENS = 65_536 
TOKENIZER_DIR = Path("Task_5_1_data/tokenizer_32768")
DATA_DIR = Path.home() / ".cache/nanochat/base_data_climbmix"
OUTPUT = Path("task_2_results").resolve()

shutil.copytree(TOKENIZER_DIR, OUTPUT / "tokenizer")
(OUTPUT / "base_data_climbmix").symlink_to(DATA_DIR, target_is_directory=True)
os.environ["NANOCHAT_BASE_DIR"] = str(OUTPUT)
os.environ["NANOCHAT_DTYPE"] = "bfloat16"

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nanochat.flash_attention as attention
attention.USE_FA3 = False           # use PyTorch SDPA; fixes the RTX 5080 kernel error
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit as data_loader
from nanochat.checkpoint_manager import save_checkpoint

assert torch.cuda.is_available(), "Run in your nanochat GPU environment."
device = torch.device("cuda")
torch.manual_seed(42)
torch.set_float32_matmul_precision("high")
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)

# 2. Build the model and calculate nanochat's automatic training settings.
def build_model(depth):
    width = math.ceil(depth * 64 / 128) * 128
    config = GPTConfig(sequence_len=SEQ_LEN, vocab_size=tokenizer.get_vocab_size(),
                       n_layer=depth, n_embd=width, n_head=width // 128,
                       n_kv_head=width // 128, window_pattern="L")
    with torch.device("meta"):
        return GPT(config)


model = build_model(DEPTH)
counts = model.num_scaling_params()
scaling_params = counts["transformer_matrices"] + counts["lm_head"]
target_tokens = 12 * scaling_params

# Nanochat uses a depth-12 reference to scale batch size, learning rates and decay.
reference = build_model(12).num_scaling_params()
reference_tokens = 12 * (reference["transformer_matrices"] + reference["lm_head"])
total_batch = 2 ** round(math.log2(2**19 * (target_tokens / reference_tokens)**0.383))
steps = target_tokens // total_batch
micro_tokens = BATCH_SIZE * SEQ_LEN
assert total_batch % micro_tokens == 0
accumulation = total_batch // micro_tokens
lr_scale = math.sqrt(total_batch / 2**19)
weight_decay = 0.28 * lr_scale * reference_tokens / target_tokens
warmdown = round(0.65 * steps)

model.to_empty(device=device)
model.init_weights()
optimizer = model.setup_optimizer(embedding_lr=0.3 * lr_scale,
                                  unembedding_lr=0.008 * lr_scale,
                                  matrix_lr=0.02 * lr_scale, scalar_lr=0.5 * lr_scale,
                                  weight_decay=weight_decay)
train_model = torch.compile(model, dynamic=False)

# Keep the numbers needed for Task 2's architecture and scaling-law questions.
settings = {"model_config": asdict(model.config), "parameter_counts": counts,
            "scaling_parameters": scaling_params, "default_target_tokens": target_tokens,
            "steps": steps, "training_tokens": steps * total_batch,
            "chinchilla_20N_tokens": 20 * counts["total"], "total_batch_tokens": total_batch,
            "micro_batch_sequences": BATCH_SIZE, "gradient_accumulation": accumulation,
            "eval_tokens_per_split": EVAL_TOKENS, "eval_every": EVAL_EVERY, "seed": 42,
            "attention": "SDPA", "dtype": "bfloat16",
            "nanochat_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
(OUTPUT / "settings.json").write_text(json.dumps(settings, indent=2))
print(f"{counts['total']:,} parameters; {steps} updates; {steps * total_batch:,} tokens")

# 3. Evaluate loss and bpb on the same initial sample of each split each time.
@torch.no_grad()
def evaluate(split):
    loader = data_loader(tokenizer, BATCH_SIZE, SEQ_LEN, split=split, device=device)
    loss_sum = ordinary_loss_sum = byte_sum = 0
    batches = EVAL_TOKENS // micro_tokens
    for _ in range(batches):
        x, y = next(loader)
        losses = model(x, y, loss_reduction="none").reshape_as(y)
        lengths = token_bytes[y]
        loss_sum += losses.sum().item()
        # Special tokens have zero bytes: exclude their loss from bpb.
        ordinary_loss_sum += (losses * (lengths > 0)).sum().item()
        byte_sum += lengths.sum().item()
    loss = loss_sum / (batches * micro_tokens)
    bpb = ordinary_loss_sum / (math.log(2) * byte_sum)
    return loss, bpb


# 4. Train. Several micro-batches contribute to each optimizer update.
train_loader = data_loader(tokenizer, BATCH_SIZE, SEQ_LEN, split="train", device=device)
results = []
with (OUTPUT / "metrics.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "tokens", "train_loss", "val_loss", "train_bpb", "val_bpb"])
    for step in range(steps + 1):
        if step % EVAL_EVERY == 0 or step == steps:
            model.eval()
            train_loss, train_bpb = evaluate("train")
            val_loss, val_bpb = evaluate("val")
            row = [step, step * total_batch, train_loss, val_loss, train_bpb, val_bpb]
            writer.writerow(row)
            f.flush()
            results.append(row)
            print(f"Step {step}: train bpb {train_bpb:.4f}, validation bpb {val_bpb:.4f}")
        if step == steps:
            break

        # Nanochat's learning-rate, momentum and weight-decay schedules.
        if step < 40:
            lr_multiplier = (step + 1) / 40
        elif step <= steps - warmdown:
            lr_multiplier = 1.0
        else:
            lr_multiplier = 0.05 + 0.95 * (steps - step) / warmdown
        if step < 400:
            momentum = 0.85 + 0.12 * step / 400
        elif step >= steps - warmdown:
            momentum = 0.97 - 0.07 * (step - (steps - warmdown)) / warmdown
        else:
            momentum = 0.97
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_multiplier
            if group["kind"] == "muon":
                group["momentum"] = momentum
                group["weight_decay"] = weight_decay * 0.5 * (1 + math.cos(math.pi * step / steps))

        model.train()
        for _ in range(accumulation):
            x, y = next(train_loader)
            loss = train_model(x, y)
            (loss / accumulation).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"Update {step + 1}/{steps}", flush=True)

# 5. Save the final model in nanochat's format (weights and metadata).
save_checkpoint(str(OUTPUT / "base_checkpoints" / f"d{DEPTH}"), steps,
                model.state_dict(), None,
                {"step": steps, "model_config": asdict(model.config), "val_bpb": val_bpb})

# 6. Plot the two bpb curves. The CSV contains the underlying measurements.
plt.figure(figsize=(7, 4))
plt.plot([r[1] / 1e6 for r in results], [r[4] for r in results], label="Training sample")
plt.plot([r[1] / 1e6 for r in results], [r[5] for r in results], label="Validation sample")
plt.xlabel("Training tokens (millions)")
plt.ylabel("Bits per byte")
plt.title(f"Nanochat depth {DEPTH}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT / "bpb.png", dpi=200)
plt.close()

# 7. Generate five raw completions: no system prompt or chat formatting.
model.eval()
prompts = ["The capital of France is", "Water freezes when", "Once upon a time,",
           "To make a cup of tea,", "def add(a, b):\n"]
with (OUTPUT / "completions.txt").open("w", encoding="utf-8") as f:
    f.write("Sampling: temperature=0.7, top_k=50, max_tokens=128, seeds=42..46\n\n")
    for i, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
        generated = list(model.generate(tokens, max_tokens=128, temperature=0.7, top_k=50, seed=42 + i))
        f.write(f"PROMPT: {prompt}\nCOMPLETION: {tokenizer.decode(generated)}\n\n")
print(f"Done. Results: {OUTPUT}")
