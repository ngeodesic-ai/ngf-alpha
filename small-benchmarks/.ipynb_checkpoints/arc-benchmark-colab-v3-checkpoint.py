# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Rudimentary Benchmark on 10 Synthetic ARC Tasks (Step 10 - CPU safe, no leakage)
# - Avoids CUDA seeding crash by hiding CUDA before importing torch
# - Does NOT leak the expected answer in the prompt
# - Parses ONLY the model's continuation (excludes prompt)
# - Early-stops when first valid 2x2 grid appears and truncates cleanly
# ==============================================================================

import os
# Hide CUDA from PyTorch to avoid any device-side asserts in notebook environments
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import re
import random
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DynamicCache
from time import time

# -----------------------------
# Reproducibility (CPU-only)
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# NOTE: We intentionally skip torch.manual_seed() to avoid CUDA seeding side-effects.

# -----------------------------
# Device & model (CPU-only)
# -----------------------------
device = "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()
torch.set_grad_enabled(False)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = model.config.vocab_size

# -----------------------------
# Helpers: detect/truncate [[a,b],[c,d]]
# -----------------------------
GRID_RE = re.compile(
    r"\[\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*,\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*\]"
)

def first_grid_anywhere(text: str):
    m = GRID_RE.search(text)
    if not m:
        return None, None
    grid = [[int(m.group(1)), int(m.group(2))],
            [int(m.group(3)), int(m.group(4))]]
    return grid, (m.start(), m.end())

def truncate_at_first_grid(text: str):
    _, span = first_grid_anywhere(text)
    return text[:span[1]] if span else text

def hallucination_rate(text: str):
    toks = text.split()
    if not toks:
        return 0.0
    bad = 0
    for t in toks:
        if t.isdigit():
            continue
        if any(ch not in "0123456789[],-" for ch in t):
            bad += 1
    return bad / len(toks)

# -----------------------------
# Stage 7: latent + PCA (simple anchor prompt)
# -----------------------------
anchor_prompt = (
    "Identify the pattern: Input grid [[1,2],[3,4]] -> Output [[4,1],[2,3]] "
    "(90 deg rotate). Apply to [[5,6],[7,8]]."
)
anchor_enc = tokenizer(anchor_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    anchor_out = model(**anchor_enc, output_hidden_states=True)

latent = anchor_out.hidden_states[-1].squeeze(0).detach().cpu().numpy()  # (T,H)
n_components = max(1, min(8, latent.shape[0] - 1))
pca = PCA(n_components=n_components).fit(latent)
reduced_latent = pca.transform(latent).mean(axis=0)

# -----------------------------
# Stage 8: warped nudge seed
# -----------------------------
correct_example = "The output is [[8,5],[6,7]]."
ex_enc = tokenizer(correct_example, return_tensors="pt").to(device)
with torch.no_grad():
    ex_out = model(**ex_enc, output_hidden_states=True)
example_latent = ex_out.hidden_states[-1].mean(dim=1).squeeze().detach().cpu().numpy()
nudge_target = pca.transform(example_latent.reshape(1, -1)).squeeze()

pull_strength, gamma = 1.5, 0.3
def symbolic_nudge(current_reduced, target_reduced, steps=40, dt=0.05):
    pos = current_reduced.copy()
    vel = np.zeros_like(pos)
    for _ in range(steps):
        pull = pull_strength * (target_reduced - pos)
        accel = pull - gamma * vel
        vel += dt * accel
        pos += dt * vel
    return pos

# -----------------------------
# Task generator (rotate 90° CW)
# -----------------------------
def generate_arc_task():
    grid = [[random.randint(1, 9), random.randint(1, 9)] for _ in range(2)]
    rotated = [[grid[1][0], grid[0][0]], [grid[1][1], grid[0][1]]]
    return grid, rotated

# -----------------------------
# Benchmark (10 tasks; continuation-only; no answer leakage)
# -----------------------------
N = 10
stock_correct = 0
warped_correct = 0
start = time()

for i in range(1, N + 1):
    input_grid, expected_output = generate_arc_task()

    # Prompt that does NOT leak the current answer; we ask for "Output:"
    prompt = (
        "Identify the pattern: Input grid [[1,2],[3,4]] -> Output [[4,1],[2,3]] (90 deg rotate). "
        f"Now apply the same rule to {input_grid}. Output:"
    )
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    # --- Stock (greedy continuation, no nudges) ---
    generated_s = enc["input_ids"]
    past_s = None
    for _ in range(80):
        with torch.no_grad():
            if past_s is None:
                mout = model(generated_s, output_hidden_states=False, use_cache=True)
            else:
                mout = model(generated_s, past_key_values=past_s, output_hidden_states=False, use_cache=True)
            past_s = mout.past_key_values
            next_tok = torch.argmax(mout.logits[:, -1, :], dim=-1).unsqueeze(0)
        generated_s = torch.cat([generated_s, next_tok], dim=1)

        # Early stop if continuation already contains a valid grid
        dec_s = tokenizer.decode(generated_s[0][prompt_len:], skip_special_tokens=True)
        g_s, span_s = first_grid_anywhere(dec_s)
        if span_s:
            stock_text = dec_s[:span_s[1]]
            break
    else:
        # No early-stop; decode and truncate post-hoc
        dec_s = tokenizer.decode(generated_s[0][prompt_len:], skip_special_tokens=True)
        stock_text = truncate_at_first_grid(dec_s)
        g_s, _ = first_grid_anywhere(dec_s)

    s_ok = (g_s == expected_output)
    stock_correct += int(s_ok)

    # --- Warped (greedy with periodic nudges + early stop) ---
    generated = enc["input_ids"]
    past = None
    warped_text = None
    for step in range(80):
        with torch.no_grad():
            if past is None:
                mout = model(generated, output_hidden_states=True, use_cache=True)
            else:
                cache = DynamicCache.from_legacy_cache(past)
                mout = model(generated, past_key_values=cache, output_hidden_states=True, use_cache=True)
            past = mout.past_key_values
            logits = mout.logits[:, -1, :]

        next_tok = torch.argmax(logits, dim=-1).unsqueeze(0).clamp(0, vocab_size - 1)
        generated = torch.cat([generated, next_tok], dim=1).to(device)

        # Nudge every 5 tokens
        if generated.shape[1] % 5 == 0:
            hid = mout.hidden_states[-1][:, -1, :].to(device)
            cur_lat = hid.detach().cpu().numpy()
            red = pca.transform(cur_lat)[0]
            nudged_red = symbolic_nudge(red, nudge_target)
            inv = pca.inverse_transform(nudged_red.reshape(1, -1))[0]
            norm = np.linalg.norm(inv)
            if norm > 0:
                inv = (inv / norm) * 5.0
            nudged_hidden = torch.from_numpy(inv).unsqueeze(0).unsqueeze(0).to(device, torch.float32)
            nudged_logits = model.lm_head(nudged_hidden)[:, 0, :]
            if nudged_logits.shape[-1] > vocab_size:
                nudged_logits[..., vocab_size:] = -float("inf")
            nudged_tok = torch.argmax(nudged_logits, dim=-1).unsqueeze(0).clamp(0, vocab_size - 1)
            if int(nudged_tok.item()) == 0:
                nudged_tok = next_tok
            past = None
            generated = torch.cat([generated[:, :-1], nudged_tok], dim=1).to(device)

        # Early stop on first valid grid in continuation
        decoded = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
        g, span = first_grid_anywhere(decoded)
        if span:
            warped_text = decoded[:span[1]]
            break

    if warped_text is None:
        warped_raw = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
        warped_text = truncate_at_first_grid(warped_raw)
        g, _ = first_grid_anywhere(warped_raw)

    w_ok = (g == expected_output)
    warped_correct += int(w_ok)

    # Per-task log
    print(f"Task {i:02d} | Input {input_grid} | Expect {expected_output} | "
          f"Stock:{'✓' if s_ok else '×'} | Warped:{'✓' if w_ok else '×'} | "
          f"Stock Halluc: {hallucination_rate(stock_text):.2%} | Warped Halluc: {hallucination_rate(warped_text):.2%}")

# -----------------------------
# Summary
# -----------------------------
print('\\n=== Summary ===')
print(f'Stock Accuracy : {stock_correct}/{N} = {100*stock_correct/N:.1f}%')
print(f'Warped Accuracy: {warped_correct}/{N} = {100*warped_correct/N:.1f}%')
