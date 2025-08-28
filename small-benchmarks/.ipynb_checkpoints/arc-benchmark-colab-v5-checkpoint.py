# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Rudimentary Benchmark on 10 Synthetic ARC Tasks (Step 10 - CPU, constrained decoding)
# - CPU-only, no CUDA seeding
# - No answer leakage; parse only continuation
# - Early-stop + truncation at first valid [[a,b],[c,d]]
# - Constrained decoding: restrict vocabulary to digits/brackets/commas/spaces
# ==============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ensure CPU-only

import re
import random
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DynamicCache

# -----------------------------
# Reproducibility (CPU-only)
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# Intentionally skip torch.manual_seed to avoid CUDA side-effects.

# -----------------------------
# Device & model
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

# -----------------------------
# Constrained decoding setup
# -----------------------------
ALLOWED_CHARS = set("0123456789[],- ")
def build_allowed_mask(tokenizer, vocab_size):
    # Allow tokens whose decoded text is composed only of ALLOWED_CHARS
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for i in range(vocab_size):
        s = tokenizer.decode([i], clean_up_tokenization_spaces=False)
        if s and all(ch in ALLOWED_CHARS for ch in s):
            mask[i] = True
    # Ensure space and brackets are allowed
    for ch in [" ", "[", "]", ","]:
        tid = tokenizer.encode(ch, add_special_tokens=False)
        for t in tid:
            mask[t] = True
    return mask

ALLOWED_MASK = build_allowed_mask(tokenizer, vocab_size)  # shape [V]

def apply_constraint(logits):
    # logits: [batch, V] or [V]; set disallowed to very negative
    if logits.dim() == 2:
        logits[:, ~ALLOWED_MASK] = -1e9
    else:
        logits[~ALLOWED_MASK] = -1e9
    return logits

# -----------------------------
# Stage 7: latent + PCA (anchor prompt)
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
# Greedy generation utilities with constraints
# -----------------------------
def greedy_continue(model, input_ids, max_steps=80):
    past = None
    generated = input_ids
    for _ in range(max_steps):
        with torch.no_grad():
            if past is None:
                out = model(generated, output_hidden_states=False, use_cache=True)
            else:
                out = model(generated[:, -1:], past_key_values=past, output_hidden_states=False, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].clone()
            apply_constraint(logits)
            next_tok = torch.argmax(logits, dim=-1).unsqueeze(0)
        generated = torch.cat([generated, next_tok], dim=1)
        yield generated, past

def greedy_continue_warped(model, input_ids, pca, nudge_target, max_steps=80):
    past = None
    generated = input_ids
    for step in range(max_steps):
        with torch.no_grad():
            if past is None:
                out = model(generated, output_hidden_states=True, use_cache=True)
            else:
                cache = DynamicCache.from_legacy_cache(past)
                out = model(generated[:, -1:], past_key_values=cache, output_hidden_states=True, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].clone()
            apply_constraint(logits)
            next_tok = torch.argmax(logits, dim=-1).unsqueeze(0)
        generated = torch.cat([generated, next_tok], dim=1)

        # nudge every 5 tokens
        if generated.shape[1] % 5 == 0:
            hid = out.hidden_states[-1][:, -1, :]
            cur_lat = hid.detach().cpu().numpy()
            red = pca.transform(cur_lat)[0]
            nudged_red = symbolic_nudge(red, nudge_target)
            inv = pca.inverse_transform(nudged_red.reshape(1, -1))[0]
            norm = np.linalg.norm(inv)
            if norm > 0:
                inv = (inv / norm) * 5.0
            nudged_hidden = torch.from_numpy(inv).unsqueeze(0).unsqueeze(0).to(device, torch.float32)
            nudged_logits = model.lm_head(nudged_hidden)[:, 0, :].clone()
            apply_constraint(nudged_logits)
            nudged_tok = torch.argmax(nudged_logits, dim=-1).unsqueeze(0)
            # replace last token and reset cache
            generated = torch.cat([generated[:, :-1], nudged_tok], dim=1)
            past = None

        yield generated, past

# -----------------------------
# Benchmark (10 tasks; continuation-only; no answer leakage)
# -----------------------------
N = 10
stock_correct = 0
warped_correct = 0

for i in range(1, N + 1):
    input_grid, expected_output = generate_arc_task()

    # Prompt that does NOT leak the current answer; ask for "Output:"
    prompt = (
        "Identify the pattern: Input grid [[1,2],[3,4]] -> Output [[4,1],[2,3]] (90 deg rotate). "
        f"Now apply the same rule to {input_grid}. Output:"
    )
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    # --- Stock (constrained greedy) ---
    stock_text, g_s = "", None
    for generated_s, _ in greedy_continue(model, enc["input_ids"], max_steps=80):
        dec_s = tokenizer.decode(generated_s[0][prompt_len:], skip_special_tokens=True)
        g_s, span_s = first_grid_anywhere(dec_s)
        if span_s:
            stock_text = dec_s[:span_s[1]]
            break
    else:
        dec_s = tokenizer.decode(generated_s[0][prompt_len:], skip_special_tokens=True)
        stock_text = truncate_at_first_grid(dec_s)
        g_s, _ = first_grid_anywhere(dec_s)

    s_ok = (g_s == expected_output)
    stock_correct += int(s_ok)

    # --- Warped (constrained greedy + nudges) ---
    warped_text, g = "", None
    for generated, _ in greedy_continue_warped(model, enc["input_ids"], pca, nudge_target, max_steps=80):
        decoded = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
        g, span = first_grid_anywhere(decoded)
        if span:
            warped_text = decoded[:span[1]]
            break
    else:
        warped_raw = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
        warped_text = truncate_at_first_grid(warped_raw)
        g, _ = first_grid_anywhere(warped_raw)

    w_ok = (g == expected_output)
    warped_correct += int(w_ok)

    print(f"Task {i:02d} | Input {input_grid} | Expect {expected_output} | "
          f"Stock:{'✓' if s_ok else '×'} | Warped:{'✓' if w_ok else '×'} | "
          f"StockOut: {stock_text} | WarpedOut: {warped_text}")

# -----------------------------
# Summary
# -----------------------------
print('\\n=== Summary ===')
print(f'Stock Accuracy : {stock_correct}/{N} = {100*stock_correct/N:.1f}%')
print(f'Warped Accuracy: {warped_correct}/{N} = {100*warped_correct/N:.1f}%')
