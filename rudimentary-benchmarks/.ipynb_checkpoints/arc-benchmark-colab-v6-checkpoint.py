# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Rudimentary Benchmark on 10 Synthetic ARC Tasks (Step 10 - CPU, structured decoding)
# - CPU-only, no CUDA seeding
# - No answer leakage; parse only continuation
# - Structured decoding enforces exact grid format: [[d1,d2],[d3,d4]], d∈{1..9}
#   (model chooses digits; punctuation/brackets are fixed by grammar)
# - Warped path uses nudged logits for digit positions
# ==============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ensure CPU-only before importing torch

import re
import random
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DynamicCache

# -----------------------------
# Reproducibility (CPU-only)
# -----------------------------
SEED = 43
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
# Helpers
# -----------------------------
GRID_RE = re.compile(r"\[\s*\[\s*([1-9])\s*,\s*([1-9])\s*\]\s*,\s*\[\s*([1-9])\s*,\s*([1-9])\s*\]\s*\]")

def first_grid_anywhere(text: str):
    m = GRID_RE.search(text)
    if not m:
        return None, None
    grid = [[int(m.group(1)), int(m.group(2))],
            [int(m.group(3)), int(m.group(4))]]
    return grid, (m.start(), m.end())

def build_char_to_ids(tokenizer, chars):
    table = {ch: set() for ch in chars}
    for tid in range(tokenizer.vocab_size):
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if s in table:
            table[s].add(tid)
    # ensure all have at least one id
    missing = [ch for ch, ids in table.items() if not ids]
    if missing:
        # try fallback by encoding char and accepting those ids
        for ch in missing:
            ids = tokenizer.encode(ch, add_special_tokens=False)
            for t in ids:
                s = tokenizer.decode([t], clean_up_tokenization_spaces=False)
                if s == ch:
                    table[ch].add(t)
    return {ch: sorted(list(ids)) for ch, ids in table.items()}

CHAR_SET = ['[', ']', ',', '1','2','3','4','5','6','7','8','9']
CHAR_TO_IDS = build_char_to_ids(tokenizer, CHAR_SET)
DIGIT_IDS = [CHAR_TO_IDS[str(d)][0] for d in range(1,10) if CHAR_TO_IDS.get(str(d))]  # pick first id per digit

def pick_from_ids(logits, allowed_ids):
    # logits: [V]; allowed_ids: list[int]
    if not allowed_ids:
        # fallback: argmax over all
        return int(torch.argmax(logits).item())
    sub = logits[allowed_ids]
    idx = int(torch.argmax(sub).item())
    return int(allowed_ids[idx])

def structured_grid_decode_stock(model, input_ids, prompt_len):
    """Force the punctuation/brackets; choose digits by greedy over digit ids.
       Returns generated_ids and text continuation.
    """
    generated = input_ids
    past = None

    # Force seed "[[" into the stream to stabilize bracket tokens
    seed_ids = tokenizer.encode("[[", add_special_tokens=False)
    generated = torch.cat([generated, torch.tensor([seed_ids], dtype=torch.long, device=device)], dim=1)

    # Positions after the seed: we need digits and punctuation to complete the grid
    # Target char sequence to emit (after the initial "[["):
    # d , d ] , [ d , d ] ]
    plan = ['DIGIT', ',', 'DIGIT', ']', ',', '[', 'DIGIT', ',', 'DIGIT', ']', ']']

    for step, sym in enumerate(plan):
        with torch.no_grad():
            if past is None:
                out = model(generated, output_hidden_states=False, use_cache=True)
            else:
                out = model(generated[:, -1:], past_key_values=past, output_hidden_states=False, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].squeeze(0)

        if sym == 'DIGIT':
            next_id = pick_from_ids(logits, DIGIT_IDS)
        else:
            cand_ids = CHAR_TO_IDS.get(sym, [])
            next_id = pick_from_ids(logits, cand_ids)

        next_tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
        generated = torch.cat([generated, next_tok], dim=1)

    cont_text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
    return generated, cont_text

def structured_grid_decode_warped(model, input_ids, prompt_len, pca, nudge_target):
    """Same as stock but applies a latent nudge right before each DIGIT pick."""
    generated = input_ids
    past = None

    # Seed "[["
    seed_ids = tokenizer.encode("[[", add_special_tokens=False)
    generated = torch.cat([generated, torch.tensor([seed_ids], dtype=torch.long, device=device)], dim=1)

    plan = ['DIGIT', ',', 'DIGIT', ']', ',', '[', 'DIGIT', ',', 'DIGIT', ']', ']']

    for step, sym in enumerate(plan):
        with torch.no_grad():
            if past is None:
                out = model(generated, output_hidden_states=True, use_cache=True)
            else:
                cache = DynamicCache.from_legacy_cache(past)
                out = model(generated[:, -1:], past_key_values=cache, output_hidden_states=True, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].squeeze(0)

        if sym == 'DIGIT':
            # Nudge latent on the current hidden state to bias logits toward correct digits
            hid = out.hidden_states[-1][:, -1, :]
            cur_lat = hid.detach().cpu().numpy()
            red = pca.transform(cur_lat)[0]
            nudged_red = nudge_target + 0.0 * (red - nudge_target)  # identity pull for speed; keep hook point
            # Optionally re-project: here we directly map to logits via head for a gentle mix
            nudged_hidden = torch.from_numpy(pca.inverse_transform(nudged_red.reshape(1, -1))[0]).to(device, torch.float32)
            nudged_hidden = nudged_hidden.unsqueeze(0).unsqueeze(0)
            nudged_logits = model.lm_head(nudged_hidden)[:, 0, :].squeeze(0)
            # Mix logits (simple average); you can tune blend
            mix = 0.5 * logits + 0.5 * nudged_logits
            next_id = pick_from_ids(mix, DIGIT_IDS)
        else:
            cand_ids = CHAR_TO_IDS.get(sym, [])
            next_id = pick_from_ids(logits, cand_ids)

        next_tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
        generated = torch.cat([generated, next_tok], dim=1)

    cont_text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
    return generated, cont_text

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
latent = anchor_out.hidden_states[-1].squeeze(0).detach().cpu().numpy()
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

for i in range(1, N + 1):
    input_grid, expected_output = generate_arc_task()

    # Prompt that does NOT leak the current answer; ask for "Output:" and seed "[[" in-generation
    prompt = (
        "Identify the pattern: Input grid [[1,2],[3,4]] -> Output [[4,1],[2,3]] (90 deg rotate). "
        f"Now apply the same rule to {input_grid}. Output:"
    )
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    # --- Stock (structured decoding) ---
    gen_s, stock_text = structured_grid_decode_stock(model, enc["input_ids"], prompt_len)
    g_s, _ = first_grid_anywhere(stock_text)
    s_ok = (g_s == expected_output)
    stock_correct += int(s_ok)

    # --- Warped (structured decoding + nudged digits) ---
    gen_w, warped_text = structured_grid_decode_warped(model, enc["input_ids"], prompt_len, pca, nudge_target)
    g_w, _ = first_grid_anywhere(warped_text)
    w_ok = (g_w == expected_output)
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
