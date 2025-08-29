# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Step 10 Benchmark (v7): CPU-only, few-shot + structured beam decoding
# - No answer leakage; parse only continuation
# - Enforced format [[d1,d2],[d3,d4]], d∈{1..9}
# - Beam search (k=5) on digit positions
# - Warped path mixes logits with nudged logits at digit positions
# ==============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ensure CPU-only

import re
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.decomposition import PCA
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DynamicCache

SEED = 44
random.seed(SEED)
np.random.seed(SEED)

device = "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()
torch.set_grad_enabled(False)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = model.config.vocab_size

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
    for ch in [',', '[', ']']:
        if not table[ch]:
            for t in tokenizer.encode(ch, add_special_tokens=False):
                s = tokenizer.decode([t], clean_up_tokenization_spaces=False)
                if s == ch:
                    table[ch].add(t)
    return {ch: sorted(list(ids)) for ch, ids in table.items()}

CHAR_SET = ['[', ']', ',', '1','2','3','4','5','6','7','8','9']
CHAR_TO_IDS = build_char_to_ids(tokenizer, CHAR_SET)
DIGIT_IDS = [CHAR_TO_IDS[str(d)][0] for d in range(1,10) if CHAR_TO_IDS.get(str(d))]

def pick_from_ids(logits, allowed_ids):
    if not allowed_ids:
        return int(torch.argmax(logits).item())
    sub = logits[allowed_ids]
    idx = int(torch.argmax(sub).item())
    return int(allowed_ids[idx])

# -----------------------------
# Stage 7/8: PCA anchor + nudge target (same as before)
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

correct_example = "The output is [[8,5],[6,7]]."
ex_enc = tokenizer(correct_example, return_tensors="pt").to(device)
with torch.no_grad():
    ex_out = model(**ex_enc, output_hidden_states=True)
example_latent = ex_out.hidden_states[-1].mean(dim=1).squeeze().detach().cpu().numpy()
nudge_target = pca.transform(example_latent.reshape(1, -1)).squeeze()

# -----------------------------
# Few-shot prompt (3 examples)
# -----------------------------
FEWSHOT = (
    "Learn the rule and answer with only the 2x2 grid.\n"
    "Example 1: Input [[1,2],[3,4]] -> Output [[4,1],[2,3]] (90 deg rotate)\n"
    "Example 2: Input [[5,6],[7,8]] -> Output [[7,5],[8,6]] (90 deg rotate)\n"
    "Example 3: Input [[2,5],[8,9]] -> Output [[8,2],[9,5]] (90 deg rotate)\n"
)

# -----------------------------
# Task generator
# -----------------------------
def generate_arc_task():
    grid = [[random.randint(1, 9), random.randint(1, 9)] for _ in range(2)]
    rotated = [[grid[1][0], grid[0][0]], [grid[1][1], grid[0][1]]]
    return grid, rotated

# -----------------------------
# Structured Beam Search over digits
# -----------------------------
@dataclass
class Beam:
    ids: List[int]   # token ids emitted after prompt
    score: float

def structured_beam_decode(logits_seq, k=5):
    """Given a list of logits tensors (one per symbol), where punctuation steps
       provide candidate id sets and digit steps provide DIGIT_IDS, run beam search.
       logits_seq: list of tuples (logits, allowed_ids) for each of 12 emitted tokens: "[", "[", d, ",", d, "]", ",", "[", d, ",", d, "]", "]"
       Returns the best sequence of token ids (length 12).
    """
    beams = [Beam(ids=[], score=0.0)]
    for logits, allowed in logits_seq:
        new_beams = []
        # ensure allowed is not empty
        if not allowed:
            # fall back to full vocab (shouldn't happen)
            allowed = list(range(vocab_size))
        logp = torch.log_softmax(logits, dim=-1)
        for b in beams:
            # pick top candidates within allowed
            allowed_scores = logp[allowed]
            topk = min(k, len(allowed))
            vals, idxs = torch.topk(allowed_scores, topk)
            for val, idx in zip(vals.tolist(), idxs.tolist()):
                tok_id = allowed[idx]
                new_beams.append(Beam(ids=b.ids + [int(tok_id)], score=b.score + float(val)))
        # prune
        new_beams.sort(key=lambda x: x.score, reverse=True)
        beams = new_beams[:k]
    return beams[0].ids

def collect_logits_for_plan(model, input_ids, plan_chars, mix_nudge=False):
    """Run the model stepwise; for each step, produce logits and allowed ids for beam search.
       plan_chars: the 12-symbol plan to emit (["[","[","DIGIT",",",...] ... ,"]"]).
       mix_nudge: if True, blend logits with nudged logits at DIGIT steps.
    """
    generated = input_ids
    past = None
    seq = []
    for sym in plan_chars:
        with torch.no_grad():
            if past is None:
                out = model(generated, output_hidden_states=True, use_cache=True)
            else:
                cache = DynamicCache.from_legacy_cache(past)
                out = model(generated[:, -1:], past_key_values=cache, output_hidden_states=True, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].squeeze(0)

        if sym == 'DIGIT' and mix_nudge:
            hid = out.hidden_states[-1][:, -1, :]
            cur_lat = hid.detach().cpu().numpy()
            red = pca.transform(cur_lat)[0]
            # gentle blend toward nudge_target
            nudged_red = 0.5 * red + 0.5 * nudge_target
            inv = pca.inverse_transform(nudged_red.reshape(1, -1))[0]
            nudged_hidden = torch.from_numpy(inv).unsqueeze(0).unsqueeze(0).to(device, torch.float32)
            nudged_logits = model.lm_head(nudged_hidden)[:, 0, :].squeeze(0)
            logits = 0.5 * logits + 0.5 * nudged_logits

        allowed = None
        if sym == 'DIGIT':
            allowed = DIGIT_IDS
        else:
            allowed = CHAR_TO_IDS.get(sym, [])
        seq.append((logits, allowed))

        # We also append the *expected* char greedily to keep the model state moving along the plan
        if sym == 'DIGIT':
            # placeholder digit '1' just to step the state; the actual digit will be replaced by beam result later
            next_id = DIGIT_IDS[0]
        else:
            ids = CHAR_TO_IDS.get(sym, [])
            next_id = ids[0] if ids else int(torch.argmax(logits).item())
        generated = torch.cat([generated, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    return seq

def realize_sequence(input_ids, token_ids):
    """Append the chosen token_ids after the prompt and decode continuation text."""
    gen = torch.cat([input_ids, torch.tensor([token_ids], dtype=torch.long, device=device)], dim=1)
    return gen

# Plan of 12 symbols for the whole grid
PLAN = ['[','[','DIGIT',',','DIGIT',']',',','[','DIGIT',',','DIGIT',']',']']

# -----------------------------
# Benchmark
# -----------------------------
N = 10
stock_correct = 0
warped_correct = 0

for i in range(1, N+1):
    input_grid, expected_output = generate_arc_task()

    fewshot_prompt = (
        FEWSHOT
        + f"Now solve this one. Input {input_grid} -> Output "
    )
    enc = tokenizer(fewshot_prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    # --- STOCK: collect logits along plan and beam-search digits ---
    seq_stock = collect_logits_for_plan(model, enc["input_ids"], PLAN, mix_nudge=False)
    ids_stock = structured_beam_decode(seq_stock, k=5)
    gen_s = realize_sequence(enc["input_ids"], ids_stock)
    stock_text = tokenizer.decode(gen_s[0][prompt_len:], skip_special_tokens=True)
    g_s, _ = first_grid_anywhere(stock_text)
    s_ok = (g_s == expected_output)
    stock_correct += int(s_ok)

    # --- WARPED: same but blend nudged logits at digit steps ---
    seq_warp = collect_logits_for_plan(model, enc["input_ids"], PLAN, mix_nudge=True)
    ids_warp = structured_beam_decode(seq_warp, k=5)
    gen_w = realize_sequence(enc["input_ids"], ids_warp)
    warped_text = tokenizer.decode(gen_w[0][prompt_len:], skip_special_tokens=True)
    g_w, _ = first_grid_anywhere(warped_text)
    w_ok = (g_w == expected_output)
    warped_correct += int(w_ok)

    print(f"Task {i:02d} | Input {input_grid} | Expect {expected_output} | "
          f"Stock:{'✓' if s_ok else '×'} | Warped:{'✓' if w_ok else '×'} | "
          f"StockOut: {stock_text} | WarpedOut: {warped_text}")

print('\\n=== Summary ===')
print(f'Stock Accuracy : {stock_correct}/{N} = {100*stock_correct/N:.1f}%')
print(f'Warped Accuracy: {warped_correct}/{N} = {100*warped_correct/N:.1f}%')
