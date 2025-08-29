# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Step 10 Benchmark (v9): CPU-only, few-shot + live structured beam with multiset constraints
# - No answer leakage; parse only continuation
# - Enforced format [[d1,d2],[d3,d4]]; digits MUST be a permutation (with counts) of input digits
# - Live beam search (k=20) with per-beam past_key_values
# - Warped path blends nudged logits at digit steps (alpha=0.9)
# - Stronger spec: explain the mapping [[a,b],[c,d]] -> [[c,a],[d,b]] (90° CW)
# ==============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ensure CPU-only

import re
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import Counter
from sklearn.decomposition import PCA
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DynamicCache

SEED = 46
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

# -----------------------------
# Token plumbing
# -----------------------------
def build_char_to_ids(tokenizer, chars: List[str]) -> Dict[str, List[int]]:
    table = {ch: set() for ch in chars}
    for tid in range(tokenizer.vocab_size):
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if s in table:
            table[s].add(tid)
    # ensure candidates exist
    for ch in chars:
        if not table[ch]:
            for t in tokenizer.encode(ch, add_special_tokens=False):
                s = tokenizer.decode([t], clean_up_tokenization_spaces=False)
                if s == ch:
                    table[ch].add(t)
    return {ch: sorted(list(ids)) for ch, ids in table.items()}

CHAR_SET = ['[', ']', ',', '1','2','3','4','5','6','7','8','9']
CHAR_TO_IDS = build_char_to_ids(tokenizer, CHAR_SET)

# map token id -> digit (str) for fast reverse lookup
ID_TO_DIGIT = {}
for d in '123456789':
    for tid in CHAR_TO_IDS[d]:
        ID_TO_DIGIT[tid] = d

def digit_ids_for_multiset(counter: Counter) -> List[int]:
    # allowed token ids for digits present (count > 0)
    allowed = []
    for d, c in counter.items():
        if c > 0:
            allowed.extend(CHAR_TO_IDS[str(d)])
    return sorted(set(allowed))

# -----------------------------
# PCA anchor + nudge target (Stages 7/8)
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
# Few-shot prompt
# -----------------------------
FEWSHOT = (
    "Learn this rule and answer with only the 2x2 grid (no extra text).\n"
    "Rotate 90° clockwise: if Input = [[a,b],[c,d]], then Output = [[c,a],[d,b]].\n"
    "Example 1: Input [[1,2],[3,4]] -> Output [[3,1],[4,2]]\n"
    "Example 2: Input [[5,6],[7,8]] -> Output [[7,5],[8,6]]\n"
    "Example 3: Input [[2,9],[3,5]] -> Output [[3,2],[5,9]]\n"
    "Use only digits from the Input and preserve their counts.\n"
)

# -----------------------------
# Task generator
# -----------------------------
def generate_arc_task():
    grid = [[random.randint(1, 9), random.randint(1, 9)] for _ in range(2)]
    rotated = [[grid[1][0], grid[0][0]], [grid[1][1], grid[0][1]]]
    return grid, rotated

# -----------------------------
# Live structured beam with multiset counts
# -----------------------------
from dataclasses import dataclass

@dataclass
class Beam:
    ids: List[int]          # emitted token ids after prompt
    past: tuple             # KV cache
    score: float
    remaining: Counter      # digits left to place (counts)

PLAN = ['[','[','DIGIT',',','DIGIT',']',',','[','DIGIT',',','DIGIT',']',']']

def init_past_for_prompt(prompt_ids):
    with torch.no_grad():
        out = model(prompt_ids, output_hidden_states=True, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :].squeeze(0), out.hidden_states[-1][:, -1, :].squeeze(0)

def step_with_past_single(past, token_id: int, need_hidden: bool):
    inp = torch.tensor([[token_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        cache = DynamicCache.from_legacy_cache(past) if past is not None else None
        out = model(inp, past_key_values=cache, output_hidden_states=need_hidden, use_cache=True)
    logits = out.logits[:, -1, :].squeeze(0)
    new_past = out.past_key_values
    hidden = out.hidden_states[-1][:, -1, :].squeeze(0) if need_hidden else None
    return new_past, logits, hidden

def structured_beam_multiset(prompt_ids, input_counter: Counter, k=20, alpha=0.0):
    base_past, base_logits, base_hidden = init_past_for_prompt(prompt_ids)
    beams = [Beam(ids=[], past=base_past, score=0.0, remaining=input_counter.copy())]

    for sym in PLAN:
        new_beams: List[Beam] = []
        for b in beams:
            # get logits for this position
            if len(b.ids) == 0:
                logits = base_logits.clone()
                hidden = base_hidden if alpha > 0 else None
                past = b.past
            else:
                past, logits, hidden = step_with_past_single(b.past, b.ids[-1], need_hidden=(alpha>0))

            # blend nudge at digit steps
            if sym == 'DIGIT' and alpha > 0 and hidden is not None:
                cur_lat = hidden.detach().cpu().numpy()
                red = pca.transform(cur_lat.reshape(1, -1))[0]
                nudged_red = (1 - alpha) * red + alpha * nudge_target
                inv = pca.inverse_transform(nudged_red.reshape(1, -1))[0]
                nudged_hidden = torch.from_numpy(inv).to(device, torch.float32).unsqueeze(0).unsqueeze(0)
                nudged_logits = model.lm_head(nudged_hidden)[:, 0, :].squeeze(0)
                logits = 0.5 * logits + 0.5 * nudged_logits

            # allowed ids
            if sym == 'DIGIT':
                allowed = digit_ids_for_multiset(b.remaining)
            else:
                allowed = CHAR_TO_IDS[sym]

            if not allowed:
                allowed = list(range(vocab_size))

            logp = torch.log_softmax(logits, dim=-1)
            allowed_scores = logp[allowed]
            topm = min(k, len(allowed))
            vals, idxs = torch.topk(allowed_scores, topm)
            for val, idx in zip(vals.tolist(), idxs.tolist()):
                tok_id = int(allowed[idx])
                # update remaining counts if digit
                new_remaining = b.remaining.copy()
                if sym == 'DIGIT':
                    d_char = ID_TO_DIGIT.get(tok_id, None)
                    if d_char is None:
                        continue
                    d_val = int(d_char)
                    if new_remaining[d_val] <= 0:
                        continue
                    new_remaining[d_val] -= 1

                # advance once to cache state for next step
                new_past, _, _ = step_with_past_single(past, tok_id, need_hidden=False)
                new_beams.append(Beam(ids=b.ids + [tok_id], past=new_past, score=b.score + float(val), remaining=new_remaining))

        # prune
        new_beams.sort(key=lambda x: x.score, reverse=True)
        beams = new_beams[:k]

    best = beams[0]
    continuation = tokenizer.decode(torch.tensor(best.ids, dtype=torch.long), skip_special_tokens=True)
    return best.ids, continuation

# -----------------------------
# Benchmark
# -----------------------------
N = 10
stock_correct = 0
warped_correct = 0

for i in range(1, N+1):
    input_grid, expected_output = generate_arc_task()
    # build multiset of digits from input
    digits = [input_grid[0][0], input_grid[0][1], input_grid[1][0], input_grid[1][1]]
    counter = Counter(digits)

    prompt = FEWSHOT + f"Now solve this one. Input {input_grid} -> Output "
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    ids_s, stock_text = structured_beam_multiset(enc["input_ids"], counter, k=20, alpha=0.0)
    ids_w, warped_text = structured_beam_multiset(enc["input_ids"], counter, k=20, alpha=0.9)

    g_s, _ = first_grid_anywhere(stock_text)
    g_w, _ = first_grid_anywhere(warped_text)
    s_ok = (g_s == expected_output)
    w_ok = (g_w == expected_output)
    stock_correct += int(s_ok)
    warped_correct += int(w_ok)

    print(f"Task {i:02d} | Input {input_grid} | Expect {expected_output} | "
          f"Stock:{'✓' if s_ok else '×'} | Warped:{'✓' if w_ok else '×'} | "
          f"StockOut: {stock_text} | WarpedOut: {warped_text}")

print('\\n=== Summary ===')
print(f'Stock Accuracy : {stock_correct}/{N} = {100*stock_correct/N:.1f}%')
print(f'Warped Accuracy: {warped_correct}/{N} = {100*warped_correct/N:.1f}%')
