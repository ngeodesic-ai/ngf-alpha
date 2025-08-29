# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Step 10 Benchmark (v8): CPU-only, few-shot + live structured beam decoding
# - No answer leakage; parse only continuation
# - Enforced format [[d1,d2],[d3,d4]], digits ∈ input digits set {a,b,c,d}
# - Live beam search (k=10) with per-beam past_key_values
# - Warped path blends logits with nudged logits at digit steps (alpha=0.8)
# ==============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ensure CPU-only

import re
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict
from sklearn.decomposition import PCA
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DynamicCache

SEED = 45
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

def build_char_to_ids(tokenizer, chars: List[str]) -> Dict[str, List[int]]:
    table = {ch: set() for ch in chars}
    for tid in range(tokenizer.vocab_size):
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if s in table:
            table[s].add(tid)
    # ensure we have candidates
    for ch in chars:
        if not table[ch]:
            # try encoding and filtering for exact match
            for t in tokenizer.encode(ch, add_special_tokens=False):
                s = tokenizer.decode([t], clean_up_tokenization_spaces=False)
                if s == ch:
                    table[ch].add(t)
    return {ch: sorted(list(ids)) for ch, ids in table.items()}

CHAR_SET = ['[', ']', ',', '1','2','3','4','5','6','7','8','9']
CHAR_TO_IDS = build_char_to_ids(tokenizer, CHAR_SET)

def digit_ids_for_set(dset: List[int]) -> List[int]:
    ids = []
    for d in dset:
        chars = CHAR_TO_IDS.get(str(d), [])
        ids.extend(chars)
    return sorted(set(ids))

@dataclass
class Beam:
    ids: List[int]              # token ids emitted so far after the prompt
    past: tuple                 # past_key_values
    score: float

def step_with_past(past, token_id: int, need_hidden: bool):
    """Advance the model one token given a past cache. Return (logits, past, hidden_opt)."""
    inp = torch.tensor([[token_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        if past is None:
            out = model(inp, output_hidden_states=need_hidden, use_cache=True)
        else:
            cache = DynamicCache.from_legacy_cache(past)
            out = model(inp, past_key_values=cache, output_hidden_states=need_hidden, use_cache=True)
    logits = out.logits[:, -1, :].squeeze(0)
    new_past = out.past_key_values
    hidden = out.hidden_states[-1][:, -1, :].squeeze(0) if need_hidden else None
    return logits, new_past, hidden

def init_past_for_prompt(prompt_ids):
    """Run the prompt once to get starting past/logits."""
    with torch.no_grad():
        out = model(prompt_ids, output_hidden_states=True, use_cache=True)
    logits = out.logits[:, -1, :].squeeze(0)
    past = out.past_key_values
    hidden = out.hidden_states[-1][:, -1, :].squeeze(0)
    return logits, past, hidden

PLAN = ['[','[','DIGIT',',','DIGIT',']',',','[','DIGIT',',','DIGIT',']',']']

def structured_beam_live(prompt_ids, input_digits: List[int], k=10, alpha=0.0):
    """Beam search that maintains per-beam past cache through the 12-token plan.
       alpha = 0.0 => stock; alpha in (0,1] => blend nudged logits at DIGIT steps.
    """
    # Initialize from prompt
    base_logits, base_past, base_hidden = init_past_for_prompt(prompt_ids)

    # Create initial beam by *emitting nothing yet*. Next, we must emit PLAN[0].
    beams = [Beam(ids=[], past=base_past, score=0.0)]

    # Precompute allowed ids per symbol
    allowed_map = {}
    allowed_map['['] = CHAR_TO_IDS['[']
    allowed_map[']'] = CHAR_TO_IDS[']']
    allowed_map[','] = CHAR_TO_IDS[',']
    digit_allowed = digit_ids_for_set(input_digits)

    # Stage-7/8 artifacts for nudging
    # Build PCA & nudge target once outside; passed via closure variables

    for sym in PLAN:
        new_beams: List[Beam] = []
        for b in beams:
            # For each beam, get logits at this step by feeding last emitted token (or using initial logits if none)
            if len(b.ids) == 0:
                logits = base_logits.clone()
                past = b.past
                hidden = base_hidden if alpha > 0 else None
            else:
                last_tok = b.ids[-1]
                logits, past, hidden = step_with_past(b.past, last_tok, need_hidden=(alpha>0))

            # Optional nudging for digits
            if sym == 'DIGIT' and alpha > 0 and hidden is not None:
                # Map hidden -> reduced -> blend -> inverse -> logits
                cur_lat = hidden.detach().cpu().numpy()
                red = pca.transform(cur_lat.reshape(1, -1))[0]
                nudged_red = (1 - alpha) * red + alpha * nudge_target
                inv = pca.inverse_transform(nudged_red.reshape(1, -1))[0]
                nudged_hidden = torch.from_numpy(inv).to(device, torch.float32).unsqueeze(0).unsqueeze(0)
                nudged_logits = model.lm_head(nudged_hidden)[:, 0, :].squeeze(0)
                logits = 0.5 * logits + 0.5 * nudged_logits

            # Mask to allowed ids
            allowed = digit_allowed if sym == 'DIGIT' else allowed_map[sym]
            if not allowed:
                allowed = list(range(vocab_size))

            logp = torch.log_softmax(logits, dim=-1)
            allowed_scores = logp[allowed]
            topm = min(k, len(allowed))
            vals, idxs = torch.topk(allowed_scores, topm)
            for val, idx in zip(vals.tolist(), idxs.tolist()):
                tok_id = int(allowed[idx])
                # Advance state by feeding this token and cache result for next step
                _, new_past, _ = step_with_past(past, tok_id, need_hidden=False)
                new_beams.append(Beam(ids=b.ids + [tok_id], past=new_past, score=b.score + float(val)))

        # prune
        new_beams.sort(key=lambda x: x.score, reverse=True)
        beams = new_beams[:k]

    # Best beam
    best = beams[0]
    # materialize continuation text
    cont_text = tokenizer.decode(torch.tensor(best.ids, dtype=torch.long)[...], skip_special_tokens=True)
    return best.ids, cont_text

# -----------------------------
# Stage 7/8: PCA anchor + nudge target
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
    "Learn the rule and answer with only the 2x2 grid (no extra text).\n"
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
# Benchmark
# -----------------------------
N = 10
stock_correct = 0
warped_correct = 0

for i in range(1, N+1):
    input_grid, expected_output = generate_arc_task()
    input_digits = sorted({input_grid[0][0], input_grid[0][1], input_grid[1][0], input_grid[1][1]})

    prompt = FEWSHOT + f"Now solve this one. Input {input_grid} -> Output "
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    # STOCK (alpha=0)
    ids_s, stock_text = structured_beam_live(enc["input_ids"], input_digits, k=10, alpha=0.0)
    # WARPED (alpha=0.8)
    ids_w, warped_text = structured_beam_live(enc["input_ids"], input_digits, k=10, alpha=0.8)

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
