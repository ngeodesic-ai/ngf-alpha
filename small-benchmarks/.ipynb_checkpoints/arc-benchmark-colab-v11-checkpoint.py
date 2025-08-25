# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Step 10 Benchmark (v11): CPU-only, few-shot + live structured beam over positions
# - Model predicts positions (A,B,C,D) with a *permutation* constraint (no repeats).
# - We realize digits from input via these positions and evaluate vs expected rotation.
# - No answer leakage; parse continuation only.
# - Live beam (k=50) with per-beam KV; warped blends nudged logits at POS steps.
# ==============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ensure CPU-only

import random, re, numpy as np, torch
from dataclasses import dataclass
from typing import List, Dict, Set
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DynamicCache
from sklearn.decomposition import PCA

SEED = 48
random.seed(SEED)
np.random.seed(SEED)

device = "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()
torch.set_grad_enabled(False)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = model.config.vocab_size

# -----------------------------
# Token utilities
# -----------------------------
def build_char_to_ids(tokenizer, chars: List[str]) -> Dict[str, List[int]]:
    table = {ch: set() for ch in chars}
    for tid in range(tokenizer.vocab_size):
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if s in table:
            table[s].add(tid)
    for ch in chars:
        if not table[ch]:
            for t in tokenizer.encode(ch, add_special_tokens=False):
                s = tokenizer.decode([t], clean_up_tokenization_spaces=False)
                if s == ch:
                    table[ch].add(t)
    return {ch: sorted(list(ids)) for ch, ids in table.items()}

CHAR_SET = ['[', ']', ',', 'A', 'B', 'C', 'D']
CHAR_TO_IDS = build_char_to_ids(tokenizer, CHAR_SET)

ID_TO_POS = {}
for L in ['A','B','C','D']:
    for tid in CHAR_TO_IDS[L]:
        ID_TO_POS[tid] = L

# -----------------------------
# PCA anchor + nudge target (Stages 7/8)
# (kept lightweight; used only to build a nudged logits blend at POS steps)
# -----------------------------
anchor_prompt = (
    "Identify the pattern: Input grid [[1,2],[3,4]] -> Output [[3,1],[4,2]] "
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
# Few-shot pointer prompt
# -----------------------------
FEWSHOT = (
    "Answer using ONLY the position grid with letters A,B,C,D (no extra text).\n"
    "Positions in the input: A=top-left, B=top-right, C=bottom-left, D=bottom-right.\n"
    "Rotate 90° clockwise. If Input=[[a,b],[c,d]], then Output positions=[[C,A],[D,B]].\n"
    "Use EACH letter exactly once; do NOT repeat letters.\n"
    "Examples:\n"
    "Input [[1,2],[3,4]] -> Output positions [[C,A],[D,B]]\n"
    "Input [[5,6],[7,8]] -> Output positions [[C,A],[D,B]]\n"
    "Input [[2,9],[3,5]] -> Output positions [[C,A],[D,B]]\n"
)

# -----------------------------
# Task generator (rotate 90° CW ground-truth)
# -----------------------------
def generate_arc_task():
    grid = [[random.randint(1, 9), random.randint(1, 9)] for _ in range(2)]
    rotated = [[grid[1][0], grid[0][0]], [grid[1][1], grid[0][1]]]
    return grid, rotated

# -----------------------------
# Live structured beam with permutation constraint
# -----------------------------
@dataclass
class Beam:
    ids: List[int]
    past: tuple
    score: float
    remaining: Set[str]  # letters still available

PLAN = ['[','[','POS',',','POS',']',',','[','POS',',','POS',']',']']

def init_past(prompt_ids):
    with torch.no_grad():
        out = model(prompt_ids, output_hidden_states=True, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :].squeeze(0), out.hidden_states[-1][:, -1, :].squeeze(0)

def step_token(past, token_id, need_hidden: bool):
    inp = torch.tensor([[token_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        cache = DynamicCache.from_legacy_cache(past) if past is not None else None
        out = model(inp, past_key_values=cache, output_hidden_states=need_hidden, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :].squeeze(0), (out.hidden_states[-1][:, -1, :].squeeze(0) if need_hidden else None)

def structured_beam_positions_perm(prompt_ids, k=50, alpha=0.0):
    base_past, base_logits, base_hidden = init_past(prompt_ids)
    beams = [Beam(ids=[], past=base_past, score=0.0, remaining=set(['A','B','C','D']))]
    allowed_pos_all = CHAR_TO_IDS['A'] + CHAR_TO_IDS['B'] + CHAR_TO_IDS['C'] + CHAR_TO_IDS['D']

    for sym in PLAN:
        new_beams: List[Beam] = []
        for b in beams:
            if len(b.ids) == 0:
                logits = base_logits.clone()
                hidden = base_hidden if alpha > 0 else None
                past = b.past
            else:
                past, logits, hidden = step_token(b.past, b.ids[-1], need_hidden=(alpha>0))

            # POS steps may be nudged
            if sym == 'POS' and alpha > 0 and hidden is not None:
                cur_lat = hidden.detach().cpu().numpy()
                red = pca.transform(cur_lat.reshape(1, -1))[0]
                nudged_red = (1 - alpha) * red + alpha * nudge_target
                inv = pca.inverse_transform(nudged_red.reshape(1, -1))[0]
                nudged_hidden = torch.from_numpy(inv).to(device, torch.float32).unsqueeze(0).unsqueeze(0)
                nudged_logits = model.lm_head(nudged_hidden)[:, 0, :].squeeze(0)
                logits = 0.5 * logits + 0.5 * nudged_logits

            if sym == 'POS':
                # allowed ids correspond ONLY to letters still remaining (no repeats)
                allowed_ids = []
                for L in b.remaining:
                    allowed_ids += CHAR_TO_IDS[L]
            else:
                allowed_ids = CHAR_TO_IDS[sym]

            logp = torch.log_softmax(logits, dim=-1)
            allowed_scores = logp[allowed_ids]
            topm = min(k, len(allowed_ids))
            vals, idxs = torch.topk(allowed_scores, topm)
            for val, idx in zip(vals.tolist(), idxs.tolist()):
                tok_id = int(allowed_ids[idx])
                # if POS, remove that letter from remaining
                if sym == 'POS':
                    L = ID_TO_POS.get(tok_id, None)
                    if (L is None) or (L not in b.remaining):
                        continue
                    new_remaining = set(b.remaining)
                    new_remaining.remove(L)
                else:
                    new_remaining = set(b.remaining)

                new_past, _, _ = step_token(past, tok_id, need_hidden=False)
                new_beams.append(Beam(ids=b.ids + [tok_id], past=new_past, score=b.score + float(val), remaining=new_remaining))

        new_beams.sort(key=lambda x: x.score, reverse=True)
        beams = new_beams[:k]

    best = beams[0]
    cont_text = tokenizer.decode(torch.tensor(best.ids, dtype=torch.long), skip_special_tokens=True)
    return best.ids, cont_text

def realize_positions_to_digits(pos_text: str, input_grid):
    mapping = {'A': input_grid[0][0], 'B': input_grid[0][1], 'C': input_grid[1][0], 'D': input_grid[1][1]}
    letters = [ch for ch in pos_text if ch in 'ABCD']
    if len(letters) != 4:
        return None, None
    P1,P2,P3,P4 = letters
    realized = [[mapping[P1], mapping[P2]], [mapping[P3], mapping[P4]]]
    return realized, letters

# -----------------------------
# Benchmark
# -----------------------------
N = 10
stock_correct = 0
warped_correct = 0

for i in range(1, N+1):
    input_grid, expected_output = generate_arc_task()
    prompt = FEWSHOT + f"Now solve this one. Input {input_grid} -> Output positions "

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]

    # STOCK (alpha=0); WARPED (alpha=0.9)
    ids_s, pos_text_s = structured_beam_positions_perm(enc["input_ids"], k=50, alpha=0.0)
    ids_w, pos_text_w = structured_beam_positions_perm(enc["input_ids"], k=50, alpha=0.9)

    realized_s, letters_s = realize_positions_to_digits(pos_text_s, input_grid)
    realized_w, letters_w = realize_positions_to_digits(pos_text_w, input_grid)

    s_ok = (realized_s == expected_output)
    w_ok = (realized_w == expected_output)
    stock_correct += int(s_ok)
    warped_correct += int(w_ok)

    print(f"Task {i:02d} | Input {input_grid} | Expect {expected_output} | "
          f"Stock:{'✓' if s_ok else '×'} | Warped:{'✓' if w_ok else '×'} | "
          f"StockPos: {pos_text_s} -> {realized_s} | WarpedPos: {pos_text_w} -> {realized_w}")

print('\\n=== Summary ===')
print(f'Stock Accuracy : {stock_correct}/{N} = {100*stock_correct/N:.1f}%')
print(f'Warped Accuracy: {warped_correct}/{N} = {100*warped_correct/N:.1f}%')
