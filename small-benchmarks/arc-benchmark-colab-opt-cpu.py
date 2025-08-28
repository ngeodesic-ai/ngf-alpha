# ==============================================================================
# arc-benchmark-colab-opt-cpu.py
# CPU-optimized A/B benchmark for synthetic ARC-style op classification.
# - 7-class (A..G) fast head: avoids full-vocab softmax
# - Vectorized nudging and calibration through the 7-way path
# - Optional smaller model (distilgpt2) for faster CPU inference
# - Same external API: make_tasks, run_pass, main()
# ==============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

import random
import numpy as np
import torch
from typing import Dict
from dataclasses import dataclass
from transformers import GPT2Tokenizer, GPT2LMHeadModel
try:
    from transformers import DynamicCache
except Exception:
    DynamicCache = None
from sklearn.decomposition import PCA

# -----------------------------
# Config / Repro
# -----------------------------
SEED = 43
# Smaller model is much faster on CPU; switch back to "gpt2" if desired.
MODEL_NAME = "distilgpt2"
N_TASKS = 20  # bump as needed

USE_PREFILTER = False
USE_CALIBRATION = True
CAL_K = 4
PRINT_TASKS = True

# Warped hyperparams
ALPHA_WARPED = 0.90
SYMB_STEPS = 12
DT = 0.06
PULL_STRENGTH = 1.6
GAMMA = 0.3
LAMBDA_REP = 0.5  # global (precomputed) repulsion

# Use last layer for simplicity
NUDGE_LAYER = -1

# --- CPU knobs ---
CPU_FAST = True         # 7-class scoring fast-path
TORCH_THREADS = 2       # try 1–4 depending on CPU cores
torch.set_num_threads(TORCH_THREADS)
torch.set_num_interop_threads(max(1, TORCH_THREADS // 2))

random.seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Ops & letters
# -----------------------------
OPS = ["rotate90","flip_h","flip_v","scale2","rotate_then_flip","swap_minmax","shift_down"]
LETTER_MAP = {
    "A": "rotate90",
    "B": "flip_h",
    "C": "flip_v",
    "D": "scale2",
    "E": "rotate_then_flip",
    "F": "swap_minmax",
    "G": "shift_down",
}
LETTERS = list(LETTER_MAP.keys())

# -----------------------------
# Model
# -----------------------------
device = "cpu"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
model.eval()
torch.set_grad_enabled(False)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Deterministic executors
# -----------------------------
def rotate90(grid): return [list(col) for col in zip(*grid[::-1])]
def flip_h(grid):   return [row[::-1] for row in grid]
def flip_v(grid):   return grid[::-1]
def scale2(grid):   return [[v*2 for v in row] for row in grid]
def rotate_then_flip(grid): return flip_h(rotate90(grid))
def swap_minmax(grid):
    flat = [v for row in grid for v in row]
    mn, mx = min(flat), max(flat)
    return [[(mx if v==mn else mn) if v in (mn,mx) else v for v in row] for row in grid]
def shift_down(grid): return [grid[-1]] + grid[:-1] if grid else grid

EXECUTOR = {
    "rotate90": rotate90, "flip_h": flip_h, "flip_v": flip_v, "scale2": scale2,
    "rotate_then_flip": rotate_then_flip, "swap_minmax": swap_minmax, "shift_down": shift_down,
}

# -----------------------------
# Few-shot (letters)
# -----------------------------
FEWSHOT = (
    "You are given an input grid and an output grid.\n"
    "Pick exactly ONE letter from this legend that maps input->output:\n"
    "{A: rotate90, B: flip_h, C: flip_v, D: scale2, E: rotate_then_flip, F: swap_minmax, G: shift_down}.\n"
    "Return ONLY the letter.\n\n"
    "Input: [[1,9],[2,3]]\nOutput: [[9,1],[2,3]]\nAnswer: F\n\n"
    "Input: [[1,2],[3,4]]\nOutput: [[1,3],[2,4]]\nAnswer: E\n\n"
    "Input: [[1,2,3],[4,5,6],[7,8,9]]\nOutput: [[7,8,9],[4,5,6],[1,2,3]]\nAnswer: C\n\n"
    "Input: [[1,2,3],[4,5,6]]\nOutput: [[4,5,6],[1,2,3]]\nAnswer: G\n\n"
    "Input: [[1,2,3],[4,5,6],[7,8,9]]\nOutput: [[7,4,1],[8,5,2],[9,6,3]]\nAnswer: A\n\n"
    "Input: [[5,6],[7,8]]\nOutput: [[6,5],[8,7]]\nAnswer: B\n\n"
    "Input: [[2,3,4],[5,6,7],[8,9,1]]\nOutput: [[4,6,8],[10,12,14],[16,18,2]]\nAnswer: D\n"
)

assert rotate_then_flip([[1,2],[3,4]]) == [[1,3],[2,4]]
assert flip_v([[1,2],[3,4],[5,6]]) == [[5,6],[3,4],[1,2]]
assert shift_down([[1,2,3],[4,5,6]]) == [[4,5,6],[1,2,3]]

# -----------------------------
# Tiny PCA on letter embeddings
# -----------------------------
def last_hidden(text: str, layer: int) -> np.ndarray:
    with torch.no_grad(), torch.inference_mode():
        out = model(**tokenizer(text, return_tensors="pt").to(device),
                    output_hidden_states=True)
    return out.hidden_states[layer][:, -1, :].squeeze(0).detach().cpu().numpy()

LETTER_PHRASES = ["{}", "Answer: {}"]  # 2 phrasings only (fast)

# Build a small dataset of hidden vectors (letters × phrasings)
H = []
for L in LETTERS:
    for tpl in LETTER_PHRASES:
        H.append(last_hidden(tpl.format(L), NUDGE_LAYER))
H = np.stack(H, axis=0)

pca = PCA(n_components=min(6, H.shape[0]-1)).fit(H)

def letter_target_reduced(letter: str) -> np.ndarray:
    vecs = [last_hidden(tpl.format(letter), NUDGE_LAYER) for tpl in LETTER_PHRASES]
    mean_vec = np.mean(np.stack(vecs, axis=0), axis=0)
    return pca.transform(mean_vec.reshape(1, -1)).squeeze()

LETTER_TARGETS = {L: letter_target_reduced(L) for L in LETTERS}
GLOBAL_ANTI = np.mean(np.stack([LETTER_TARGETS[L] for L in LETTERS], axis=0), axis=0)

# -----------------------------
# Symbolic spring (latent nudge)
# -----------------------------
def symbolic_loop(vec: np.ndarray, tgt: np.ndarray, steps=SYMB_STEPS, dt=DT) -> np.ndarray:
    pos = vec * 12.0  # slightly smaller radius
    vel = np.zeros_like(pos)
    for _ in range(steps):
        pull = PULL_STRENGTH * (tgt - pos)
        accel = pull - GAMMA * vel
        vel += dt * accel
        pos += dt * vel
    return pos

# -----------------------------
# Tokenization helpers (letters; single-token only)
# -----------------------------
def encode_letter_token(letter: str) -> int:
    # prefer a single token variant; fall back to the best scoring variant if needed
    ids_plain = tokenizer.encode(letter, add_special_tokens=False)
    if len(ids_plain) == 1: return ids_plain[0]
    ids_sp = tokenizer.encode(" " + letter, add_special_tokens=False)
    return ids_sp[0] if len(ids_sp) == 1 else ids_plain[0]

LETTER_TOKEN = {L: encode_letter_token(L) for L in LETTERS}

# -----------------------------
# 7-class (A..G) projection cache for CPU-fast path
# -----------------------------
LETTER_ORDER = LETTERS  # A..G order
LETTER_TOKEN_IDS = np.array([LETTER_TOKEN[L] for L in LETTER_ORDER], dtype=np.int64)

with torch.no_grad(), torch.inference_mode():
    W_full = model.lm_head.weight.detach().cpu().numpy()     # [V, H]
    b_full = (model.lm_head.bias.detach().cpu().numpy()
              if model.lm_head.bias is not None
              else np.zeros(W_full.shape[0], dtype=W_full.dtype))
LM_ROWS = W_full[LETTER_TOKEN_IDS, :]  # [7, H]
LM_BIAS = b_full[LETTER_TOKEN_IDS]     # [7]

def _logsumexp_1d(x: np.ndarray) -> float:
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

# -----------------------------
# Prompt cache
# -----------------------------
@dataclass
class PromptCache:
    past: tuple
    logits: torch.Tensor
    hidden: torch.Tensor

def _init_prompt_cache(prompt_ids: torch.Tensor) -> PromptCache:
    with torch.no_grad(), torch.inference_mode():
        out = model(prompt_ids, output_hidden_states=True, use_cache=True)
    return PromptCache(
        past=out.past_key_values,
        logits=out.logits[:, -1, :].squeeze(0),
        hidden=out.hidden_states[NUDGE_LAYER][:, -1, :].squeeze(0),
    )

# -----------------------------
# 7-class fast scoring
# -----------------------------
def score_letters_fast(hidden_last: torch.Tensor, alpha: float,
                       cal_offsets: Dict[str, float]) -> Dict[str, float]:
    """
    Returns per-letter log-prob scores for A..G only (CPU fast-path).
    """
    h = hidden_last.detach().cpu().numpy()        # [H]
    base7 = h @ LM_ROWS.T + LM_BIAS               # [7]

    if alpha <= 0.0:
        lse = _logsumexp_1d(base7)
        logp7 = base7 - lse
        return {L: float(logp7[i]) - cal_offsets.get(L, 0.0)
                for i, L in enumerate(LETTER_ORDER)}

    # with nudging: compute nudged 7-way vector per target letter
    red = pca.transform(h.reshape(1, -1)).squeeze()
    scores = {}
    for i, L in enumerate(LETTER_ORDER):
        tgt = LETTER_TARGETS[L] - LAMBDA_REP * GLOBAL_ANTI
        nudged_red = symbolic_loop(red, tgt, steps=SYMB_STEPS, dt=DT)
        inv = pca.inverse_transform(nudged_red.reshape(1, -1)).squeeze()
        nrm = np.linalg.norm(inv)
        if nrm > 0:
            inv = (inv / nrm) * 5.0

        nudged7 = inv @ LM_ROWS.T + LM_BIAS      # [7]
        mix7 = (1.0 - alpha) * base7 + alpha * nudged7
        lse = _logsumexp_1d(mix7)
        logp7 = mix7 - lse
        scores[L] = float(logp7[i]) - cal_offsets.get(L, 0.0)

    return scores

# -----------------------------
# Multi-prompt calibration (small K) using fast path
# -----------------------------
def build_calibration_offsets(alpha: float, k: int = CAL_K) -> Dict[str, float]:
    if not USE_CALIBRATION:
        return {L: 0.0 for L in LETTERS}
    sums = {L: 0.0 for L in LETTERS}
    for _ in range(k):
        h, w = random.choice([2,3]), random.choice([2,3])
        grid = [[random.randint(1,9) for _ in range(w)] for _ in range(h)]
        neutral = FEWSHOT + f"\nInput: {grid}\nOutput: {grid}\nAnswer: "
        pid = tokenizer(neutral, return_tensors="pt").to(device)["input_ids"]

        with torch.no_grad(), torch.inference_mode():
            out = model(pid, output_hidden_states=True, use_cache=True)
        hidden_last = out.hidden_states[NUDGE_LAYER][:, -1, :].squeeze(0)

        # no offsets inside the call
        fast_scores = score_letters_fast(hidden_last, alpha, {L:0.0 for L in LETTERS})
        for L in LETTERS:
            sums[L] += fast_scores[L]

        # mean-center this prompt’s scores to remove global shift
        mu = np.mean([sums[L] for L in LETTERS])
        for L in LETTERS:
            sums[L] -= mu

    return {L: sums[L] / k for L in LETTERS}

# -----------------------------
# Candidates (for logging only)
# -----------------------------
def consistent_ops(inp, out):
    c = []
    for op in OPS:
        try:
            if EXECUTOR[op](inp) == out:
                c.append(op)
        except Exception:
            pass
    return c

# -----------------------------
# Synthetic tasks
# -----------------------------
def random_grid(h, w): return [[random.randint(1,9) for _ in range(w)] for _ in range(h)]

def make_tasks(n: int, seed: int = SEED):
    rng = random.Random(seed)
    tasks = []
    for _ in range(n):
        h, w = rng.choice([2,3]), rng.choice([2,3])
        grid = [[rng.randint(1,9) for _ in range(w)] for _ in range(h)]
        op = rng.choice(OPS)
        expected = EXECUTOR[op](grid)
        tasks.append((grid, op, expected))
    return tasks

# -----------------------------
# Pass
# -----------------------------
def run_pass(tasks, alpha: float, tag: str):
    eq_total = strict_total = 0
    unamb = amb = 0

    cal = build_calibration_offsets(alpha)

    if PRINT_TASKS:
        print(f"--- {tag} (alpha={alpha:.2f}) ---")

    for i, (grid, true_op, expected) in enumerate(tasks, start=1):
        cands = consistent_ops(grid, expected)
        is_unamb = (len(cands) == 1)
        if is_unamb: unamb += 1
        else: amb += 1

        prompt = FEWSHOT + f"\nInput: {grid}\nOutput: {expected}\nAnswer: "
        pid = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]

        # one forward; then 7-class fast scoring
        cache = _init_prompt_cache(pid)
        if CPU_FAST:
            fast_scores = score_letters_fast(cache.hidden, alpha, cal)
            scores = sorted(fast_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            # unsafe path shouldn't be needed; kept for completeness
            base_logp = torch.log_softmax(cache.logits, dim=-1)
            tmp = []
            for L in LETTERS:
                s = float(base_logp[LETTER_TOKEN[L]].item())
                s -= cal.get(L, 0.0)
                tmp.append((L, s))
            scores = sorted(tmp, key=lambda x: x[1], reverse=True)

        predL, best = scores[0]
        second = scores[1][1] if len(scores) > 1 else float("-inf")
        margin = best - second

        pred_op = LETTER_MAP[predL]
        eq_ok = (EXECUTOR[pred_op](grid) == expected)
        strict_ok = (pred_op == true_op)
        eq_total += int(eq_ok); strict_total += int(strict_ok)

        if PRINT_TASKS:
            tag_task = "Unambiguous" if is_unamb else "Ambiguous"
            print(f"Task {i:03d} | True={true_op} | {tag_task} | Cands={cands if cands else 'ALL?'} | "
                  f"{tag}={pred_op} (letter {predL}) eq={'✓' if eq_ok else '×'} "
                  f"strict={'✓' if strict_ok else '×'} Δ{margin:.3f}")

    if PRINT_TASKS:
        N = len(tasks)
        print(f\"\\n=== {tag} Summary ===\")
        print(f\"Prefilter used   : {False}\")
        print(f\"Unambiguous pairs: {unamb}\")
        print(f\"Ambiguous pairs  : {amb}\")
        print(f\"Equivalence Acc  : {eq_total}/{N} = {100*eq_total/N:.1f}%\")
        print(f\"Strict-label Acc : {strict_total}/{N} = {100*strict_total/N:.1f}%\\n\")

    return {\"N\": len(tasks), \"eq\": eq_total, \"strict\": strict_total}

# -----------------------------
# Main
# -----------------------------
def main():
    tasks = make_tasks(N_TASKS, seed=SEED)
    stock  = run_pass(tasks, alpha=0.0, tag=\"Stock\")
    warped = run_pass(tasks, alpha=ALPHA_WARPED, tag=\"Warped\")

    print(\"=== Side-by-side Summary (same tasks) ===\")
    print(f\"N={stock['N']} | Prefilter={False} | Calibration={USE_CALIBRATION} \"
          f\"| Alpha_warped={ALPHA_WARPED} | NUDGE_LAYER={NUDGE_LAYER} | CAL_K={CAL_K}\")
    print(f\"Equivalence Acc  : Stock {stock['eq']}/{stock['N']} = {100*stock['eq']/stock['N']:.1f}% | \"
          f\"Warped {warped['eq']}/{warped['N']} = {100*warped['eq']/warped['N']:.1f}%\")
    print(f\"Strict-label Acc : Stock {stock['strict']}/{stock['N']} = {100*stock['strict']/stock['N']:.1f}% | \"
          f\"Warped {warped['strict']}/{warped['N']} = {100*warped['strict']/warped['N']:.1f}%\")

if __name__ == \"__main__\":
    main()
