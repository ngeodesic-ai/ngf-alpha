# ==============================================================================
# arc-benchmark-colab-v12m-lite.py
# Lightweight A/B benchmark for synthetic ARC-style op classification.
# - Letters (A–G) as labels -> op mapping
# - Multi-prompt contextual calibration (small K)
# - Target averaging over 2 phrasings
# - PCA fit on letter embeddings only (tiny)
# - Shorter symbolic loop, global repulsion
# - Prefilter OFF (realistic) and apples-to-apples Stock vs Warped
# ==============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

import random
import numpy as np
import torch
from typing import List, Tuple, Dict
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
MODEL_NAME = "gpt2"          # lighter; try "gpt2-medium" later
N_TASKS = 20                 # bump as needed

USE_PREFILTER = False
USE_CALIBRATION = True
CAL_K = 4                    # fewer neutral prompts
PRINT_TASKS = True

# Warped hyperparams (lighter)
ALPHA_WARPED = 0.90
SYMB_STEPS = 12
DT = 0.06
PULL_STRENGTH = 1.6
GAMMA = 0.3
LAMBDA_REP = 0.5             # global (precomputed) repulsion

# Use last layer for simplicity
NUDGE_LAYER = -1

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
    with torch.no_grad():
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
# Symbolic spring & logits
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

def nudged_logits_from_hidden(hidden_last: torch.Tensor,
                              target_reduced: np.ndarray) -> torch.Tensor:
    cur = hidden_last.detach().cpu().numpy()
    red = pca.transform(cur.reshape(1, -1)).squeeze()
    tgt = target_reduced - LAMBDA_REP * GLOBAL_ANTI   # global repulsion (cheap)
    nudged_red = symbolic_loop(red, tgt, steps=SYMB_STEPS, dt=DT)
    inv = pca.inverse_transform(nudged_red.reshape(1, -1)).squeeze()
    norm = np.linalg.norm(inv)
    if norm > 0:
        inv = (inv / norm) * 5.0
    hidden_t = torch.from_numpy(inv).to(device, torch.float32)
    return model.lm_head(hidden_t.unsqueeze(0)).squeeze(0)

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

@dataclass
class PromptCache:
    past: tuple
    logits: torch.Tensor
    hidden: torch.Tensor

def _init_prompt_cache(prompt_ids: torch.Tensor) -> PromptCache:
    with torch.no_grad():
        out = model(prompt_ids, output_hidden_states=True, use_cache=True)
    return PromptCache(
        past=out.past_key_values,
        logits=out.logits[:, -1, :].squeeze(0),
        hidden=out.hidden_states[NUDGE_LAYER][:, -1, :].squeeze(0),
    )

def score_letter_once(prompt_ids: torch.Tensor, letter: str, alpha: float) -> float:
    cache = _init_prompt_cache(prompt_ids)
    logits = cache.logits.clone()
    if alpha > 0:
        nudged = nudged_logits_from_hidden(cache.hidden, LETTER_TARGETS[letter])
        logits = (1 - alpha) * logits + alpha * nudged
    logp = torch.log_softmax(logits, dim=-1)
    return float(logp[LETTER_TOKEN[letter]].item())

# -----------------------------
# Multi-prompt calibration (small K)
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
        # one forward per neutral; then per-letter just mixes logits (fast)
        cache = _init_prompt_cache(pid)
        base_logp = torch.log_softmax(cache.logits, dim=-1)
        for L in LETTERS:
            if alpha > 0:
                nudged = nudged_logits_from_hidden(cache.hidden, LETTER_TARGETS[L])
                logp = torch.log_softmax((1 - alpha) * cache.logits + alpha * nudged, dim=-1)
            else:
                logp = base_logp
            sums[L] += float(logp[LETTER_TOKEN[L]].item())
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

        # score all letters (single-token, fast path)
        scores = []
        # reuse one forward cache; then mix nudges per letter
        cache = _init_prompt_cache(pid)
        base_logp = torch.log_softmax(cache.logits, dim=-1)
        for L in LETTERS:
            if alpha > 0:
                nudged = nudged_logits_from_hidden(cache.hidden, LETTER_TARGETS[L])
                logp = torch.log_softmax((1 - alpha) * cache.logits + alpha * nudged, dim=-1)
                s = float(logp[LETTER_TOKEN[L]].item())
            else:
                s = float(base_logp[LETTER_TOKEN[L]].item())
            # calibration
            s -= cal.get(L, 0.0)
            scores.append((L, s))
        scores.sort(key=lambda x: x[1], reverse=True)
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
        print(f"\n=== {tag} Summary ===")
        print(f"Prefilter used   : {False}")
        print(f"Unambiguous pairs: {unamb}")
        print(f"Ambiguous pairs  : {amb}")
        print(f"Equivalence Acc  : {eq_total}/{N} = {100*eq_total/N:.1f}%")
        print(f"Strict-label Acc : {strict_total}/{N} = {100*strict_total/N:.1f}%\n")

    return {"N": len(tasks), "eq": eq_total, "strict": strict_total}


# =============================
# Hyperparameter Sweep Utility
# =============================
from contextlib import contextmanager
from itertools import product
import csv
import time

# Centralize mutable hparams in one place for safe override/restore
HPARAM_KEYS = ["ALPHA_WARPED", "PULL_STRENGTH", "GAMMA", "SYMB_STEPS", "DT", "LAMBDA_REP"]

def _get_hparams():
    return {
        "ALPHA_WARPED": ALPHA_WARPED,
        "PULL_STRENGTH": PULL_STRENGTH,
        "GAMMA": GAMMA,
        "SYMB_STEPS": SYMB_STEPS,
        "DT": DT,
        "LAMBDA_REP": LAMBDA_REP,
    }

def _set_hparams(hd):
    global ALPHA_WARPED, PULL_STRENGTH, GAMMA, SYMB_STEPS, DT, LAMBDA_REP
    ALPHA_WARPED = hd["ALPHA_WARPED"]
    PULL_STRENGTH = hd["PULL_STRENGTH"]
    GAMMA = hd["GAMMA"]
    SYMB_STEPS = hd["SYMB_STEPS"]
    DT = hd["DT"]
    LAMBDA_REP = hd["LAMBDA_REP"]

@contextmanager
def override_hparams(**kwargs):
    """Temporarily override module-level hparams used by the nudge dynamics."""
    old = _get_hparams()
    try:
        merged = {**old, **kwargs}
        _set_hparams(merged)
        yield
    finally:
        _set_hparams(old)

def make_param_grid(grid_dict):
    """
    grid_dict: e.g.
      {
        "ALPHA_WARPED": [0.6, 0.9, 1.2],
        "PULL_STRENGTH": [1.2, 1.6, 2.0],
        "GAMMA": [0.2, 0.3, 0.4],
        "SYMB_STEPS": [12, 20, 32],
        "DT": [0.04, 0.06],
        "LAMBDA_REP": [0.3, 0.5, 0.8],
      }
    Returns list of dicts (one per combination).
    """
    keys = list(grid_dict.keys())
    vals = [grid_dict[k] for k in keys]
    combos = []
    for tup in product(*vals):
        combos.append({k: v for k, v in zip(keys, tup)})
    return combos

def run_warped_once(tasks, alpha=None):
    """
    Runs 'Warped' pass with provided alpha (if not None) using current globals.
    Returns dict with 'eq', 'strict', 'N'.
    """
    a = ALPHA_WARPED if (alpha is None) else float(alpha)
    return run_pass(tasks, alpha=a, tag="Warped")

def run_stock_once(tasks):
    """Runs 'Stock' (alpha=0.0) once. Returns dict with 'eq','strict','N'."""
    return run_pass(tasks, alpha=0.0, tag="Stock")

def sweep_nudge(
    param_grid_dict: dict,
    n_tasks: int = N_TASKS,
    seed: int = SEED,
    repeats: int = 2,
    save_csv_path: str = "sweep_results.csv",
    include_stock_baseline: bool = True,
    print_top_k: int = 10,
):
    """
    Perform a grid search over the provided param grid, reusing the same tasks for all configs.

    - param_grid_dict: dict of lists (see make_param_grid docstring)
    - n_tasks: tasks per run
    - seed: seed for reproducibility (same tasks across all configs)
    - repeats: average metrics over multiple runs per config
    - save_csv_path: where to store results
    - include_stock_baseline: optionally run and log one stock baseline on the same tasks
    - print_top_k: print best configs by equivalence acc

    Returns: list of results dicts
    """
    # Build tasks once for fairness
    tasks = make_tasks(n_tasks, seed=seed)

    # Optionally capture a baseline
    baselines = []
    if include_stock_baseline:
        print("\n>>> Running STOCK baseline (once) for reference...\n")
        stock_res = run_stock_once(tasks)
        stock_row = {
            "is_stock": True,
            "ALPHA_WARPED": 0.0,
            "PULL_STRENGTH": PULL_STRENGTH,
            "GAMMA": GAMMA,
            "SYMB_STEPS": SYMB_STEPS,
            "DT": DT,
            "LAMBDA_REP": LAMBDA_REP,
            "eq": stock_res["eq"],
            "strict": stock_res["strict"],
            "N": stock_res["N"],
            "eq_acc": stock_res["eq"] / stock_res["N"],
            "strict_acc": stock_res["strict"] / stock_res["N"],
            "repeats": 1,
            "seed": seed,
            "timestamp": time.time(),
        }
        baselines.append(stock_row)

    combos = make_param_grid(param_grid_dict)
    print(f"\n>>> Starting sweep over {len(combos)} configs; repeats={repeats}, n_tasks={n_tasks}\n")

    results = []
    for ci, cfg in enumerate(combos, start=1):
        # Ensure alpha present; default to current ALPHA_WARPED if omitted
        cfg_full = {**_get_hparams(), **cfg}
        print(f"[{ci}/{len(combos)}] cfg={cfg_full}")

        agg_eq = agg_strict = 0
        start_t = time.time()
        with override_hparams(**cfg_full):
            # If alpha is in cfg_full it will override ALPHA_WARPED inside the context
            for _ in range(repeats):
                warped = run_warped_once(tasks, alpha=cfg_full["ALPHA_WARPED"])
                agg_eq += warped["eq"]
                agg_strict += warped["strict"]

        dur = time.time() - start_t
        row = {
            "is_stock": False,
            **{k: cfg_full[k] for k in HPARAM_KEYS},
            "eq": agg_eq / repeats,
            "strict": agg_strict / repeats,
            "N": warped["N"],  # from last run; N invariant across repeats
            "eq_acc": (agg_eq / repeats) / warped["N"],
            "strict_acc": (agg_strict / repeats) / warped["N"],
            "repeats": repeats,
            "seed": seed,
            "sec_per_cfg": dur,
            "timestamp": time.time(),
        }
        results.append(row)

    # Write CSV
    fieldnames = [
        "is_stock",
        "ALPHA_WARPED", "PULL_STRENGTH", "GAMMA", "SYMB_STEPS", "DT", "LAMBDA_REP",
        "eq", "strict", "N", "eq_acc", "strict_acc",
        "repeats", "seed", "sec_per_cfg", "timestamp",
    ]
    with open(save_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in (baselines + results):
            writer.writerow(r)

    # Print quick leaderboard
    ranked = sorted(results, key=lambda x: (x["eq_acc"], x["strict_acc"]), reverse=True)
    print("\n=== Sweep Leaderboard (by eq_acc, then strict_acc) ===")
    for i, r in enumerate(ranked[:print_top_k], start=1):
        print(
            f"{i:02d}. eq={r['eq']:.1f}/{r['N']} ({100*r['eq_acc']:.1f}%) | "
            f"strict={r['strict']:.1f}/{r['N']} ({100*r['strict_acc']:.1f}%) | "
            f"alpha={r['ALPHA_WARPED']:.2f}, pull={r['PULL_STRENGTH']:.2f}, "
            f"gamma={r['GAMMA']:.2f}, steps={int(r['SYMB_STEPS'])}, dt={r['DT']:.3f}, "
            f"rep={r['LAMBDA_REP']:.2f}"
        )
    if include_stock_baseline and baselines:
        sb = baselines[0]
        print(
            f"\nStock baseline: eq={sb['eq']}/{sb['N']} ({100*sb['eq_acc']:.1f}%) | "
            f"strict={sb['strict']}/{sb['N']} ({100*sb['strict_acc']:.1f}%)"
        )

    print(f"\nSaved CSV -> {save_csv_path}")
    return baselines, results



# -----------------------------
# Main
# -----------------------------
def main():
    # Option 1: keep your side-by-side run
    tasks = make_tasks(N_TASKS, seed=SEED)
    stock  = run_pass(tasks, alpha=0.0, tag="Stock")
    warped = run_pass(tasks, alpha=ALPHA_WARPED, tag="Warped")

    print("=== Side-by-side Summary (same tasks) ===")
    print(f"N={stock['N']} | Prefilter={False} | Calibration={USE_CALIBRATION} "
          f"| Alpha_warped={ALPHA_WARPED} | NUDGE_LAYER={NUDGE_LAYER} | CAL_K={CAL_K}")
    print(f"Equivalence Acc  : Stock {stock['eq']}/{stock['N']} = {100*stock['eq']/stock['N']:.1f}% | "
          f"Warped {warped['eq']}/{warped['N']} = {100*warped['eq']/warped['N']:.1f}%")
    print(f"Strict-label Acc : Stock {stock['strict']}/{stock['N']} = {100*stock['strict']/stock['N']:.1f}% | "
          f"Warped {warped['strict']}/{warped['N']} = {100*warped['strict']/warped['N']:.1f}%")

    # Option 2: run a sweep (comment in to use)
    grid = {
        "ALPHA_WARPED":  [0.6, 0.9, 1.2],
        "PULL_STRENGTH": [1.2, 1.6, 2.0],
        "GAMMA":         [0.2, 0.3, 0.4],
        "SYMB_STEPS":    [12, 24, 36],
        "DT":            [0.04, 0.06],
        "LAMBDA_REP":    [0.3, 0.5, 0.8],
    }
    # Smaller n_tasks & repeats for quick runs; scale up once stable
    sweep_nudge(
        param_grid_dict=grid,
        n_tasks=30,
        seed=SEED,
        repeats=2,
        save_csv_path="sweep_results.csv",
        include_stock_baseline=True,
        print_top_k=10,
    )


if __name__ == "__main__":
    main()
