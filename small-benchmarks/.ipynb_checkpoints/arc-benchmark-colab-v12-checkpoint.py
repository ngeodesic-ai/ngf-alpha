# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Step 10 Benchmark (v12): CPU-only operation classifier + executor for synthetic ARC tasks
# - Model outputs ONE label from {A..G}; we deterministically execute the transform.
# - Supports: rotate90, flip_h, flip_v, scale2, rotate_then_flip, swap_minmax, shift_down
# - Few-shot prompt; constrained decoding over A..G; continuation-only parsing (no leakage)
# ==============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ensure CPU-only before importing torch

import random, numpy as np, torch, re
from typing import List, Tuple
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 43
random.seed(SEED)
np.random.seed(SEED)
# (intentionally not calling torch.manual_seed to avoid CUDA seeding in some envs)

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
# Operations (deterministic executor)
# -----------------------------
def rotate90(grid: List[List[int]]) -> List[List[int]]:
    # 90° clockwise rotation for rectangular grids
    return [list(col) for col in zip(*grid[::-1])]

def flip_h(grid: List[List[int]]) -> List[List[int]]:
    # Horizontal flip (mirror left<->right)
    return [row[::-1] for row in grid]

def flip_v(grid: List[List[int]]) -> List[List[int]]:
    # Vertical flip (top<->bottom)
    return grid[::-1]

def scale2(grid: List[List[int]]) -> List[List[int]]:
    return [[v * 2 for v in row] for row in grid]

def rotate_then_flip(grid: List[List[int]]) -> List[List[int]]:
    # Define as rotate90 then horizontal flip (match common "rotate then mirror" pattern)
    return flip_h(rotate90(grid))

def swap_minmax(grid: List[List[int]]) -> List[List[int]]:
    flat = [v for row in grid for v in row]
    mn, mx = min(flat), max(flat)
    out = []
    for row in grid:
        out.append([mx if v == mn else (mn if v == mx else v) for v in row])
    return out

def shift_down(grid: List[List[int]]) -> List[List[int]]:
    # Circular shift rows down by 1 (bottom row moves to top)
    if not grid:
        return grid
    return [grid[-1]] + grid[:-1]

OP_MAP = {
    'A': 'rotate90',
    'B': 'flip_h',
    'C': 'flip_v',
    'D': 'scale2',
    'E': 'rotate_then_flip',
    'F': 'swap_minmax',
    'G': 'shift_down',
}
EXECUTOR = {
    'rotate90': rotate90,
    'flip_h': flip_h,
    'flip_v': flip_v,
    'scale2': scale2,
    'rotate_then_flip': rotate_then_flip,
    'swap_minmax': swap_minmax,
    'shift_down': shift_down,
}

# -----------------------------
# Few-shot prompt (no leakage). Model must answer with ONE letter A..G.
# -----------------------------
FEWSHOT = (
    "You are given an input grid of digits (size 2x2 or 3x3).\n"
    "Choose exactly ONE operation label that maps the input to the correct output:\n"
    "A=rotate90, B=flip_h, C=flip_v, D=scale2, E=rotate_then_flip, F=swap_minmax, G=shift_down.\n"
    "Return ONLY the single letter (A..G), nothing else.\n\n"
    "Examples:\n"
    "Input: [[1,2],[3,4]] -> Output: [[3,1],[4,2]] => A\n"
    "Input: [[5,6],[7,8]] -> Output: [[6,5],[8,7]] => B\n"
    "Input: [[1,2,3],[4,5,6],[7,8,9]] -> Output: [[7,8,9],[4,5,6],[1,2,3]] => C\n"
    "Input: [[2,3],[4,5]] -> Output: [[4,6],[8,10]] => D\n"
    "Input: [[1,2],[3,4]] -> Output: [[3,4],[1,2]] => E\n"
    "Input: [[1,9],[2,3]] -> Output: [[9,1],[2,3]] => F\n"
    "Input: [[1,2,3],[4,5,6]] -> Output: [[4,5,6],[1,2,3]] => G\n"
)

# -----------------------------
# Constrained single-letter decoding (A..G)
# -----------------------------
def letter_token_ids(letter: str) -> List[int]:
    # Return token ids that decode exactly to the single uppercase letter
    ids = []
    for t in range(vocab_size):
        s = tokenizer.decode([t], clean_up_tokenization_spaces=False)
        if s == letter:
            ids.append(t)
    # Fallback: accept encode letter and filter exact match
    if not ids:
        for t in tokenizer.encode(letter, add_special_tokens=False):
            s = tokenizer.decode([t], clean_up_tokenization_spaces=False)
            if s == letter:
                ids.append(t)
    return ids

LETTER_IDS = {L: letter_token_ids(L) for L in list("ABCDEFG")}
ALLOWED_IDS = sorted(set([tid for ids in LETTER_IDS.values() for tid in ids]))

def constrained_letter_decode(prompt: str, max_steps: int = 8) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    prompt_len = input_ids.shape[1]

    generated = input_ids
    past = None
    picked_letter = None

    for _ in range(max_steps):
        with torch.no_grad():
            if past is None:
                out = model(generated, output_hidden_states=False, use_cache=True)
            else:
                out = model(generated[:, -1:], past_key_values=past, output_hidden_states=False, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].clone()

        # Mask to allowed letter tokens only
        mask = torch.full_like(logits, -1e9)
        mask[..., ALLOWED_IDS] = 0.0
        logits = logits + mask

        next_tok = torch.argmax(logits, dim=-1).unsqueeze(0)
        generated = torch.cat([generated, next_tok], dim=1)

        # Inspect continuation only
        cont = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
        for L in "ABCDEFG":
            if L in cont:
                picked_letter = L
                break
        if picked_letter:
            break

    return picked_letter or "A"  # default to 'A' if nothing found

# -----------------------------
# Synthetic task generator (covers 2x2 and 3x3; all ops)
# -----------------------------
def random_grid(h: int, w: int) -> List[List[int]]:
    return [[random.randint(1, 9) for _ in range(w)] for _ in range(h)]

OPS = list(OP_MAP.values())

def generate_task() -> Tuple[List[List[int]], str, List[List[int]]]:
    h = random.choice([2,3]); w = random.choice([2,3])
    grid = random_grid(h, w)
    op_name = random.choice(OPS)
    out = EXECUTOR[op_name](grid)
    return grid, op_name, out

# -----------------------------
# Benchmark
# -----------------------------
def run_benchmark(N: int = 20):
    correct = 0
    for i in range(1, N+1):
        grid, op_name, expected = generate_task()
        prompt = FEWSHOT + f"\nInput: {grid} -> Output: {expected}\nAnswer:"  # few-shot format, no leakage for current test (we pass only input & ask for op label)
        # Wait – to keep it blind, we must NOT include the expected output for the current test.
        # Use a classification prompt that shows ONLY the input and asks for the operation.
        prompt = FEWSHOT + f"\nClassify the operation for this pair. Input: {grid}\nAnswer:"

        L = constrained_letter_decode(prompt)
        pred_op = OP_MAP.get(L, 'rotate90')
        realized = EXECUTOR[pred_op](grid)
        ok = (realized == expected)
        correct += int(ok)
        print(f"Task {i:02d} | Op={op_name} | Input {grid} | Expected {expected} | Pred {pred_op} via {L} | "
              f"{'✓' if ok else '×'}")
    print("\n=== Summary ===")
    print(f"Accuracy: {correct}/{N} = {100*correct/N:.1f}%")

if __name__ == "__main__":
    run_benchmark(20)
