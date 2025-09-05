#!/usr/bin/env python3
"""
Real ARC sanity harness (fast + bounded)

What it does
------------
- Loads a handful of ARC JSON tasks (training-style) from a folder you point at
- Builds a minimal text prompt from the first train pair, applies it to the first test input
- Generates **twice** per task with a local HF model (greedy or sampled)
- Optional **logits EMA** smoothing across steps (stabilization)
- Computes sanity metrics: valid grid rate, shape match rate, self-consistency, and
  exact-match (only when a GT test output exists)

Why this exists
---------------
This is a clean, standalone script you can run without touching your v10e file.
It contains all the fixes we discussed: sampling flags, token budget, early stops,
no_grad, and a progress heartbeat so it never feels "hung".

Usage
-----
python3 stage11-real-arc-sanity.py \
  --arc_path ./arc_demo/training \
  --lm distilgpt2 \
  --max_files 20 \
  --text_sample 1 --temperature 0.7 --top_p 0.9 \
  --text_ema 1 --text_ema_decay 0.85 \
  --max_new_tokens 64

Tip: start with distilgpt2 on CPU for speed, then try gpt2.

python3 stage11-well-benchmark-v10e.py \
  --arc_path /Users/ian_moore/repos/ngf-alpha/manifold-warping/arc_demo/training \
  --lm distilgpt2 \
  --max_files 3 \
  --trials 10 \
  --text_sample 1 --temperature 0.6 --top_p 0.95 \
  --text_ema 1 --text_ema_decay 0.85 \
  --shape_guard 1 \
  --max_new_tokens 64
  --lm gpt2 --trials 10 --temperature 0.7 --top_p 0.9 --shape_guard 1

"""

from __future__ import annotations
import argparse, os, re, json, glob, random, sys
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------
# Argparse
# -----------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real ARC sanity harness")
    p.add_argument("--arc_path", type=str, required=True,
                   help="Folder containing ARC JSON files (training split recommended)")
    p.add_argument("--lm", type=str, default="gpt2", help="HF model name (e.g., distilgpt2, gpt2)")
    p.add_argument("--max_files", type=int, default=40, help="Max ARC files to sample")

    # decoding controls
    p.add_argument("--text_sample", type=int, choices=[0,1], default=0,
                   help="1 = use nucleus sampling; 0 = greedy")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling cumulative probability")
    p.add_argument("--top_k", type=int, default=0, help="Optional top-k cap (0=off)")
    p.add_argument("--max_new_tokens", type=int, default=64, help="Token budget per generation")

    # stabilization
    p.add_argument("--text_ema", type=int, choices=[0,1], default=0, help="Logits EMA smoothing on/off")
    p.add_argument("--text_ema_decay", type=float, default=0.85, help="EMA decay")

    # reproducibility
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json_constrained", type=int, choices=[0,1], default=0, help="Restrict decoding to JSON grid tokens [0-9,[] ]")
    p.add_argument("--shape_constrained", type=int, choices=[0,1], default=1,
    help="Force decoding to exactly the target grid shape (rows×cols)")
    p.add_argument(
    "--shape_guard", type=int, choices=[0, 1], default=1,
    help="Force decoding to exactly the target grid shape (rows×cols)"
    )
    p.add_argument("--trials", type=int, default=5,
    help="How many independent generations per task (>=2)")
    
    return p

# -----------------------
# ARC helpers
# -----------------------

def find_arc_jsons(root: str, k: int) -> List[str]:
    root = os.path.abspath(os.path.expanduser(root))
    files = glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    random.shuffle(files)
    return files[:k]

_GRID_RE = re.compile(r"\[\s*\[.*?\]\s*\]", re.S)

def grid_to_text(g: List[List[int]]) -> str:
    return "[" + ",".join("[" + ",".join(str(int(x)) for x in row) + "]" for row in g) + "]"

def render_arc_prompt(task: Dict[str, Any]) -> Tuple[str, Tuple[int,int]]:
    train = task.get("train", [])
    test = task.get("test", [])
    if not train:
        raise ValueError("Task missing 'train'.")
    ex = train[0]
    tin = ex["input"]; tout = ex["output"]
    if test and isinstance(test[0], dict) and "input" in test[0]:
        target_in = test[0]["input"]
    else:
        target_in = tin
    tgt_shape = (len(target_in), len(target_in[0]))

    prompt = (
        "You are given an input grid and its transformed output grid.\n"
        "Apply the same transformation to the new input and return ONLY the output grid "
        "as bracketed numbers like [[a,b],[c,d]]. Do not include any explanation.\n\n"
        f"Example Input: {grid_to_text(tin)}\n"
        f"Example Output: {grid_to_text(tout)}\n\n"
        f"New Input: {grid_to_text(target_in)}\n"
        "New Output: [["
    )
    return prompt, tgt_shape

def extract_first_grid(text: str) -> Optional[List[List[int]]]:
    m = _GRID_RE.search(text)
    if not m:
        return None
    frag = m.group(0)
    safe = re.sub(r"[^0-9,\[\]\s-]", "", frag)
    try:
        arr = json.loads(safe)
        if isinstance(arr, list) and arr and isinstance(arr[0], list):
            return [[int(v) for v in row] for row in arr]
    except Exception:
        return None
    return None

def shape_matches(grid: List[List[int]], shape: Tuple[int,int]) -> bool:
    if not grid: return False
    r,c = shape
    return len(grid)==r and all(isinstance(row, list) and len(row)==c for row in grid)

# -----------------------
# Decoding (greedy or sampled) with optional logits EMA
# -----------------------

# Optional whitelist of token IDs allowed during decoding (set in run())
ALLOWED_IDS = None

# Replace your build_whitelist(...) with this:
def build_whitelist(tok, allowed_chars: str = "0123456789,[]"):
    ids = []
    allowed = set(allowed_chars)
    for i in range(tok.vocab_size):
        s = tok.decode([i], clean_up_tokenization_spaces=False)
        if s in allowed:  # exact one-char match only (no spaces/newlines)
            ids.append(i)
    return torch.tensor(ids) if ids else None


class EMA:
    def __init__(self, decay: float):
        self.decay = float(decay)
        self.state = None
    def step(self, logits: torch.Tensor) -> torch.Tensor:
        if self.state is None:
            self.state = logits.detach()
        else:
            self.state = self.decay * self.state + (1.0 - self.decay) * logits.detach()
        return self.state

def sample_from_logits(logits: torch.Tensor, top_p: float, top_k: int, temperature: float) -> int:
    # temperature
    logits = logits / max(temperature, 1e-6)
    # optional top-k
    if top_k and top_k > 0:
        k = min(int(top_k), logits.numel())
        vals, idx = torch.topk(logits, k=k)
        masked = torch.full_like(logits, float("-inf")); masked[idx] = vals
        logits = masked
    # top-p nucleus
    probs = F.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sp, dim=-1)
        thr_idx = int((cdf >= top_p).nonzero(as_tuple=True)[0][0]) if torch.any(cdf >= top_p) else sp.numel() - 1
        keep = si[:thr_idx+1]
        filt = torch.zeros_like(probs); filt[keep] = probs[keep]
        probs = filt / (filt.sum() + 1e-12)
    return int(torch.multinomial(probs, 1)[0])

import torch
import torch.nn.functional as F

def _map_symbol_tokens(tok):
    """
    Build maps for exact single-character tokens for the alphabet: 0-9, ',', '[', ']'.
    Returns:
      char2ids: dict char -> list[token_id]
      id2char:  dict token_id -> char
      digit_ids: flat list of all token ids that decode to a single digit.
    """
    alphabet = list("0123456789,[]")
    char2ids, id2char = {}, {}
    # Scan vocab once (OK on GPT-2; ~50k ids)
    for tid in range(tok.vocab_size):
        s = tok.decode([tid], clean_up_tokenization_spaces=False)
        if s in alphabet:
            char2ids.setdefault(s, []).append(tid)
            id2char[tid] = s
    # Fallback if any char missing (shouldn't happen on GPT-2)
    for ch in alphabet:
        if ch not in char2ids:
            enc = tok.encode(ch, add_special_tokens=False)
            enc = [i for i in enc if tok.decode([i], clean_up_tokenization_spaces=False) == ch]
            if not enc:
                raise RuntimeError(f"No token that decodes to '{ch}'")
            char2ids[ch] = enc
            for i in enc: id2char[i] = ch
    digit_ids = [i for d in "0123456789" for i in char2ids[d]]
    return char2ids, id2char, digit_ids

def _expected_shape_symbols(rows:int, cols:int):
    """
    Return the sequence of symbols (from { '[', ']', ',', 'd' }) that exactly
    describes an r x c grid like [[[d,d,...],[...]], ... ].
    """
    seq = ['[', '[']
    for r in range(rows):
        seq.append('[')
        for c in range(cols):
            seq.append('d')
            if c < cols-1: seq.append(',')
        seq.append(']')
        if r < rows-1: seq += [',', '[']
    seq.append(']')
    return seq

def _map_symbol_tokens(tok):
    """
    Exact single-char tokens for alphabet: 0-9 , [ ]
    Returns:
      char2ids: dict char -> list[token_id]
      id2char:  dict token_id -> char
      digit_ids: list of token_ids for digits
    """
    alphabet = list("0123456789,[]")
    char2ids, id2char = {}, {}
    for tid in range(tok.vocab_size):
        s = tok.decode([tid], clean_up_tokenization_spaces=False)
        if s in alphabet:
            char2ids.setdefault(s, []).append(tid)
            id2char[tid] = s
    # ensure coverage
    for ch in alphabet:
        if ch not in char2ids:
            enc = tok.encode(ch, add_special_tokens=False)
            enc = [i for i in enc if tok.decode([i], clean_up_tokenization_spaces=False) == ch]
            if not enc:
                raise RuntimeError(f"No token that decodes to '{ch}'")
            char2ids[ch] = enc
            for i in enc: id2char[i] = ch
    digit_ids = [i for d in "0123456789" for i in char2ids[d]]
    return char2ids, id2char, digit_ids

def _expected_shape_symbols(rows:int, cols:int):
    """
    Sequence over symbols { '[', ']', ',', 'd' } for an r×c grid.
    Example for 2×2: [[ [ d , d ] , [ d , d ] ]]
    """
    seq = ['[', '[']
    for r in range(rows):
        seq.append('[')
        for c in range(cols):
            seq.append('d')
            if c < cols-1: seq.append(',')
        seq.append(']')
        if r < rows-1: seq += [',', '[']
    seq.append(']')
    return seq

def _expected_after_seed(rows: int, cols: int):
    """
    We already seeded '[[' into the model context AND into out_chars.
    This returns the remaining symbol sequence over { 'd', ',', '[', ']' } to
    produce exactly an r×c grid: [[ <row0> , <row1> , ... ]]
    """
    seq = []
    for r in range(rows):
        # row r: digits with commas between cells
        for c in range(cols):
            seq.append('d')
            if c < cols - 1:
                seq.append(',')
        # close this row
        seq.append(']')
        # open next row if needed: ",["
        if r < rows - 1:
            seq += [',', '[']
    # close outer list
    seq.append(']')
    return seq



def _map_chars(tok):
    # byte-level BPE => single-char tokens exist for these
    chars = "0123456789,[]"
    ch2id = {}
    for ch in chars:
        ids = tok.encode(ch, add_special_tokens=False)
        # prefer single-token encodings; for GPT-2 this is true for these chars
        ch2id[ch] = ids[0]
    return ch2id, [ch2id[str(d)] for d in range(10)]

def _expected_shape_sequence(rows:int, cols:int):
    # Serialized target pattern over the alphabet { '[', ']', ',', 'd' }
    seq = ['[', '[']  # we always start "[["
    for i in range(rows):
        seq.append('[')
        for j in range(cols):
            seq.append('d')
            if j < cols-1:
                seq.append(',')
        seq.append(']')
        if i < rows-1:
            seq += [',','[']
    seq.append(']')
    return seq  # e.g., [[ [ d , d ] , [ d , d ] ]]

def _compress_symbols(s: str):
    # keep only digits and bracket/comma, map digits→'d'
    keep = ''.join(ch for ch in s if ch in '0123456789,[]')
    return ''.join(('d' if ch.isdigit() else ch) for ch in keep)

def generate_once_shape_constrained(
    model, tok, prompt: str, rows:int, cols:int,
    max_new_tokens:int = 64,
    use_sampling: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 0,
    use_ema: bool = True,
    ema_decay: float = 0.85,
    eos_token_id: int | None = None,
) -> str:
    """
    Emit exactly an r×c bracketed grid by forcing the next symbol class and
    masking logits to allowed token IDs. Still uses the model's probabilities
    (greedy or sampled) within the allowed set.
    Returns the GENERATED GRID STRING (no prompt).
    """
    device = next(model.parameters()).device
    x = tok(prompt, return_tensors="pt")
    input_ids = x["input_ids"][0].to(device)
    out_ids = input_ids.clone()

    # strong priming: seed "[[" into context
    seed_ids = tok.encode("[[", add_special_tokens=False)
    if seed_ids:
        out_ids = torch.cat([out_ids, torch.tensor(seed_ids, device=device)])

    char2ids, id2char, digit_ids = _map_symbol_tokens(tok)
    expected = _expected_after_seed(rows, cols)   # <-- was _expected_shape_symbols(...)
    L = len(expected)

    class EMA:
        def __init__(self, decay): self.d=decay; self.s=None
        def step(self, z): self.s = z.detach() if self.s is None else self.d*self.s + (1-self.d)*z.detach(); return self.s
    ema = EMA(ema_decay) if use_ema else None

    def _sample(z):
        z = z / max(temperature, 1e-6)
        if top_k and top_k > 0:
            k = min(int(top_k), z.numel())
            vals, idx = torch.topk(z, k=k)
            mask = torch.full_like(z, float("-inf")); mask[idx] = vals
            z = mask
        p = F.softmax(z, dim=-1)
        if 0.0 < top_p < 1.0:
            sp, si = torch.sort(p, descending=True)
            cdf = torch.cumsum(sp, dim=-1)
            k = int((cdf >= top_p).nonzero(as_tuple=True)[0][0]) if torch.any(cdf >= top_p) else sp.numel()-1
            keep = si[:k+1]
            filt = torch.zeros_like(p); filt[keep] = p[keep]
            p = filt / (filt.sum() + 1e-12)
        return int(torch.multinomial(p, 1)[0])

    out_chars: list[str] = ['[', '[']   # <-- add this

    for t in range(min(int(max_new_tokens), L)):
        with torch.no_grad():
            y = model(out_ids.unsqueeze(0))
            logits = y.logits[0, -1, :]

        if ema is not None:
            logits = ema.step(logits)

        sym = expected[t]
        allowed = digit_ids if sym == 'd' else char2ids[sym]

        # hard mask to allowed set (fallback to '0' if numeric mask is degenerate)
        masked = torch.full_like(logits, float("-inf"))
        idx = torch.tensor(allowed, device=logits.device, dtype=torch.long)
        masked[idx] = logits[idx]
        if sym == 'd' and not torch.isfinite(masked).any():
            # extremely defensive: allow '0'
            masked = torch.full_like(logits, float("-inf"))
            masked[char2ids['0'][0]] = 0.0
        logits = masked

        nxt = _sample(logits) if use_sampling else int(torch.argmax(logits))
        ch = id2char.get(nxt)
        if ch is None:
            ch = '0' if sym == 'd' else sym
        out_chars.append(ch)
        out_ids = torch.cat([out_ids, torch.tensor([nxt], device=device)])

        if eos_token_id is not None and nxt == int(eos_token_id):
            break

    return "".join(out_chars)




def generate_once(
    model, tok, prompt: str,
    max_new_tokens: int = 64,
    use_sampling: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    use_ema: bool = False,
    ema_decay: float = 0.85,
    eos_token_id: Optional[int] = None,
) -> str:
    device = next(model.parameters()).device
    x = tok(prompt, return_tensors="pt")
    input_ids = x["input_ids"][0].to(device)
    out_ids = input_ids.clone()

    # Strongly bias generation toward a bracketed grid by seeding an opening bracket
    bracket_ids = tok.encode("[[", add_special_tokens=False)
    if bracket_ids:
        out_ids = torch.cat([out_ids, torch.tensor(bracket_ids, device=device)])

    ema = EMA(ema_decay) if use_ema else None

    for t in range(int(max_new_tokens)):
        with torch.no_grad():
            y = model(out_ids.unsqueeze(0))
            logits = y.logits[0, -1, :]
        if ema is not None:
            logits = ema.step(logits)
        # If a whitelist is configured, mask logits to allowed token IDs only
        if 'ALLOWED_IDS' in globals() and ALLOWED_IDS is not None:
            aid = ALLOWED_IDS.to(logits.device)
            masked = torch.full_like(logits, float('-inf'))
            masked[aid] = logits[aid]
            logits = masked
        if use_sampling:
            next_id = sample_from_logits(logits, top_p=top_p, top_k=top_k, temperature=temperature)
        else:
            next_id = int(torch.argmax(logits))
        out_ids = torch.cat([out_ids, torch.tensor([next_id], device=device)])
        if eos_token_id is not None and next_id == int(eos_token_id):
            break
        # early stops
        gen_text = tok.decode(out_ids[len(input_ids):], skip_special_tokens=True)
        txt = gen_text.strip()
        # If brackets appear balanced and we end with ']', stop early
        opens = txt.count('[')
        closes = txt.count(']')
        if opens > 0 and closes >= opens and txt.endswith(']'):
            break
        if txt.endswith("]] ") or txt.endswith("]] ") or txt.endswith("]] "):
            break
        if txt.endswith("]]"):
            break
        if t > 64 and "[" not in txt:
            break
    return tok.decode(out_ids[len(input_ids):], skip_special_tokens=True)

# -----------------------
# Main runner
# -----------------------

def run(args):
    # --- imports (safe if already imported above) ---
    import os, glob, json, random
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # --- seeds ---
    seed = getattr(args, "seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # --- resolve ARC path & collect files ---
    arc_path = getattr(args, "arc_path", None) or getattr(args, "real_arc_path", None)
    if not arc_path:
        raise SystemExit("--arc_path (or --real_arc_path alias) is required")

    arc_root = os.path.abspath(os.path.expanduser(arc_path))
    files = [p for p in glob.glob(os.path.join(arc_root, "**", "*.json"), recursive=True) if os.path.isfile(p)]
    max_files = int(getattr(args, "max_files", len(files)))
    files = files[:max_files]

    print(f"[real-arc] scanning: {arc_root}", flush=True)
    print(f"[real-arc] found {len(files)} json files", flush=True)
    if not files:
        raise SystemExit(f"No ARC JSON found under: {arc_root}")

    # --- tokenizer & model ---
    tok = AutoTokenizer.from_pretrained(args.lm)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.lm)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- optional strict JSON whitelist (only digits/comma/brackets) ---
    if getattr(args, "json_constrained", 0) == 1:
        allowed = set("0123456789,[]")
        ids = []
        for tid in range(tok.vocab_size):
            s = tok.decode([tid], clean_up_tokenization_spaces=False)
            if s in allowed:  # exact one-char match only
                ids.append(tid)
        globals()["ALLOWED_IDS"] = torch.tensor(ids) if len(ids) > 0 else None
        print(f"[real-arc] json_constrained=1 allowed_ids={len(ids)} tokens", flush=True)

    # --- accumulators ---
    N = 0
    valid_sum = 0.0
    shape_ok_sum = 0.0
    consistency_pairs_sum = 0.0
    exact_match = 0
    counted_exact = 0

    trials = int(getattr(args, "trials", 2))
    use_shape_guard = int(getattr(args, "shape_guard", getattr(args, "shape_constrained", 1))) == 1

    # --- main per-task loop ---
    for i, path in enumerate(files):
        if i % 5 == 0:
            print(f"[real-arc] progress {i}/{len(files)}", flush=True)

        # load task
        try:
            with open(path, "r") as f:
                task = json.load(f)
        except Exception:
            continue

        # prompt + target shape
        try:
            prompt, tgt_shape = render_arc_prompt(task)
        except Exception:
            continue

        # multi-trial generation
        outs = []
        for _ in range(trials):
            if use_shape_guard:
                outs.append(
                    generate_once_shape_constrained(
                        model, tok, prompt, tgt_shape[0], tgt_shape[1],
                        max_new_tokens=getattr(args, "max_new_tokens", 64),
                        use_sampling=bool(getattr(args, "text_sample", 0)),
                        temperature=getattr(args, "temperature", 0.7),
                        top_p=getattr(args, "top_p", 0.9),
                        top_k=getattr(args, "top_k", 0),
                        use_ema=bool(getattr(args, "text_ema", 0)),
                        ema_decay=getattr(args, "text_ema_decay", 0.85),
                        eos_token_id=tok.eos_token_id,
                    )
                )
            else:
                outs.append(
                    generate_once(
                        model, tok, prompt,
                        max_new_tokens=getattr(args, "max_new_tokens", 64),
                        use_sampling=bool(getattr(args, "text_sample", 0)),
                        temperature=getattr(args, "temperature", 0.7),
                        top_p=getattr(args, "top_p", 0.9),
                        top_k=getattr(args, "top_k", 0),
                        use_ema=bool(getattr(args, "text_ema", 0)),
                        ema_decay=getattr(args, "text_ema_decay", 0.85),
                        eos_token_id=tok.eos_token_id,
                    )
                )

        # parse and score this task
        grids = [extract_first_grid(o) for o in outs]

        valid_count = sum(g is not None for g in grids)
        shape_ok_count = sum(g is not None and shape_matches(g, tgt_shape) for g in grids)

        # pairwise self-consistency across trials
        pairs_equal, total_pairs = 0, 0
        for a in range(len(grids)):
            for b in range(a + 1, len(grids)):
                total_pairs += 1
                if grids[a] is not None and grids[b] is not None and grids[a] == grids[b]:
                    pairs_equal += 1
        task_pairwise = (pairs_equal / total_pairs) if total_pairs else 0.0

        # accumulate
        valid_sum += valid_count / max(1, trials)
        shape_ok_sum += shape_ok_count / max(1, trials)
        consistency_pairs_sum += task_pairwise

        # exact-match where GT exists (use simple majority vote among valid trials)
        test = task.get("test", [])
        gt = test[0]["output"] if (test and isinstance(test[0], dict) and "output" in test[0]) else None
        if gt is not None:
            counted_exact += 1
            cands = [g for g in grids if g is not None]
            if cands:
                # majority vote
                keys = ["::".join(",".join(map(str, row)) for row in g) for g in cands]
                from collections import Counter
                best_key, _ = Counter(keys).most_common(1)[0]
                pred = next(g for g in cands if "::".join(",".join(map(str, row)) for row in g) == best_key)
            else:
                pred = None
            if pred == gt:
                exact_match += 1

        N += 1

    # --- final metrics ---
    eps = 1e-9
    result = {
        "files": N,
        "model": getattr(args, "lm", "gpt2"),
        "text_sample": bool(getattr(args, "text_sample", 0)),
        "temperature": getattr(args, "temperature", 0.7),
        "top_p": getattr(args, "top_p", 0.9),
        "top_k": getattr(args, "top_k", 0),
        "text_ema": bool(getattr(args, "text_ema", 0)),
        "text_ema_decay": getattr(args, "text_ema_decay", 0.85),
        "max_new_tokens": getattr(args, "max_new_tokens", 64),
        "trials": trials,
        # means across tasks
        "valid_rate_mean": valid_sum / max(1, N),
        "shape_match_rate_mean": shape_ok_sum / max(1, N),
        "self_consistency_pairwise": consistency_pairs_sum / max(1, N),
        # conditional exact match
        "exact_match_rate_conditional": (exact_match / (counted_exact + eps)) if counted_exact > 0 else None,
        "exact_match_n": counted_exact,
    }
    print("[REAL-ARC] " + json.dumps(result, indent=2))



if __name__ == "__main__":
    run(build_argparser().parse_args())
