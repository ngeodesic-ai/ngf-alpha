#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, math, random, re, time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- ARC-style task spec (programmatic, no .txt needed) ----------

@dataclass
class Task:
    id: str
    prompt: str
    truth: str           # exact string you expect
    task_type: str       # {"int","grid","text"}

def to_grid_str(grid):
    return "[[" + "],[".join(",".join(str(x) for x in row) for row in grid) + "]]"

def rot90(grid):
    # clockwise rotation
    rows, cols = len(grid), len(grid[0])
    return [[grid[rows-1-r][c] for r in range(rows)] for c in range(cols)]

def flip_h(grid):
    return [row[::-1] for row in grid]

def transpose(grid):
    return list(map(list, zip(*grid)))

def gen_arc_suite(seed=123) -> List[Task]:
    rng = random.Random(seed)
    tasks: List[Task] = []

    # INT arithmetic (exact-match integer)
    for a,b in [(13,50),(14,50),(16,48),(2, -99),(21,-49)]:
        p = f"Compute {a}+{b}. Return only the integer."
        t = str(a+b)
        tasks.append(Task(id=f"add_{a}_{b}", prompt=p, truth=t, task_type="int"))
    for a,b in [(7,8),(9,11),(12,12),(16,3)]:
        p = f"Compute {a}*{b}. Return only the integer."
        t = str(a*b)
        tasks.append(Task(id=f"mul_{a}_{b}", prompt=p, truth=t, task_type="int"))

    # Small grids (2x2, 2x3, 3x3) with rotate/flip/transpose
    grids = [
        ([[2,3],[4,5]], "2x2"),
        ([[1,2,3],[4,5,6]], "2x3"),
        ([[1,2,3],[4,5,6],[7,8,9]], "3x3"),
    ]
    for g,name in grids:
        g_rot = rot90(g)
        tasks.append(Task(
            id=f"rot90_{name}",
            prompt=f"Rotate {to_grid_str(g)} by 90 degrees clockwise. Return only the grid.",
            truth=to_grid_str(g_rot),
            task_type="grid"
        ))
        g_flip = flip_h(g)
        tasks.append(Task(
            id=f"flipH_{name}",
            prompt=f"Flip {to_grid_str(g)} horizontally. Return only the grid.",
            truth=to_grid_str(g_flip),
            task_type="grid"
        ))
        g_T = transpose(g)
        tasks.append(Task(
            id=f"transpose_{name}",
            prompt=f"Transpose {to_grid_str(g)}. Return only the grid.",
            truth=to_grid_str(g_T),
            task_type="grid"
        ))

    # Optional: a simple color-map pattern (stay numeric for exactness)
    src = [[0,1,1],[2,2,0]]
    dst = [[1,2,2],[0,0,1]]  # map: 0->1, 1->2, 2->0
    tasks.append(Task(
        id="colormap_2x3",
        prompt=f"Map values in {to_grid_str(src)} using 0→1, 1→2, 2→0. Return only the grid.",
        truth=to_grid_str(dst),
        task_type="grid"
    ))
    return tasks

# ---------- early-stop + sanitizer ----------

GRID_RE = re.compile(r"\[\s*\[.*?\]\s*\]", re.DOTALL)
INT_RE  = re.compile(r"[-+]?\d+")

def early_stop_hit(text: str, task_type: str) -> bool:
    if task_type == "int":
        return INT_RE.search(text) is not None
    if task_type == "grid":
        return GRID_RE.search(text) is not None
    return False

def extract_answer(text: str, task_type: str) -> str:
    if task_type == "int":
        m = INT_RE.search(text)
        return m.group(0) if m else text.strip()
    if task_type == "grid":
        m = GRID_RE.search(text)
        if not m: return re.sub(r"\s+","",text.strip())
        return re.sub(r"\s+","",m.group(0))
    return text.strip()

def exact_match(pred: str, truth: str, task_type: str) -> bool:
    if task_type == "grid":
        pred_c  = re.sub(r"\s+","",pred)
        truth_c = re.sub(r"\s+","",truth)
        return pred_c == truth_c
    return pred.strip() == truth.strip()

# ---------- greedy stock decode w/ KV cache ----------

@torch.no_grad()
def generate_greedy(model, tok, prompt: str, max_new: int, task_type: str, device: str) -> Tuple[str,str,int]:
    enc = tok(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    eos_id = tok.eos_token_id

    # prime text for incremental append (faster than re-decode entire sequence)
    generated = tok.decode(input_ids[0], skip_special_tokens=True)
    past = None
    steps = 0
    cur_ids = input_ids

    for _ in range(max_new):
        steps += 1
        out = model(input_ids=cur_ids[:, -1:], use_cache=True, past_key_values=past)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_id], dim=1)
        token_text = tok.decode(next_id[0], skip_special_tokens=True)
        generated += token_text

        if eos_id is not None and int(next_id[0,0]) == eos_id:
            break
        if early_stop_hit(generated[len(prompt):], task_type):
            break

    extracted = extract_answer(generated[len(prompt):], task_type)
    return generated, extracted, steps

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser("ARC text stock baseline (single script, no .txt)")
    ap.add_argument("--model", default="gpt2", help="HF model name, e.g., gpt2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_jsonl", default="generations_stock.jsonl")
    ap.add_argument("--metrics_out", default="metrics_stock.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()

    tasks = gen_arc_suite(seed=args.seed)

    n = len(tasks); ok = 0
    t0 = time.time()
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i, t in enumerate(tasks, 1):
            raw, ans, steps = generate_greedy(model, tok, t.prompt, args.max_new_tokens, t.task_type, args.device)
            hit = exact_match(ans, t.truth, t.task_type); ok += int(hit)
            f.write(json.dumps({
                "id": t.id, "prompt": t.prompt, "truth": t.truth, "task_type": t.task_type,
                "raw_text": raw, "answer_extracted": ans, "match": bool(hit), "steps": steps
            }) + "\n")
            if i % 10 == 0 or i == n:
                print(f"[{i}/{n}] acc_so_far={ok/i:.3f}")

    elapsed = time.time() - t0
    metrics = {"n": n, "correct": ok, "accuracy": ok / n if n else 0.0,
               "model": args.model, "max_new_tokens": args.max_new_tokens,
               "elapsed_sec": elapsed, "device": args.device}
    with open(args.metrics_out, "w") as mf:
        mf.write(json.dumps(metrics, indent=2))
    print(f"[DONE] Accuracy={metrics['accuracy']:.3f}  → {args.out_jsonl}, {args.metrics_out}")

if __name__ == "__main__":
    main()
