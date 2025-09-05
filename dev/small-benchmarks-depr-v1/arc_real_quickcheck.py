
import os, re, json, glob, random, argparse
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def find_arc_jsons(root: str, max_files: int) -> List[str]:
    pats = [
        os.path.join(root, "**", "*.json")
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    random.shuffle(files)
    return files[:max_files]

def grid_to_text(g: List[List[int]]) -> str:
    # row as comma-separated digits 0-9
    return "[" + ",".join("[" + ",".join(str(int(x)) for x in row) + "]" for row in g) + "]"

def render_arc_prompt(task: Dict[str, Any]) -> Tuple[str, Tuple[int, int]]:
    # Use first train pair as an example, then ask model to produce test output.
    # If no test given, we still ask for an output shaped like the last train output.
    train = task.get("train", [])
    test = task.get("test", [])
    if not train:
        raise ValueError("Task missing 'train' examples.")
    # choose first train pair for demonstration
    ex = train[0]
    tin = ex["input"]
    tout = ex["output"]
    in_text = grid_to_text(tin)
    out_text = grid_to_text(tout)
    # pick a test input to apply (fallback: reuse the same input)
    if test and isinstance(test[0], dict) and "input" in test[0]:
        target_in = test[0]["input"]
    else:
        target_in = tin
    target_shape = (len(target_in), len(target_in[0]))

    prompt = (
        "You are given an input grid and its transformed output grid.\n"
        "The transformation should be applied to a new input grid. "
        "Return ONLY the output grid in bracketed numeric form like [[a,b],[c,d]].\n\n"
        f"Example Input: {in_text}\n"
        f"Example Output: {out_text}\n\n"
        f"New Input: {grid_to_text(target_in)}\n"
        "New Output: "
    )
    return prompt, target_shape

GRID_RE = re.compile(r"\[\s*\[.*?\]\s*\]", re.S)

def extract_first_grid(text: str) -> Optional[List[List[int]]]:
    m = GRID_RE.search(text)
    if not m:
        return None
    frag = m.group(0)
    # keep only digits, commas, brackets and spaces
    safe = re.sub(r"[^0-9,\[\]\s-]", "", frag)
    # try to json-load after a light normalization
    try:
        arr = json.loads(safe)
        if (isinstance(arr, list) and arr and isinstance(arr[0], list) 
            and all(isinstance(x, list) for x in arr)):
            # ensure ints
            out = []
            for row in arr:
                out.append([int(v) for v in row])
            return out
    except Exception:
        return None
    return None

def grid_shape(g: List[List[int]]) -> Tuple[int,int]:
    if not g:
        return (0,0)
    return (len(g), len(g[0]) if isinstance(g[0], list) else 0)

def shape_matches(g: List[List[int]], target_shape: Tuple[int,int]) -> bool:
    if not g:
        return False
    r, c = target_shape
    if len(g) != r:
        return False
    if any(len(row) != c for row in g):
        return False
    return True

class EMASelector:
    def __init__(self, decay: float = 0.85, vocab: int = 50257):
        self.decay = decay
        self.state = None
        self.vocab = vocab
    def step(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: (vocab,)
        if self.state is None:
            self.state = logits.detach()
        else:
            self.state = self.decay * self.state + (1 - self.decay) * logits.detach()
        return self.state

def generate_once(model, tok, prompt: str, max_new_tokens=256, mode="off", ema_decay=0.85, eos_token_id=None):
    device = next(model.parameters()).device
    x = tok(prompt, return_tensors="pt").to(device)
    input_ids = x["input_ids"][0]
    ema = EMASelector(decay=ema_decay, vocab=model.config.vocab_size) if mode != "off" else None
    out_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        y = model(out_ids.unsqueeze(0))
        logits = y.logits[0, -1, :]  # (vocab,)
        if ema is not None:
            smoothed = ema.step(logits)
            next_id = int(torch.argmax(smoothed))
        else:
            next_id = int(torch.argmax(logits))
        out_ids = torch.cat([out_ids, torch.tensor([next_id], device=device)])
        if eos_token_id is not None and next_id == eos_token_id:
            break
        # quick stop if we see two closing brackets
        text_so_far = tok.decode(out_ids[len(input_ids):])
        if text_so_far.count(']') >= 2 and ']' in text_so_far and '[' in text_so_far:
            # give a few more tokens to close nicely
            if text_so_far.count(']') - text_so_far.count('[') >= 0 and text_so_far.strip().endswith(']'):
                break
    gen_text = tok.decode(out_ids[len(input_ids):])
    return gen_text

def run_sanity(root: str, model_name: str, max_files: int, mode: str, ema_decay: float, seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    files = find_arc_jsons(root, max_files)
    if not files:
        raise SystemExit(f"No ARC json files found under: {root}")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    valid = 0
    shape_ok = 0
    exact_match = 0  # only computed when target test output exists
    counted_exact = 0
    agree = 0
    N = 0

    for path in files:
        try:
            task = json.load(open(path, "r"))
        except Exception:
            continue
        try:
            prompt, tgt_shape = render_arc_prompt(task)
        except Exception:
            continue

        # Generate twice to measure self-consistency
        t1 = generate_once(model, tok, prompt, mode=mode, ema_decay=ema_decay, eos_token_id=tok.eos_token_id)
        t2 = generate_once(model, tok, prompt, mode=mode, ema_decay=ema_decay, eos_token_id=tok.eos_token_id)
        g1 = extract_first_grid(t1)
        g2 = extract_first_grid(t2)

        if g1 is not None:
            valid += 1
            if shape_matches(g1, tgt_shape):
                shape_ok += 1

        # self-consistency on parsed grids
        if g1 is not None and g2 is not None and g1 == g2:
            agree += 1

        # Optional exact match if a test output exists (not guaranteed)
        # If task has a known test output, use the *first* one.
        test = task.get("test", [])
        gt = None
        if test and isinstance(test[0], dict) and "output" in test[0]:
            gt = test[0]["output"]
        if gt is not None and g1 is not None:
            counted_exact += 1
            if g1 == gt:
                exact_match += 1

        N += 1

    eps = 1e-9
    metrics = {
        "files": N,
        "mode": mode,
        "ema_decay": ema_decay,
        "valid_rate": valid / (N + eps),
        "shape_match_rate": shape_ok / (N + eps),
        "self_consistency_rate": agree / (N + eps),
        "exact_match_rate_conditional": (exact_match / (counted_exact + eps)) if counted_exact > 0 else None,
        "exact_match_n": counted_exact,
    }
    print(json.dumps(metrics, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc_path", type=str, required=True, help="Folder containing ARC JSON files (training split recommended)")
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--max_files", type=int, default=50)
    ap.add_argument("--mode", type=str, choices=["off", "ema"], default="off",
                    help="Denoiser mode: 'off' or 'ema' (simple logits EMA)")
    ap.add_argument("--ema_decay", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_sanity(args.arc_path, args.model, args.max_files, args.mode, args.ema_decay, args.seed)

if __name__ == "__main__":
    main()
