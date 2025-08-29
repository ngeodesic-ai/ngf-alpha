# arc_benchmark_symbolic_nudge_fusion.py
# ------------------------------------------------------------
# Minimal, fast ARC-style op-class benchmark with:
#  - per-class calibration bias (cheap, per seed),
#  - dynamic alpha for ambiguous families,
#  - late fusion (stock + warped) with separate betas for unamb/amb.
# You can paste-and-run as-is (uses DummyScorer),
# or swap in your own LLM scorer by implementing Scorer.score_labels().
# ------------------------------------------------------------

from __future__ import annotations
import math, random, statistics
from typing import List, Tuple, Dict, Callable, Any

# ---------- Ops & utilities ----------

OpName = str
Grid = List[List[int]]

OPS: List[OpName] = [
    "rotate90",        # A
    "flip_h",          # B
    "flip_v",          # C
    "scale2",          # D
    "rotate_then_flip",# E (rotate90 -> flip_h)
    "swap_minmax",     # F (global min<->max)
    "shift_down",      # G (cyclic row shift by 1)
]
LETTER = { # for compact logs
    "rotate90":"A","flip_h":"B","flip_v":"C","scale2":"D",
    "rotate_then_flip":"E","swap_minmax":"F","shift_down":"G"
}
IDX = {op:i for i,op in enumerate(OPS)}

def deepcopy_grid(g: Grid)->Grid:
    return [row[:] for row in g]

def rotate90(g: Grid)->Grid:
    n, m = len(g), len(g[0])
    out = [[0]*n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            out[i][j] = g[n-1-j][i]
    return out

def flip_h(g: Grid)->Grid:
    return [row[::-1] for row in g]

def flip_v(g: Grid)->Grid:
    return g[::-1]

def scale2(g: Grid)->Grid:
    return [[v*2 for v in row] for row in g]

def rotate_then_flip(g: Grid)->Grid:
    return flip_h(rotate90(g))

def swap_minmax(g: Grid)->Grid:
    flat = [v for row in g for v in row]
    mn, mx = min(flat), max(flat)
    def f(v): 
        if v==mn: return mx
        if v==mx: return mn
        return v
    return [[f(v) for v in row] for row in g]

def shift_down(g: Grid)->Grid:
    return [g[-1]] + g[:-1]

OP_IMPL: Dict[OpName, Callable[[Grid], Grid]] = {
    "rotate90": rotate90,
    "flip_h": flip_h,
    "flip_v": flip_v,
    "scale2": scale2,
    "rotate_then_flip": rotate_then_flip,
    "swap_minmax": swap_minmax,
    "shift_down": shift_down,
}

def grids_equal(a: Grid, b: Grid)->bool:
    return a==b

def pretty_grid(g: Grid)->str:
    return "[" + ", ".join(str(r) for r in g) + "]"

# ---------- Synthetic task generator (small & quick) ----------

def gen_task(rng: random.Random)->Tuple[OpName, Grid, Grid]:
    # small shapes to mimic ARC toy
    n = rng.choice([2,3])
    m = rng.choice([2,3])
    g = [[rng.randint(1,9) for _ in range(m)] for _ in range(n)]
    op = rng.choice(OPS)
    y = OP_IMPL[op](g)
    return op, g, y

def gen_tasks(N:int, seed:int)->List[Tuple[OpName, Grid, Grid]]:
    rng = random.Random(seed)
    return [gen_task(rng) for _ in range(N)]

# ---------- Ambiguity analysis ----------

def equivalent_ops_for_input(inp: Grid, true_out: Grid)->List[OpName]:
    # ops that yield exactly the same output as the expected one (for THIS input)
    cands = []
    for op in OPS:
        out = OP_IMPL[op](inp)
        if grids_equal(out, true_out):
            cands.append(op)
    return cands

# ---------- Scorer interface (plug in your model here) ----------

class Scorer:
    """Replace DummyScorer with a wrapper around your real model.
    Must return a list of raw logits (len=|OPS|), *un-normalized*.
    mode: 'stock' or 'warped'. alpha, layer may be used by your impl."""
    def __init__(self, seed:int):
        self.seed = seed
        self.rng = random.Random(seed)

    def score_labels(self, context:str, labels:List[OpName], *, mode:str="stock",
                     alpha:float=0.0, layer:int=-1)->List[float]:
        raise NotImplementedError

class DummyScorer(Scorer):
    """A lightweight simulator with biased letter priors (to test the harness)."""
    def __init__(self, seed:int):
        super().__init__(seed)
        # letter priors (imitate 'G' & 'D' attraction you've seen)
        base = { "A":-0.2, "B":-0.1, "C": 0.1, "D": 0.3, "E":-0.05, "F":-0.15, "G": 0.45 }
        self.bias = [base[LETTER[op]] for op in OPS]

    def score_labels(self, context:str, labels:List[OpName], *, mode:str="stock",
                     alpha:float=0.0, layer:int=-1)->List[float]:
        # logit = prior + small context noise; "warped" adds a nudge
        r = self.rng
        noise = [r.uniform(-0.03, 0.03) for _ in labels]
        logits = [b + n for b,n in zip(self.bias, noise)]
        if mode=="warped" and alpha>0:
            # toy nudge: magnify differences by alpha toward the first op token present in context
            # (real impl would use your v12e nudge; this is just to exercise the pipeline)
            # push the op mentioned in context, if any
            for i,op in enumerate(labels):
                if op in context:
                    logits[i] += 0.15*alpha
        return logits

# ---------- Calibration (per-class bias) ----------

CAL_K = 12  # still fast; improves stability
CAL_PROMPTS = [
    "Operation: {}", "Op: {}", "Apply: {}", "Transform: {}",
    "Label: {}", "Task op = {}", "Answer op: {}", "Pick: {}",
]

def build_cal_bias(scorer:Scorer, label_space:List[OpName], rng:random.Random)->List[float]:
    acc = [0.0]*len(label_space)
    for _ in range(CAL_K):
        prompt = rng.choice(CAL_PROMPTS)
        logits = scorer.score_labels(prompt, label_space, mode="stock", alpha=0.0, layer=-1)
        c = sum(logits)/len(logits)
        for i,v in enumerate(logits):
            acc[i] += (v - c)
    return [a/float(CAL_K) for a in acc]

def apply_calibration(logits:List[float], bias:List[float])->List[float]:
    return [x - b for x,b in zip(logits, bias)]

def top2_margin(logits:List[float])->float:
    s = sorted(logits, reverse=True)
    return (s[0] - s[1]) if len(s)>=2 else 0.0

# ---------- Dynamic alpha for ambiguous families ----------

ALPHA_BASE   = 0.30
ALPHA_AMBIG  = 0.90
AMBIG_FAMS = [
    {"flip_v","shift_down"},
    {"rotate90","rotate_then_flip"},
    {"flip_h","swap_minmax"},
]

def dynamic_alpha(cands:List[OpName])->float:
    S = set(cands)
    for fam in AMBIG_FAMS:
        if fam.issubset(S):
            return ALPHA_AMBIG
    return ALPHA_BASE

# ---------- Late fusion ----------

BLEND_BETA_UNAMB = 0.30   # lean on stock more when clear
BLEND_BETA_AMB   = 0.55   # lean on warped more when ambiguous

def fuse_logits(stock_logits:List[float], warped_logits:List[float], is_unamb:bool)->List[float]:
    beta = BLEND_BETA_UNAMB if is_unamb else BLEND_BETA_AMB
    return [(1.0-beta)*s + beta*w for s,w in zip(stock_logits, warped_logits)]

# ---------- Evaluation harness ----------

def evaluate_tasks(tasks:List[Tuple[OpName,Grid,Grid]], seed:int,
                   scorer_factory:Callable[[int], Scorer],
                   use_fusion:bool=True)->None:
    rng = random.Random(seed)
    scorer = scorer_factory(seed)

    # per-seed calibration bias (on STOCK logits)
    cal_bias = build_cal_bias(scorer, OPS, rng)

    def predict(inp:Grid, cands_for_true:List[OpName], true_op:OpName):
        context = f"Input={pretty_grid(inp)}"
        # STOCK (calibrated)
        stock_raw = scorer.score_labels(context, OPS, mode="stock", alpha=0.0, layer=-1)
        stock_cal = apply_calibration(stock_raw, cal_bias)
        stock_idx = max(range(len(OPS)), key=lambda i: stock_cal[i])
        stock_op  = OPS[stock_idx]
        stock_m   = top2_margin(stock_cal)

        # WARPED with dynamic alpha (calibrated later if used alone)
        alpha = dynamic_alpha(cands_for_true)
        warped_raw = scorer.score_labels(context, OPS, mode="warped", alpha=alpha, layer=-2)
        warped_cal = apply_calibration(warped_raw, cal_bias)
        warped_idx = max(range(len(OPS)), key=lambda i: warped_cal[i])
        warped_op  = OPS[warped_idx]
        warped_m   = top2_margin(warped_cal)

        # Late fusion (optional)
        fused_op, fused_m = None, None
        if use_fusion:
            fused_raw = fuse_logits(stock_raw, warped_raw, is_unamb=(len(cands_for_true)==1))
            fused_cal = apply_calibration(fused_raw, cal_bias)
            fused_idx = max(range(len(OPS)), key=lambda i: fused_cal[i])
            fused_op  = OPS[fused_idx]
            fused_m   = top2_margin(fused_cal)

        # correctness (equivalence & strict)
        def marks(pred:OpName):
            eq  = pred in cands_for_true
            st  = (pred == true_op)
            return eq, st

        stock_eq, stock_st = marks(stock_op)
        warped_eq, warped_st = marks(warped_op)
        fused_eq, fused_st = (False, False)
        if use_fusion:
            fused_eq, fused_st = marks(fused_op)

        return {
            "stock": (stock_op, stock_eq, stock_st, stock_m),
            "warped": (warped_op, warped_eq, warped_st, warped_m),
            "fused": (fused_op, fused_eq, fused_st, fused_m) if use_fusion else None,
        }

    # run
    eq_s, st_s = 0, 0
    eq_w, st_w = 0, 0
    eq_f, st_f = 0, 0
    unamb, amb = 0, 0

    for t, (true_op, x, y) in enumerate(tasks, 1):
        cands = equivalent_ops_for_input(x, y)
        is_unamb = (len(cands)==1)
        unamb += 1 if is_unamb else 0
        amb   += 0 if is_unamb else 1

        res = predict(x, cands, true_op)
        sop, seq, sst, sm = res["stock"]
        wop, weq, wst, wm = res["warped"]
        eq_s += int(seq); st_s += int(sst)
        eq_w += int(weq); st_w += int(wst)

        fused_line = ""
        if res["fused"] is not None:
            fop, feq, fst, fm = res["fused"]
            eq_f += int(feq); st_f += int(fst)
            fused_line = f" | Fused={fop} ({LETTER[fop]}) eq={'✓' if feq else '×'} strict={'✓' if fst else '×'} Δ{fm:.3f}"

        tag = "Unambiguous" if is_unamb else "Ambiguous"
        ctag = f"Cands={list(cands)}"
        print(f"Task {t:03d} | True={true_op} | {tag} | {ctag} | "
              f"Stock={sop} ({LETTER[sop]}) eq={'✓' if seq else '×'} strict={'✓' if sst else '×'} Δ{sm:.3f} | "
              f"Warped={wop} ({LETTER[wop]}) eq={'✓' if weq else '×'} strict={'✓' if wst else '×'} Δ{wm:.3f}"
              f"{fused_line}")

    N = len(tasks)
    print("\n=== Summary ===")
    print(f"N={N} | Unamb={unamb} | Amb={amb} | CalK={CAL_K} | α_base={ALPHA_BASE} | α_amb={ALPHA_AMBIG} | blend_unamb={BLEND_BETA_UNAMB} | blend_amb={BLEND_BETA_AMB}")
    print(f"Equiv Acc : Stock {eq_s}/{N} = {eq_s/N:.1%} | Warped {eq_w}/{N} = {eq_w/N:.1%}" + (f" | Fused {eq_f}/{N} = {eq_f/N:.1%}" if eq_f or st_f else ""))
    print(f"Strict Acc: Stock {st_s}/{N} = {st_s/N:.1%} | Warped {st_w}/{N} = {st_w/N:.1%}" + (f" | Fused {st_f}/{N} = {st_f/N:.1%}" if eq_f or st_f else ""))

# ---------- Entrypoint ----------

if __name__ == "__main__":
    # knobs you’ll most likely sweep
    SEEDS     = [13, 21, 43, 71, 99]   # multi-seed for stability
    N_PER_SEED= 50                      # keep small (fast)
    USE_FUSION= True                    # print “Fused” head too

    for s in SEEDS:
        print(f"\n--- Seed {s} ---")
        tasks = gen_tasks(N_PER_SEED, seed=s)
        evaluate_tasks(tasks, seed=s, scorer_factory=DummyScorer, use_fusion=USE_FUSION)
