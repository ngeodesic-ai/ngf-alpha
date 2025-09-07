# Demo: Minimal NGF-style sidecar harness (with a mock scorer)
# This demonstrates how the black-box API wires into a decoding/reranking loop.
# In a real setup, replace `MockScorer.score_vec` with your Stage-11 Warp→Detect→Denoise scorer.

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

# ---------------------------
# Mock encoder
# ---------------------------
def simple_text_vec(text: str) -> np.ndarray:
    """
    Super-lightweight text featurizer for demo purposes only.
    Produces a fixed-size vector from character-level counts and a few regex-y cues.
    In practice, you'd replace this with pooled hidden states or a sentence embedding.
    """
    import re
    # char buckets
    lowers = sum(c.islower() for c in text)
    uppers = sum(c.isupper() for c in text)
    digits = sum(c.isdigit() for c in text)
    spaces = text.count(" ")
    dots = text.count(".")
    commas = text.count(",")
    other_punct = sum(c in "!?:;-%" for c in text)
    length = len(text)
    # numeric cues
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    has_num = 1 if nums else 0
    num_len_avg = np.mean([len(n) for n in nums]) if nums else 0.0
    has_one_decimal = 1 if any(re.match(r"^-?\d+\.\d$", n) for n in nums) else 0
    has_two_decimals = 1 if any(re.match(r"^-?\d+\.\d\d$", n) for n in nums) else 0
    has_three_decimals = 1 if any(re.match(r"^-?\d+\.\d\d\d$", n) for n in nums) else 0
    # pack
    v = np.array([lowers, uppers, digits, spaces, dots, commas, other_punct, length,
                  has_num, num_len_avg, has_one_decimal, has_two_decimals, has_three_decimals],
                 dtype=float)
    # normalize
    v = (v - v.mean()) / (v.std() + 1e-6)
    return v

# ---------------------------
# Mock Warp→Detect→Denoise scorer
# ---------------------------
@dataclass
class Score:
    depth: float
    margin: float
    conf: float
    gates_ok: bool

class MockScorer:
    """
    A toy stand-in for Stage-11. It mimics behavior:
    - prefers outputs matching one-decimal formatting when the instruction asks for it
    - penalizes off-topic or non-numeric text
    - computes a 'margin' vs the 2nd-best option
    NOTE: for illustration only. Replace with your real scorer.
    """
    def __init__(self, context: str):
        self.context = context
        self._history = []  # for mock denoise
    
    def score_batch(self, prompt: str, candidates: List[str]) -> List[Score]:
        base = simple_text_vec(prompt)
        # Encode candidates
        mats = np.stack([simple_text_vec(prompt + " " + c) for c in candidates], axis=0)
        # "Warp": project by PCA to 2D for a radial depth (demo only)
        X = mats - mats.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Z = X @ Vt.T[:, :2]  # 2D scores
        radii = np.linalg.norm(Z, axis=1)
        # Heuristic: smaller radius => deeper (we flip and scale)
        depth_raw = -radii
        # Boost candidates that look like one-decimal numbers if the prompt asks for rounding
        import re
        asks_one_dec = "one decimal" in prompt.lower()
        fmt_bonus = np.array([1.0 if asks_one_dec and re.search(r"\b-?\d+\.\d\b", c) else 0.0 for c in candidates])
        # Penalize non-numeric
        numeric_mask = np.array([1.0 if re.search(r"\d", c) else 0.0 for c in candidates])
        depth = (depth_raw - depth_raw.min()) / (depth_raw.ptp() + 1e-6)
        depth = 0.75*depth + 0.35*fmt_bonus + 0.10*numeric_mask
        # "Detect": threshold vs a null made from shuffled features (toy)
        null = np.mean(depth) - 0.25*np.std(depth)
        gates = depth > null
        # "Margin": depth gap to next-best among gate-passing
        order = np.argsort(-depth)
        best = order[0]
        second = order[1] if len(order) > 1 else best
        margin = np.zeros_like(depth)
        margin[best] = max(0.0, depth[best] - depth[second])
        # confidence ~ squashed depth
        conf = 1 / (1 + np.exp(-5*(depth - depth.mean())))
        # package
        return [Score(depth=float(depth[i]), margin=float(margin[i]), conf=float(conf[i]), gates_ok=bool(gates[i])) for i in range(len(candidates))]

# ---------------------------
# NGF Sidecar harness (black-box API)
# ---------------------------
class NGFSidecar:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = {}

    def rerank(self, prompt: str, candidates: List[str]) -> Dict[str, Any]:
        scorer = MockScorer(context=prompt)
        scores = scorer.score_batch(prompt, candidates)
        # choose by depth + gamma*margin but only among gate-passing ones
        gamma = 0.5
        idx_ok = [i for i,s in enumerate(scores) if s.gates_ok]
        if not idx_ok:
            # abstain: pick by raw depth anyway but mark abstain
            best_idx = int(np.argmax([s.depth for s in scores]))
            return {
                "best": candidates[best_idx],
                "abstain": True,
                "scores": [s.__dict__ for s in scores]
            }
        comp = np.array([scores[i].depth + gamma*scores[i].margin for i in idx_ok])
        best_local = idx_ok[int(np.argmax(comp))]
        return {
            "best": candidates[best_local],
            "abstain": False,
            "scores": [s.__dict__ for s in scores]
        }

# ---------------------------
# Demo 1: rounding example
# ---------------------------
prompt1 = "Round 7.432 to one decimal."
cands1  = ["7.4", "7.43", "7.5", "7.40", "8"]
sidecar = NGFSidecar()
out1 = sidecar.rerank(prompt1, cands1)

# ---------------------------
# Demo 2: HellaSwag-style toy
# ---------------------------
prompt2 = "A man is riding a skateboard down a ramp. He loses his balance. The most plausible next event is:"
cands2  = [
    "He falls off and lands on the ground.",
    "A cat jumps onto the table.",
    "The man starts typing on a keyboard.",
    "The skateboard floats into space."
]
out2 = sidecar.rerank(prompt2, cands2)

print("=== Demo 1: Rounding ===")
print("Prompt:", prompt1)
print("Best:", out1["best"], "Abstain:", out1["abstain"])
for s, c in zip(out1["scores"], cands1):
    print(f"{c:8s} -> depth={s['depth']:.3f} margin={s['margin']:.3f} conf={s['conf']:.2f} gate={s['gates_ok']}")

print("\n=== Demo 2: HellaSwag Toy ===")
print("Prompt:", prompt2)
print("Best:", out2["best"], "Abstain:", out2["abstain"])
for s, c in zip(out2["scores"], cands2):
    print(f"{c:45s} -> depth={s['depth']:.3f} margin={s['margin']:.3f} conf={s['conf']:.2f} gate={s['gates_ok']}")
