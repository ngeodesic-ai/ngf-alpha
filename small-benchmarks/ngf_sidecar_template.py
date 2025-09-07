
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional

import numpy as np

@dataclass
class NGFScore:
    depth: float
    margin: float
    conf: float
    gates_ok: bool

class NGFSidecar:
    """
    Black-box NGF sidecar that *actually* uses your Stage-11 scorer.
    You supply `score_fn(vec) -> NGFScore` that wraps Warp→Detect→Denoise.
    Optionally supply an encoder to turn (prompt, candidate) → vec.
    """
    def __init__(
        self,
        score_fn: Callable[[np.ndarray], NGFScore],
        encode_fn: Optional[Callable[[str, str], np.ndarray]] = None,
        gamma: float = 0.5,
        penalty: float = 0.4
    ):
        self.score_fn = score_fn
        self.encode_fn = encode_fn
        self.gamma = gamma
        self.penalty = penalty
        self._state = {}

    def reset(self):
        """Call once per prompt to clear denoiser/EMA/etc (if your score_fn tracks state)."""
        self._state.clear()

    def rerank(self, prompt: str, candidates: List[str]) -> Dict[str, Any]:
        assert self.encode_fn is not None, "encode_fn required for rerank"
        scores: List[NGFScore] = []
        for c in candidates:
            vec = self.encode_fn(prompt, c)
            s = self.score_fn(vec)
            scores.append(s)

        idx_ok = [i for i,s in enumerate(scores) if s.gates_ok]
        if not idx_ok:
            # abstain but still return the argmax by depth+gamma*margin for observability
            comp = [s.depth + self.gamma*s.margin for s in scores]
            best_idx = int(np.argmax(comp))
            return {
                "best": candidates[best_idx],
                "abstain": True,
                "scores": [s.__dict__ for s in scores],
            }

        comp_ok = [scores[i].depth + self.gamma*scores[i].margin for i in idx_ok]
        best_local = idx_ok[int(np.argmax(comp_ok))]
        return {
            "best": candidates[best_local],
            "abstain": False,
            "scores": [s.__dict__ for s in scores],
        }

    def bias_logits(
        self,
        prompt: str,
        topk_ids: List[int],
        logits: np.ndarray,
        id_to_text: Callable[[int], str]
    ) -> np.ndarray:
        """Return adjusted logits using NGF score. Requires encode_fn."""
        assert self.encode_fn is not None, "encode_fn required for bias_logits"
        adj = np.array(logits, dtype=float)
        for j, tid in enumerate(topk_ids):
            ttxt = id_to_text(tid)
            vec = self.encode_fn(prompt, ttxt)
            s = self.score_fn(vec)
            if s.gates_ok:
                adj[j] += (s.depth + self.gamma*s.margin)
            else:
                adj[j] -= self.penalty
        return adj

# ----------------------
# Helpers you can reuse
# ----------------------
def hf_hidden_pooler(model, tokenizer, layer_offset: int = -9, k_last: int = 12):
    """
    Returns an encode_fn that pools last-k token hidden states from a chosen layer.
    Usage:
        encode = hf_hidden_pooler(model, tok, layer_offset=-9, k_last=12)
        vec = encode(prompt, candidate)
    """
    import torch
    def encode(prompt: str, candidate: str) -> np.ndarray:
        with torch.no_grad():
            out = model(**tokenizer(prompt + candidate, return_tensors="pt"),
                        output_hidden_states=True)
        hs = out.hidden_states[layer_offset][0]  # [T, D]
        vec = hs[-k_last:].mean(dim=0).cpu().numpy()
        return vec
    return encode

# ----------------------
# Example: wiring it up
# ----------------------
if __name__ == "__main__":
    # 1) Bring your real NGF scorer here.
    # Example shim:
    #
    # from your_repo.stage11 import score_latent  # <- your function
    # def score_fn(vec: np.ndarray) -> NGFScore:
    #     m = score_latent(vec)  # returns dict with depth, margin, conf, gates_ok
    #     return NGFScore(**m)
    #
    # 2) Choose an encoder (HF hidden-state pooler or your sentence/ARC encoder).
    # encode_fn = hf_hidden_pooler(hf_model, hf_tokenizer, layer_offset=-9, k_last=12)
    #
    # 3) Create the sidecar and call rerank or bias_logits.
    #
    # sidecar = NGFSidecar(score_fn=score_fn, encode_fn=encode_fn)
    # sidecar.reset()
    # result = sidecar.rerank("Round 7.432 to one decimal.", ["7.4","7.43","7.5","7.40","8"])
    # print(result)
    pass
