from itertools import product
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from bench_ops import apply_program, PRIMS

@dataclass
class Result:
    tasks_pred: List[str]
    order_pred: List[str]
    grid_pred: np.ndarray
    info: Dict

class StockAdapter:
    """Deterministic baseline via tiny program search (no LLM required).
    Tries all sequences up to max_len over {flip_h, flip_v, rotate} and returns the first hit.
    """
    name = "Stock-ProgramSearch"

    def __init__(self, max_len: int = 3):
        self.max_len = max_len
        self.ops = list(PRIMS.keys())

    def run_case(self, case):
        # brute-force search for a program that exactly matches the target grid
        for L in range(1, self.max_len + 1):
            for seq in product(self.ops, repeat=L):
                out = apply_program(case.grid_in, seq)
                if out.shape == case.grid_out.shape and (out == case.grid_out).all():
                    tasks = sorted(set(seq))
                    return Result(tasks_pred=tasks, order_pred=list(seq), grid_pred=out, info={"hit_len": L})
        # fallback: identity
        return Result(tasks_pred=[], order_pred=[], grid_pred=case.grid_in.copy(), info={"hit_len": 0})