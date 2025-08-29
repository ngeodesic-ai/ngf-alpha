# bench_dataset.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Sequence
import numpy as np
from bench_ops import apply_program

Primitive = str

@dataclass
class Case:
    case_id: str
    grid_in: np.ndarray
    grid_out: np.ndarray
    tasks_true: List[Primitive]
    order_true: List[Primitive]
    meta: Dict
    signals_path: Optional[str] = None

def make_synth_cases(n: int = 12, min_len: int = 1, max_len: int = 3, seed: int = 0) -> List[Case]:
    rng = np.random.default_rng(seed)
    ops = ["flip_h", "flip_v", "rotate"]

    cases: List[Case] = []
    for i in range(n):
        H, W = 7, 7
        grid_in = rng.integers(0, 6, size=(H, W), dtype=int)

        L = rng.integers(min_len, max_len + 1)
        order: Sequence[Primitive] = [ops[rng.integers(0, len(ops))] for _ in range(L)]
        tasks = sorted(set(order))

        grid_out = apply_program(grid_in, order)

        cases.append(Case(
            case_id=f"toy{i+1:02d}",
            grid_in=grid_in,
            grid_out=grid_out,
            tasks_true=list(tasks),
            order_true=list(order),
            meta={"len": L},
            signals_path=None,
        ))
    return cases
