# bench_ops.py
import numpy as np
from typing import Sequence, Dict, Callable

def flip_h(grid: np.ndarray) -> np.ndarray:
    return np.flip(grid, axis=1)

def flip_v(grid: np.ndarray) -> np.ndarray:
    return np.flip(grid, axis=0)

def rotate(grid: np.ndarray) -> np.ndarray:
    # 90° clockwise (ARC often uses this)
    return np.rot90(grid, k=-1)

PRIMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "flip_h": flip_h,
    "flip_v": flip_v,
    "rotate": rotate,
}

def apply_program(grid: np.ndarray, seq: Sequence[str]) -> np.ndarray:
    out = grid
    for op in seq:
        out = PRIMS[op](out)
    return out
