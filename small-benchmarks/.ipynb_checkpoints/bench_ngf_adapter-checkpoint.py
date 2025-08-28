# bench_ngf_adapter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import numpy as np

from bench_ops import apply_program

@dataclass
class NGFConfig:
    tau_area: float = 10.0
    tau_corr: float = 0.7

@dataclass
class NGFResult:
    tasks_pred: List[str]
    order_pred: List[str]
    grid_pred: np.ndarray
    info: Dict

# --- try to import your Stage-10 v2 parser ---
_parse_fn: Optional[Callable] = None
try:
    # you can rename this to match your file (e.g., arc_geodesic_task_parser_latest)
    from arc_geodesic_task_parser_latest import parse_from_signals as _v2_parse
    _parse_fn = _v2_parse
except Exception:
    try:
        # matches what your bench file imports by name
        from arc_geodesic_task_parser_latest import parse_from_signals_example as _v2_parse_ex
        _parse_fn = _v2_parse_ex
    except Exception:
        _parse_fn = None

def parse_from_signals_example(grid_in: np.ndarray, grid_out: np.ndarray, cfg: NGFConfig) -> Dict:
    """
    Expected to return a dict with at least:
      {'tasks': [...], 'order': [...], 'debug': {...}}
    """
    if _parse_fn is not None:
        return _parse_fn(grid_in, grid_out, tau_area=cfg.tau_area, tau_corr=cfg.tau_corr)

    # Fallback: tiny deterministic search (keeps the harness runnable).
    # NOTE: This is just a stand-in if the parser isn't importable.
    from itertools import product
    ops = ["flip_h", "flip_v", "rotate"]
    for L in range(1, 4):
        for seq in product(ops, repeat=L):
            if np.array_equal(apply_program(grid_in, seq), grid_out):
                return {"tasks": sorted(set(seq)), "order": list(seq), "debug": {"fallback": True}}
    return {"tasks": [], "order": [], "debug": {"fallback": True}}

class NGFAdapter:
    name = "NGF-v2"

    def __init__(self, parse_fn: Callable = parse_from_signals_example, cfg: Optional[NGFConfig] = None):
        self.parse_fn = parse_fn
        self.cfg = cfg or NGFConfig()

    def run_case(self, case) -> NGFResult:
        # 1) parse tasks + order from (in, out) using geodesic signals (or fallback)
        parsed = self.parse_fn(case.grid_in, case.grid_out, self.cfg)
        tasks = list(parsed.get("tasks", []))
        order = list(parsed.get("order", []))

        # 2) execute predicted order
        pred = apply_program(case.grid_in, order) if order else case.grid_in.copy()

        return NGFResult(
            tasks_pred=tasks,
            order_pred=order,
            grid_pred=pred,
            info={"parsed": parsed},
        )
