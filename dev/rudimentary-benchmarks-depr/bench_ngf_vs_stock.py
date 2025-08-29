
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Protocol, Optional, Iterable
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from bench_ngf_adapter import NGFAdapter, NGFConfig, parse_from_signals_example
from bench_stock_adapter import StockAdapter
from bench_dataset import make_synth_cases   # add this import

ngf = NGFAdapter(parse_from_signals_example, NGFConfig(tau_area=10.0, tau_corr=0.7))
stock = StockAdapter(max_len=3)

Primitive = str  # e.g., "flip_h", "flip_v", "rotate"

@dataclass
class Case:
    case_id: str
    grid_in: np.ndarray          # (H,W) int (ARC palette indices)
    grid_out: np.ndarray         # (H,W) int
    tasks_true: List[Primitive]  # unique primitive names present (set-level truth)
    order_true: List[Primitive]  # ordered sequence (may repeat if needed)
    meta: Dict                   # anything else you need
    signals_path: Optional[str] = None  # optional: path to precomputed latents/signals

@dataclass
class Result:
    tasks_pred: List[Primitive]
    order_pred: List[Primitive]
    grid_pred: np.ndarray
    info: Dict

class Adapter(Protocol):
    name: str
    def run_case(self, case: Case) -> Result: ...

def task_set_accuracy(tasks_true: List[Primitive], tasks_pred: List[Primitive]) -> float:
    return float(set(tasks_true) == set(tasks_pred))

def order_exact_accuracy(order_true: List[Primitive], order_pred: List[Primitive]) -> float:
    return float(order_true == order_pred)

def execution_exact(grid_true: np.ndarray, grid_pred: np.ndarray) -> float:
    same_shape = grid_true.shape == grid_pred.shape
    return float(same_shape and np.array_equal(grid_true, grid_pred))

def _plot_grid(ax, grid: np.ndarray, title: str):
    ax.imshow(grid, interpolation="nearest")  # default colormap; no explicit colors
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

def save_comparison_figure(case: Case, r_ngf: Result, r_stock: Result, out_path: str):
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    _plot_grid(axs[0], case.grid_in,  f"Input\n{case.case_id}")
    _plot_grid(axs[1], r_ngf.grid_pred, f"NGF\norder: { ' → '.join(r_ngf.order_pred) or '—' }")
    _plot_grid(axs[2], r_stock.grid_pred, f"Stock\norder: { ' → '.join(r_stock.order_pred) or '—' }")
    _plot_grid(axs[3], case.grid_out, "Target")
    fig.suptitle("ARC — NGF vs Stock", fontsize=12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

class SimpleListDataset:
    def __init__(self, cases: List[Case]):
        self._cases = cases
    def __iter__(self) -> Iterable[Case]:
        return iter(self._cases)

class NGFAdapterStub:
    name = "NGF-v2"
    def run_case(self, case: Case) -> Result:
        # TODO: plug in your Stage-10 v2 parser+executor here
        return Result(tasks_pred=[], order_pred=[], grid_pred=case.grid_in.copy(), info={})

class StockAdapterStub:
    name = "Stock-LLM"
    def run_case(self, case: Case) -> Result:
        # TODO: plug in your stock baseline here
        return Result(tasks_pred=[], order_pred=[], grid_pred=case.grid_in.copy(), info={})

def run_benchmark(dataset: Iterable[Case], ngf: Adapter, stock: Adapter, out_dir: str, max_cases: Optional[int] = None):
    os.makedirs(out_dir, exist_ok=True)
    logs = []
    for i, case in enumerate(dataset):
        if max_cases is not None and i >= max_cases:
            break
        r_ngf = ngf.run_case(case)
        r_stock = stock.run_case(case)
        m = {
            "case_id": case.case_id,
            "ngf": {
                "task_set_acc": task_set_accuracy(case.tasks_true, r_ngf.tasks_pred),
                "order_acc":    order_exact_accuracy(case.order_true, r_ngf.order_pred),
                "exec_acc":     execution_exact(case.grid_out, r_ngf.grid_pred),
                "tasks_pred":   r_ngf.tasks_pred,
                "order_pred":   r_ngf.order_pred,
            },
            "stock": {
                "task_set_acc": task_set_accuracy(case.tasks_true, r_stock.tasks_pred),
                "order_acc":    order_exact_accuracy(case.order_true, r_stock.order_pred),
                "exec_acc":     execution_exact(case.grid_out, r_stock.grid_pred),
                "tasks_pred":   r_stock.tasks_pred,
                "order_pred":   r_stock.order_pred,
            },
        }
        logs.append(m)
        fig_path = os.path.join(out_dir, f"{case.case_id}.png")
        save_comparison_figure(case, r_ngf, r_stock, fig_path)
        print(f"[{case.case_id}] NGF exec={m['ngf']['exec_acc']} | Stock exec={m['stock']['exec_acc']} | "
              f"true order={case.order_true} | NGF={r_ngf.order_pred} | Stock={r_stock.order_pred} | plot={fig_path}")
    with open(os.path.join(out_dir, "run_log.json"), "w") as f:
        json.dump(logs, f, indent=2)

def make_toy_cases(n: int = 3) -> List[Case]:
    rng = np.random.default_rng(0)
    cases = []
    for i in range(n):
        H, W = 7, 7
        grid_in = rng.integers(0, 6, size=(H, W), dtype=int)
        grid_out = grid_in.copy()
        cases.append(Case(
            case_id=f"toy{i+1:02d}",
            grid_in=grid_in,
            grid_out=grid_out,
            tasks_true=[],
            order_true=[],
            meta={},
            signals_path=None,
        ))
    return cases

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/bench", help="Output directory for figures/logs")
    ap.add_argument("--max_cases", type=int, default=None, help="Limit number of cases")
    ap.add_argument("--n", type=int, default=12, help="number of synthetic cases")
    ap.add_argument("--min_len", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=3)
    args = ap.parse_args()

    dataset = make_synth_cases(n=args.n, min_len=args.min_len, max_len=args.max_len)

    # if your real NGF parser is importable, this line in your file already works:
    #   ngf = NGFAdapter(parse_from_signals_example, NGFConfig(tau_area=10.0, tau_corr=0.7))
    # otherwise, it'll use the fallback search and still run
    run_benchmark(dataset, ngf, stock, out_dir=args.out, max_cases=args.max_cases)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--out", type=str, default="results/bench", help="Output directory for figures/logs")
#     ap.add_argument("--max_cases", type=int, default=None, help="Limit number of cases")
#     args = ap.parse_args()
#     dataset = SimpleListDataset(make_toy_cases(6))
#     ngf = NGFAdapterStub()
#     stock = StockAdapterStub()
#     run_benchmark(dataset, ngf, stock, out_dir=args.out, max_cases=args.max_cases)

if __name__ == "__main__":
    main()
