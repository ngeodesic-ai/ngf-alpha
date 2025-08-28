ARC demo (training-style) tasks
===============================

This folder contains three ARC-style JSON tasks suitable for sanity checks:

1) 001_rotate90.json — 90° clockwise rotation
2) 002_flip_h.json   — horizontal mirror (flip left-right)
3) 003_color_swap.json — swap colors 1<->2 and 3<->4

Each file has:
- "train": a couple of input/output pairs
- "test" : one input with its ground-truth output (since we're emulating the *training* split)

You can point the harness' --arc_path to this folder to run quick checks.
