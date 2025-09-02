# warp_hook.py
import json
from pathlib import Path

def load_warp_config(path: str):
    cfg = json.loads(Path(path).read_text())
    # Normalize / sanity
    cfg["tap"] = int(cfg.get("tap", -9))
    return cfg
