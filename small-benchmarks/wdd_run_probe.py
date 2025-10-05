# wdd_run_probe.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wdd_stage11 import attach_wdd, STAGE11_PRESET

MODEL = "gpt2"
LAYER = 9    # your current best tap

model = AutoModelForCausalLM.from_pretrained(MODEL)
tok   = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token; tok.padding_side = "right"

# Attach full WDD (Warp+Detect+Denoise) with Stage-11 preset
mods, cache, handle = attach_wdd(model, layer_idx=LAYER,
                                 alpha=1.0, beta=0.5,
                                 preset=STAGE11_PRESET)

model.eval()
text = [
  "In a quiet village, a curious child asked a question about truth.",
  "The elder replied with a parable about mirrors and light and angles.",
  "Years later, the student revisited the riddle with first principles."
]
batch = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=160)
with torch.no_grad():
    _ = model(**batch)

print("cache shapes:", cache["pre"].shape, cache["post"].shape)
handle.remove()
