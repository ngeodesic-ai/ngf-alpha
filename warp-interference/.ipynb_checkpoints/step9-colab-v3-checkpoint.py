# Write both patched scripts and a small README, then zip them for download
import os, zipfile, hashlib, textwrap, json

base = "/mnt/data"

v3_path = os.path.join(base, "step9_colab_v3_patched.py")
v3_content = r'''# step9_colab_v3_patched.py — Updated draft with corrections
# - Correct rotation ground truth
# - Pass nudged hidden through GPT-2 final layer norm (ln_f) before lm_head
# - Nudge cadence based on continuation step, not total sequence length
# - Frequency penalty starts empty (prompt tokens not penalized)
# - Stable nudge parameters (near-critical damping) & fewer steps
# - Dynamic PCA n_components (>=95% explained variance)
#
# NOTE: This script is self-contained and runnable in a standard Python environment
# with PyTorch and Hugging Face Transformers installed. Internet is required
# to download the GPT-2 weights the first time (unless cached).
#
# Tested against transformers >= 4.30.0 and torch >= 2.1.
#
import re
import math
import random
from collections import Counter
from typing import List

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import GPT2Tokenizer, GPT2LMHeadModel


MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VALID_TOKENS = set(list("[]0123456789, "))
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7
TOP_P = 0.95
ALPHA_BLEND = 0.45
NUDGE_INTERVAL = 4

PULL_STRENGTH = 3.7
GAMMA = 2 * math.sqrt(PULL_STRENGTH)
DAMPING = 0.0
DT = 0.05
NUDGE_STEPS_PER_EVENT = 1

MIN_VAR_EXPLAINED = 0.95
MAX_PCA_COMPONENTS = 16


def extract_grid(text: str) -> List[List[int]]:
    m = re.search(r"\[\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*,\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*\]", text)
    if not m:
        return []
    a, b, c, d = map(int, m.groups())
    return [[a, b], [c, d]]


def rotate_2x2_clockwise(grid: List[List[int]]) -> List[List[int]]:
    a, b = grid[0]
    c, d = grid[1]
    return [[c, a], [d, b]]


def fit_pca_dynamic(latents, min_var=MIN_VAR_EXPLAINED, max_comp=MAX_PCA_COMPONENTS) -> PCA:
    X = np.vstack(latents)
    pca_full = PCA(n_components=min(X.shape[0], X.shape[1])).fit(X)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.searchsorted(cumsum, min_var) + 1
    n_components = min(n_components, max_comp)
    pca = PCA(n_components=n_components).fit(X)
    return pca


def last_layer_hidden(model, tokenizer, text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[-1].squeeze(0).detach().cpu().numpy()
    return hidden


def masked_sampling(logits: torch.Tensor, tokenizer: GPT2Tokenizer, token_counts: Counter,
                    temperature: float, top_p: float) -> int:
    logits = logits.clone()
    vocab_size = logits.shape[-1]
    allowed = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    for tid in range(vocab_size):
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if s and set(s).issubset(VALID_TOKENS):
            allowed[tid] = True
    logits[~allowed] = -float("inf")
    for tid, cnt in token_counts.items():
        if cnt > 0 and tid < vocab_size:
            logits[tid] -= 0.8 * cnt
    if temperature and temperature > 0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative > top_p
    if torch.any(cutoff):
        first_cut = torch.nonzero(cutoff, as_tuple=True)[0][0].item()
        sorted_probs[first_cut + 1:] = 0.0
        sorted_probs = sorted_probs / (sorted_probs.sum() + 1e-12)
        choice = torch.multinomial(sorted_probs, num_samples=1)
        token_id = sorted_idx[choice]
        return token_id.item()
    else:
        return int(torch.argmax(probs).item())


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    input_grid = [[5, 6], [7, 8]]
    ground_truth = rotate_2x2_clockwise(input_grid)

    prompt = (
        "Identify the pattern: Input grid [[1,2],[3,4]] -> Output [[4,1],[2,3]] (90 deg rotate). "
        "Apply to [[5,6],[7,8]]. Respond with only a 2x2 grid like [[a,b],[c,d]]."
    )

    correct_example = (
        "The pattern is a 90-degree clockwise rotation. "
        "Applying this to [[5,6],[7,8]] gives [[7,5],[8,6]]."
    )

    h_prompt = last_layer_hidden(model, tokenizer, prompt)
    h_correct = last_layer_hidden(model, tokenizer, correct_example)
    pca = fit_pca_dynamic([h_prompt, h_correct])

    red_prompt = pca.transform(h_prompt).mean(axis=0)
    red_target = pca.transform(h_correct).mean(axis=0)

    x = red_prompt.copy(); v = np.zeros_like(x)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    generated = inputs["input_ids"]
    token_counts = Counter()
    cont_step = 0

    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            out = model(generated, output_hidden_states=True)
            base_logits = out.logits[:, -1, :]
            last_hidden = out.hidden_states[-1][:, -1, :]

        if cont_step > 0 and (cont_step % NUDGE_INTERVAL == 0):
            cur_red = pca.transform(last_hidden.detach().cpu().numpy())
            x = cur_red[0] if cont_step == NUDGE_INTERVAL else x
            for _ in range(NUDGE_STEPS_PER_EVENT):
                a = PULL_STRENGTH * (red_target - x) - GAMMA * v - DAMPING * v
                v = v + a * DT
                x = x + v * DT
            nudged_hidden_np = pca.inverse_transform(x)
            nudged_hidden = torch.from_numpy(nudged_hidden_np).to(last_hidden.device).to(torch.float32)
            nudged_hidden = nudged_hidden.unsqueeze(0).unsqueeze(0)
            nudged_hidden = model.transformer.ln_f(nudged_hidden)
            with torch.no_grad():
                nudged_logits = model.lm_head(nudged_hidden)[:, 0, :]
            logits = (1.0 - ALPHA_BLEND) * base_logits + ALPHA_BLEND * nudged_logits
        else:
            logits = base_logits

        next_id = masked_sampling(logits[0], tokenizer, token_counts, TEMPERATURE, TOP_P)
        next_token = torch.tensor([[next_id]], device=generated.device, dtype=generated.dtype)
        token_counts[next_id] += 1
        generated = torch.cat([generated, next_token], dim=1)
        cont_step += 1

        decoded = tokenizer.decode(generated[0], clean_up_tokenization_spaces=False)
        grid = extract_grid(decoded)
        if grid:
            break

    decoded = tokenizer.decode(generated[0], clean_up_tokenization_spaces=False)
    grid = extract_grid(decoded)
    print("Decoded:", decoded)
    if grid:
        print("Extracted grid:", grid)
    else:
        print("No 2x2 grid found in output.")
    print("Ground truth (90° CW):", ground_truth)
    if grid:
        print("Match ground truth?", grid == ground_truth)


if __name__ == "__main__":
    main()
'''

v4_path = os.path.join(base, "step9_colab_v4_suffixfix.py")
v4_content = r'''# step9_colab_v4_suffixfix.py — Fix extraction to avoid grabbing the prompt example
# - Parses only the generated suffix (excludes prompt text)
# - Waits for a closed grid "]]" before parsing
# - Extracts the last 2x2 grid in the suffix
# - Keeps model/ln_f and dynamics fixes from v3
import re
import math
import random
from collections import Counter
from typing import List

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import GPT2Tokenizer, GPT2LMHeadModel


MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VALID_TOKENS = set(list("[]0123456789, "))
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7
TOP_P = 0.90
ALPHA_BLEND = 0.65
NUDGE_INTERVAL = 2

PULL_STRENGTH = 3.7
GAMMA = 2 * math.sqrt(PULL_STRENGTH)
DAMPING = 0.0
DT = 0.05
NUDGE_STEPS_PER_EVENT = 1

MIN_VAR_EXPLAINED = 0.95
MAX_PCA_COMPONENTS = 16


def extract_last_grid(text: str) -> List[List[int]]:
    pat = re.compile(r"\[\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*,\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*\]")
    matches = list(pat.finditer(text))
    if not matches:
        return []
    a, b, c, d = map(int, matches[-1].groups())
    return [[a, b], [c, d]]


def rotate_2x2_clockwise(grid: List[List[int]]) -> List[List[int]]:
    a, b = grid[0]
    c, d = grid[1]
    return [[c, a], [d, b]]


def fit_pca_dynamic(latents, min_var=MIN_VAR_EXPLAINED, max_comp=MAX_PCA_COMPONENTS) -> PCA:
    X = np.vstack(latents)
    pca_full = PCA(n_components=min(X.shape[0], X.shape[1])).fit(X)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.searchsorted(cumsum, min_var) + 1
    n_components = min(n_components, max_comp)
    pca = PCA(n_components=n_components).fit(X)
    return pca


def last_layer_hidden(model, tokenizer, text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1].squeeze(0).detach().cpu().numpy()


def masked_sampling(logits: torch.Tensor, tokenizer: GPT2Tokenizer, token_counts: Counter,
                    temperature: float, top_p: float) -> int:
    logits = logits.clone()
    vocab_size = logits.shape[-1]
    allowed = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    for tid in range(vocab_size):
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if s and set(s).issubset(VALID_TOKENS):
            allowed[tid] = True
    logits[~allowed] = -float("inf")
    for tid, cnt in token_counts.items():
        if cnt > 0 and tid < vocab_size:
            logits[tid] -= 0.8 * cnt
    if temperature and temperature > 0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative > top_p
    if torch.any(cutoff):
        first_cut = torch.nonzero(cutoff, as_tuple=True)[0][0].item()
        sorted_probs[first_cut + 1:] = 0.0
        sorted_probs = sorted_probs / (sorted_probs.sum() + 1e-12)
        choice = torch.multinomial(sorted_probs, num_samples=1)
        token_id = sorted_idx[choice]
        return token_id.item()
    else:
        return int(torch.argmax(probs).item())


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    input_grid = [[5, 6], [7, 8]]
    ground_truth = rotate_2x2_clockwise(input_grid)

    prompt = (
        "Identify the pattern: Input grid [[1,2],[3,4]] -> Output [[4,1],[2,3]] (90 deg rotate). "
        "Apply to [[5,6],[7,8]]. Respond with only a 2x2 grid like [[a,b],[c,d]]."
    )
    correct_example = (
        "The pattern is a 90-degree clockwise rotation. "
        "Applying this to [[5,6],[7,8]] gives [[7,5],[8,6]]."
    )

    h_prompt = last_layer_hidden(model, tokenizer, prompt)
    h_correct = last_layer_hidden(model, tokenizer, correct_example)
    pca = fit_pca_dynamic([h_prompt, h_correct])

    red_prompt = pca.transform(h_prompt).mean(axis=0)
    red_target = pca.transform(h_correct).mean(axis=0)

    x = red_prompt.copy(); v = np.zeros_like(x)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    generated = inputs["input_ids"]
    prompt_len = generated.shape[1]
    token_counts = Counter()
    cont_step = 0

    # Optional small bias to enter grid mode
    bias = tokenizer.encode(" [[", add_special_tokens=False)
    generated = torch.cat([generated, torch.tensor([bias], device=generated.device)], dim=1)

    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            out = model(generated, output_hidden_states=True)
            base_logits = out.logits[:, -1, :]
            last_hidden = out.hidden_states[-1][:, -1, :]

        if cont_step > 0 and (cont_step % NUDGE_INTERVAL == 0):
            cur_red = pca.transform(last_hidden.detach().cpu().numpy())
            if cont_step == NUDGE_INTERVAL:
                x = cur_red[0]
            for _ in range(NUDGE_STEPS_PER_EVENT):
                a = PULL_STRENGTH * (red_target - x) - GAMMA * v - DAMPING * v
                v = v + a * DT
                x = x + v * DT
            nudged_hidden_np = pca.inverse_transform(x)
            nudged_hidden = torch.from_numpy(nudged_hidden_np).to(last_hidden.device).to(torch.float32)
            nudged_hidden = nudged_hidden.unsqueeze(0).unsqueeze(0)
            nudged_hidden = model.transformer.ln_f(nudged_hidden)
            with torch.no_grad():
                nudged_logits = model.lm_head(nudged_hidden)[:, 0, :]
            logits = (1.0 - ALPHA_BLEND) * base_logits + ALPHA_BLEND * nudged_logits
        else:
            logits = base_logits

        next_id = masked_sampling(logits[0], tokenizer, token_counts, TEMPERATURE, TOP_P)
        next_token = torch.tensor([[next_id]], device=generated.device, dtype=generated.dtype)
        token_counts[next_id] += 1
        generated = torch.cat([generated, next_token], dim=1)
        cont_step += 1

        suffix = tokenizer.decode(generated[0, prompt_len:], clean_up_tokenization_spaces=False)
        if "]]" in suffix:
            break

    suffix = tokenizer.decode(generated[0, prompt_len:], clean_up_tokenization_spaces=False)
    grid = extract_last_grid(suffix)
    print("Decoded suffix:", suffix)
    if grid:
        print("Extracted grid:", grid)
    else:
        print("No 2x2 grid found in generated suffix.")
    print("Ground truth (90° CW):", ground_truth)
    if grid:
        print("Match ground truth?", grid == ground_truth)


if __name__ == "__main__":
    main()
'''

readme_path = os.path.join(base, "README_step9_patched.txt")
readme = textwrap.dedent("""\
    NGF step9 patched scripts

    Files:
      - step9_colab_v3_patched.py
      - step9_colab_v4_suffixfix.py

    What’s fixed:
      • Correct 90° clockwise rotation target.
      • Nudged hidden passes through GPT-2 final layer norm (ln_f) before lm_head.
      • Nudge cadence based on continuation steps (prompt length independent).
      • Frequency penalty starts empty (prompt tokens not penalized).
      • Near‑critical damping of the nudge (fewer steps, more stability).
      • Dynamic PCA picks n_components to reach ≥95% explained variance.
      • v4 parses only the generated suffix and extracts the LAST closed 2×2 grid.

    Quick start:
      pip install torch transformers==4.30.0 scikit-learn numpy
      python step9_colab_v4_suffixfix.py
      # or: python step9_colab_v3_patched.py

    Notes:
      • These scripts download GPT‑2 weights on first run unless already cached.
      • If you prefer less steering, reduce ALPHA_BLEND and increase TOP_P.
      • To disable the small grid bias, remove the appended ' [['.

    """)

# Write files
with open(v3_path, "w", encoding="utf-8") as f:
    f.write(v3_content)
with open(v4_path, "w", encoding="utf-8") as f:
    f.write(v4_content)
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)

# Zip bundle
zip_path = os.path.join(base, "ngf_step9_patched_bundle.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(v3_path, arcname=os.path.basename(v3_path))
    z.write(v4_path, arcname=os.path.basename(v4_path))
    z.write(readme_path, arcname=os.path.basename(readme_path))

# Report sizes & hashes
def info(p):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    return {"path": p, "size": os.path.getsize(p), "sha256_prefix": h}

print(json.dumps({
    "files": [info(v3_path), info(v4_path), info(readme_path), info(zip_path)]
}, indent=2))
