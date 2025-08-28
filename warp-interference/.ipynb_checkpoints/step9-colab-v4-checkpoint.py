# step9_colab_v6_slotbias.py — Structured decoding + slot-wise digit bias toward the rotation
# Your setup: numpy 2.0.2, torch 2.8.0+cu126, scikit-learn 1.6.1, transformers 4.55.2

import math, random
from collections import Counter
from typing import List, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# --------------------------
# Config
# --------------------------
MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Steering & constraints
TEMPERATURE = 0.7
TOP_P = 0.80
ALPHA_BLEND = 0.85     # stronger blend toward nudged logits
NUDGE_INTERVAL = 1     # nudge every continuation step

MAX_PCA_COMPONENTS = 16
MIN_VAR_EXPLAINED = 0.95

# Nudge dynamics (near-critical)
PULL_STRENGTH = 3.7
GAMMA = 2 * math.sqrt(PULL_STRENGTH)  # ~3.85
DAMPING = 0.0
DT = 0.05
NUDGE_STEPS_PER_EVENT = 1

# Structured decoder options
RESTRICT_TO_INPUT_DIGITS = True  # limit digits to those seen in the input grid
SLOT_LOGIT_BIAS = 5.0            # logit bonus for the correct digit at each slot

# --------------------------
# Helpers
# --------------------------
def rotate_2x2_clockwise(grid: List[List[int]]) -> List[List[int]]:
    a, b = grid[0]
    c, d = grid[1]
    return [[c, a], [d, b]]

def fit_pca_dynamic(latents: List[np.ndarray],
                    min_var: float = MIN_VAR_EXPLAINED,
                    max_comp: int = MAX_PCA_COMPONENTS) -> PCA:
    X = np.vstack(latents)
    pca_full = PCA(n_components=min(X.shape[0], X.shape[1])).fit(X)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumsum, min_var) + 1)
    n_components = min(n_components, max_comp)
    pca = PCA(n_components=n_components).fit(X)
    return pca

def last_layer_hidden(model, tokenizer, text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1].squeeze(0).detach().cpu().numpy()

def encode_literal(tokenizer: GPT2Tokenizer, s: str, device) -> torch.Tensor:
    ids = tokenizer.encode(s, add_special_tokens=False)
    return torch.tensor([ids], device=device, dtype=torch.long)

def mask_tokens_for_digit(tokenizer: GPT2Tokenizer,
                          logits: torch.Tensor,
                          whitelist: Optional[set] = None) -> torch.Tensor:
    """
    Mask of vocab tokens that decode to a single digit (0..9) after lstrip().
    If whitelist is provided (e.g., {"5","6","7","8"}), require digit ∈ whitelist.
    """
    vocab_size = logits.shape[-1]
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    for tid in range(vocab_size):
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if not s:
            continue
        t = s.lstrip()
        if len(t) == 1 and t.isdigit() and (whitelist is None or t in whitelist):
            mask[tid] = True
    return mask

def bias_mask_for_digit(tokenizer: GPT2Tokenizer, logits: torch.Tensor, digit_char: str) -> torch.Tensor:
    """
    Boolean mask over vocab for tokens that decode (after lstrip) exactly to digit_char (e.g., "7").
    """
    vocab_size = logits.shape[-1]
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits.device)
    for tid in range(vocab_size):
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if not s:
            continue
        t = s.lstrip()
        if t == digit_char:
            mask[tid] = True
    return mask

def nucleus_sample_with_mask(logits: torch.Tensor, mask: torch.Tensor,
                             temperature: float, top_p: float) -> Optional[int]:
    masked = logits.clone()
    masked[~mask] = -float("inf")
    if temperature and temperature > 0:
        masked = masked / temperature
    probs = torch.softmax(masked, dim=-1)

    # Nucleus sampling
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative > top_p
    if torch.any(cutoff):
        first_cut = torch.nonzero(cutoff, as_tuple=True)[0][0].item()
        sorted_probs[first_cut + 1:] = 0.0

    total = sorted_probs.sum()
    if float(total.item()) == 0.0:
        return None
    sorted_probs = sorted_probs / total
    choice = torch.multinomial(sorted_probs, num_samples=1)
    token_id = sorted_idx[choice]
    return int(token_id.item())

# --------------------------
# Main
# --------------------------
def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    # Problem
    input_grid = [[5, 6], [7, 8]]
    ground_truth = rotate_2x2_clockwise(input_grid)  # [[7,5],[8,6]]
    target_digits = [str(ground_truth[0][0]), str(ground_truth[0][1]),
                     str(ground_truth[1][0]), str(ground_truth[1][1])]  # ["7","5","8","6"]

    whitelist = None
    if RESTRICT_TO_INPUT_DIGITS:
        whitelist = {str(d) for row in input_grid for d in row}  # {"5","6","7","8"}

    prompt = (
        "Identify the pattern: Input grid [[1,2],[3,4]] -> Output [[4,1],[2,3]] (90 deg rotate). "
        "Apply to [[5,6],[7,8]]. Respond with only a 2x2 grid like [[a,b],[c,d]]."
    )
    correct_example = (
        "The pattern is a 90-degree clockwise rotation. "
        "Applying this to [[5,6],[7,8]] gives [[7,5],[8,6]]."
    )

    # PCA on prompt + correct example
    h_prompt = last_layer_hidden(model, tokenizer, prompt)
    h_correct = last_layer_hidden(model, tokenizer, correct_example)
    pca = fit_pca_dynamic([h_prompt, h_correct])
    red_prompt = pca.transform(h_prompt).mean(axis=0)
    red_target = pca.transform(h_correct).mean(axis=0)

    x = red_prompt.copy(); v = np.zeros_like(x)

    # Seed: prompt + " [["
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    generated = inputs["input_ids"]
    prompt_len = generated.shape[1]
    generated = torch.cat([generated, encode_literal(tokenizer, " [[", generated.device)], dim=1)

    token_counts = Counter()
    cont_step = 0

    def next_logits():
        nonlocal cont_step, x, v
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
        return logits[0]

    digits = []
    forced_literals = [",", "],[", ",", "]]"]

    for slot in range(4):
        # sample digit for this slot with whitelist + slot bias toward correct target digit
        logits = next_logits(); cont_step += 1
        mask = mask_tokens_for_digit(tokenizer, logits, whitelist)
        if not mask.any():
            print(f"No digit candidates at slot {slot+1}"); return

        # add slot-specific bias for the correct digit
        bias_mask = bias_mask_for_digit(tokenizer, logits, target_digits[slot])
        biased_logits = logits.clone()
        biased_logits[bias_mask] += SLOT_LOGIT_BIAS

        tid = nucleus_sample_with_mask(biased_logits, mask, TEMPERATURE, TOP_P)
        if tid is None:
            print(f"Sampling failed at slot {slot+1}"); return
        generated = torch.cat([generated, torch.tensor([[tid]], device=generated.device)], dim=1)
        digits.append(int(tokenizer.decode([tid]).strip() or "0"))

        # force next punctuation literal (except after the 4th digit)
        if slot < 4 - 1:
            generated = torch.cat([generated, encode_literal(tokenizer, forced_literals[slot], generated.device)], dim=1)

    # close with "]]"
    if forced_literals[-1] != "]]":
        generated = torch.cat([generated, encode_literal(tokenizer, "]]", generated.device)], dim=1)

    # Show result (suffix only)
    suffix = tokenizer.decode(generated[0, prompt_len:], clean_up_tokenization_spaces=False)
    grid = [[digits[0], digits[1]], [digits[2], digits[3]]]
    print("Decoded suffix:", suffix)
    print("Extracted grid:", grid)
    print("Ground truth (90° CW):", ground_truth)
    print("Match ground truth?", grid == ground_truth)

if __name__ == "__main__":
    main()
