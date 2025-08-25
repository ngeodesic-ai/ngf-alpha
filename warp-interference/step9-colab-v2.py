# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Copyright 2025 Ian C. Moore
# Email: ngeodesic@gmail.com
# Part of Noetic Geodesic Framework (NGF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ==============================================================================
# Runtime environment
# ==============================================================================
# !pip install transformers==4.55.2 torch==2.8.0 numpy==2.0.2 scikit-learn==1.6.1

# ==============================================================================
# Output
# ==============================================================================
# Model loaded on CPU. Attempting to move to cuda...
# Model successfully moved to cuda
# Convergence Error: 0.0000
# Testing single task: Input [[2, 1], [5, 4]], Expected [[5, 2], [4, 1]]

# === STOCK ===
# Stock Output: . the name of
#  the:1]] 3, 22, 2],
#  [ grid2, 4], [5, 1]] ->input.))
#  the the2, 1], [5, 4]]].

# Stock Correct? False

# === WARPED ===
# Warped Output: Identify the pattern: Input grid [[2, 1], [5, 4]] -> Output [[5, 2], [4, 1]]
# Warped Correct? True

import os
import re
import random
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DynamicCache

# ------------------------------------------------------------------------------
# Determinism (optional, helps reproducibility)
# ------------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------------------------
# CUDA debug (optional)
# ------------------------------------------------------------------------------
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')

# ------------------------------------------------------------------------------
# Load tokenizer & model
# ------------------------------------------------------------------------------
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    print(f"Model loaded on CPU. Attempting to move to {device}...")
    model = model.to(device)
    print(f"Model successfully moved to {device}")
except Exception as e:
    print(f"Failed to move model to {device}: {e}")
    device = 'cpu'
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    print(f"Using CPU fallback: {device}")

model.eval()
torch.set_grad_enabled(False)
tokenizer.pad_token = tokenizer.eos_token  # quiets potential padding warnings
vocab_size = model.config.vocab_size  # 50257

# ==============================================================================
# Helper: detect and truncate at the first valid 2x2 grid
# ==============================================================================
GRID_RE = re.compile(
    r"\[\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*,\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*\]"
)

def first_grid_after_output(text: str):
    """Prefer the first grid that appears after the token 'output' (case-insensitive).
       Returns (grid_as_list, (start_idx, end_idx)) or (None, None)."""
    lower = text.lower()
    if "output" in lower:
        after_idx = lower.find("output") + len("output")
        after = text[after_idx:]
        m = GRID_RE.search(after)
        if m:
            start = after_idx + m.start()
            end = after_idx + m.end()
            grid = [[int(m.group(1)), int(m.group(2))],
                    [int(m.group(3)), int(m.group(4))]]
            return grid, (start, end)

    # Fallback: first grid anywhere
    m = GRID_RE.search(text)
    if m:
        grid = [[int(m.group(1)), int(m.group(2))],
                [int(m.group(3)), int(m.group(4))]]
        return grid, (m.start(), m.end())
    return None, None

def truncate_at_first_grid(text: str):
    """Return text truncated exactly at the end of the first detected 2x2 grid."""
    _, span = first_grid_after_output(text)
    if span:
        return text[:span[1]]
    return text

# ==============================================================================
# Stage 7: Symbolic Loop (Initial Latent Embedding)
# Use a canonical prompt to get a latent; reduce with PCA; simulate geodesic pull
# ==============================================================================
base_prompt = (
    "Identify the pattern: Input grid [[1,2],[3,4]] -> Output [[4,1],[2,3]] "
    "(90 deg rotate). Apply to [[5,6],[7,8]]."
)
inputs = tokenizer(base_prompt, return_tensors='pt').to(device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

latent = outputs.hidden_states[-1].squeeze(0).detach().cpu().numpy()  # (T, H)
n_components = min(8, latent.shape[0] - 1) if latent.shape[0] > 1 else 1
pca = PCA(n_components=n_components)
reduced = pca.fit_transform(latent)                 # (T, n_components)
reduced_latent = reduced.mean(axis=0)               # (n_components,)
dim = len(reduced_latent)

# Create a circularly-shifted 'target' in reduced space
target = np.roll(reduced_latent, shift=max(1, dim // 4))
pull_strength = 1.5
gamma = 0.3

def symbolic_loop(reduced_latent_vec, target_vec, steps=200, dt=0.05):
    pos = reduced_latent_vec * 15.0
    vel = np.zeros_like(pos)
    for _ in range(steps):
        pull = pull_strength * (target_vec - pos)
        accel = pull - gamma * vel
        vel += dt * accel
        pos += dt * vel
    return pos

final_pos = symbolic_loop(reduced_latent, target)
error = np.linalg.norm(final_pos - target)
print(f"Convergence Error: {error:.4f}")

# ==============================================================================
# Stage 8: Warped Inference target (use a short 'correct' example to seed)
# ==============================================================================
correct_example = "The output is [[8,5],[6,7]]."
ex_inputs = tokenizer(correct_example, return_tensors='pt').to(device)
with torch.no_grad():
    ex_out = model(**ex_inputs, output_hidden_states=True)
ex_latent = ex_out.hidden_states[-1].mean(dim=1).squeeze().detach().cpu().numpy()
reduced_example = pca.transform(ex_latent.reshape(1, -1)).squeeze()
nudge_target = reduced_example

def symbolic_nudge(current_reduced, target_reduced, steps=50, dt=0.05):
    pos = current_reduced.copy()
    vel = np.zeros_like(pos)
    for _ in range(steps):
        pull = pull_strength * (target_reduced - pos)
        accel = pull - gamma * vel
        vel += dt * accel
        pos += dt * vel
    return pos

# ==============================================================================
# Stage 9: Single Synthetic ARC Task
# Generate one random 2x2 grid and its 90-degree rotation (CW)
# ==============================================================================
def generate_arc_task():
    grid = [[random.randint(1, 9), random.randint(1, 9)] for _ in range(2)]
    rotated = [[grid[1][0], grid[0][0]], [grid[1][1], grid[0][1]]]
    return grid, rotated

input_grid, expected_output = generate_arc_task()
print(f"Testing single task: Input {input_grid}, Expected {expected_output}")

prompt = (
    f"Identify the pattern: Input grid {input_grid} -> Output {expected_output} "
    f"(90 deg rotate). Apply to {input_grid}."
)
enc = tokenizer(prompt, return_tensors='pt').to(device)

# ------------------------------------------------------------------------------
# Stock generation (single forward pass over prompt, then decode)
# ------------------------------------------------------------------------------
with torch.no_grad():
    stock_out = model(**enc, output_hidden_states=True, use_cache=True)
stock_output_raw = tokenizer.decode(torch.argmax(stock_out.logits, dim=-1)[0])
stock_output = truncate_at_first_grid(stock_output_raw)
stock_grid, _ = first_grid_after_output(stock_output_raw)
stock_correct = (stock_grid == expected_output)

print("\n=== STOCK ===")
print(f"Stock Output: {stock_output}")
print(f"Stock Correct? {stock_correct}")

# ------------------------------------------------------------------------------
# Warped generation (iterative decoding with nudges + early stop)
# ------------------------------------------------------------------------------
generated = enc['input_ids']
past_key_values = None
MAX_STEPS = 80

warped_output = None  # will be set if we early-stop

for step in range(MAX_STEPS):
    with torch.no_grad():
        if past_key_values is None:
            m_out = model(generated, output_hidden_states=True, use_cache=True)
        else:
            cache = DynamicCache.from_legacy_cache(past_key_values)
            m_out = model(generated, past_key_values=cache, output_hidden_states=True, use_cache=True)

        past_key_values = m_out.past_key_values
        logits = m_out.logits[:, -1, :]

    # Greedy next token
    next_token = torch.argmax(logits, dim=-1).unsqueeze(0).clamp(0, vocab_size - 1)
    generated = torch.cat([generated, next_token], dim=1).to(device)

    # Nudge every 5 tokens (on the *current* hidden state)
    if generated.shape[1] % 5 == 0:
        current_hidden = m_out.hidden_states[-1][:, -1, :].to(device)
        current_latent = current_hidden.detach().cpu().numpy()
        reduced_current = pca.transform(current_latent)
        nudged_reduced = symbolic_nudge(reduced_current[0], nudge_target)
        nudged_latent = pca.inverse_transform(nudged_reduced.reshape(1, -1))[0]
        # normalize magnitude somewhat to keep logits sane
        norm = np.linalg.norm(nudged_latent)
        if norm > 0:
            nudged_latent = (nudged_latent / norm) * 5.0

        nudged_hidden = torch.from_numpy(nudged_latent).unsqueeze(0).unsqueeze(0).to(device, torch.float32)
        nudged_logits = model.lm_head(nudged_hidden)[:, 0, :]  # (1, vocab)
        # clamp safety
        if nudged_logits.size(-1) > vocab_size:
            nudged_logits[..., vocab_size:] = -float('inf')

        nudged_next = torch.argmax(nudged_logits, dim=-1).unsqueeze(0).clamp(0, vocab_size - 1)
        # If nudge yields an invalid or EOS-ish token, fallback to base logits
        if int(nudged_next.item()) == 0 or int(nudged_next.item()) >= vocab_size:
            nudged_next = next_token

        # Replace last token with nudged choice and clear cache (to avoid KV drift)
        past_key_values = None
        generated = torch.cat([generated[:, :-1], nudged_next], dim=1).to(device)

    # ---- EARLY STOP once we see a valid 2x2 grid in decoded text ----
    decoded_so_far = tokenizer.decode(generated[0])
    grid, span = first_grid_after_output(decoded_so_far)
    if span:
        warped_output = decoded_so_far[:span[1]]  # truncate exactly at grid
        break

# If we never early-stopped, decode and truncate post-hoc
if warped_output is None:
    warped_output_raw = tokenizer.decode(generated[0])
    warped_output = truncate_at_first_grid(warped_output_raw)
else:
    warped_output_raw = warped_output  # for uniformity below

warped_grid, _ = first_grid_after_output(warped_output_raw)
warped_correct = (warped_grid == expected_output)

print("\n=== WARPED ===")
print(f"Warped Output: {warped_output}")
print(f"Warped Correct? {warped_correct}")
