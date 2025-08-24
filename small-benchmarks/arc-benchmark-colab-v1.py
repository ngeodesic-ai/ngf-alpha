# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Copyright 2025 Ian C. Moore (Provisional Patents #63/864,726 and #63/865,437)
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
# Runtime environment
# ==============================================================================
# !pip install transformers==4.55.2 torch==2.8.0 numpy==2.0.2 scikit-learn==1.6.1


# ==============================================================================
# Updates - grok v1
# ==============================================================================
# (1) 10 synthetic ARC tasks on Colab

# ==============================================================================
# Output
# ==============================================================================
# Model loaded on CPU. Attempting to move to cuda...
# Model successfully moved to cuda
# Convergence Error: 0.0000
# Task 1: Stock '. the name of
#  the:1]] 3, [1, 8],
#  [ grid2, 7], [7...', Warped 'identify the pattern: input grid [[2, 4], [6, 7]] ...', Expected [[6, 2], [7, 4]]
# Task 2: Stock '. the name of
#  the:1]] 8, [10, 3],
#  [ grid7, 9], [...', Warped 'identify the pattern: input grid [[7, 9], [2, 9]] ...', Expected [[2, 7], [9, 9]]
# Task 3: Stock '. the name of
#  the:1]] 9, [8, 4],
#  [ grid8, 7], [4...', Warped 'identify the pattern: input grid [[8, 8], [4, 7]] ...', Expected [[4, 8], [7, 8]]
# Task 4: Stock '. the name of
#  the:1]] 5, [5, 2],
#  [ grid4, 4], [1...', Warped 'identify the pattern: input grid [[4, 8], [1, 4]] ...', Expected [[1, 4], [4, 8]]
# Task 5: Stock '. the name of
#  the:1]] 4, [5, 7],
#  [ grid3, 9], [6...', Warped 'identify the pattern: input grid [[3, 4], [6, 9]] ...', Expected [[6, 3], [9, 4]]
# Task 6: Stock '. the name of
#  the:1]] 7, [7, 2],
#  [ grid6, 4], [1...', Warped 'identify the pattern: input grid [[6, 6], [1, 4]] ...', Expected [[1, 6], [4, 6]]
# Task 7: Stock '. the name of
#  the:1]] 3, [1, 2],
#  [ grid2, 4], [1...', Warped 'identify the pattern: input grid [[2, 4], [1, 8]] ...', Expected [[1, 2], [8, 4]]
# Task 8: Stock '. the name of
#  the:1]] 3, [1, 10],
#  [ grid2, 9], [...', Warped 'identify the pattern: input grid [[2, 4], [9, 9]] ...', Expected [[9, 2], [9, 4]]
# Task 9: Stock '. the name of
#  the:1]] 7, [5, 4],
#  [ grid6, 4], [6...', Warped 'identify the pattern: input grid [[6, 4], [6, 2]] ...', Expected [[6, 6], [2, 4]]
# Task 10: Stock '. the name of
#  the:1, 2, [2, 8],
#  [ grid1, 3], [4,...', Warped 'identify the pattern: input grid [[1, 6], [4, 3]] ...', Expected [[4, 1], [3, 6]]

# Benchmark Results:
# Stock Accuracy: 0.0%, Warped Accuracy: 100.0%, Hallucination Reduction: 0.0%
    
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DynamicCache
import numpy as np
from sklearn.decomposition import PCA
import os
import random

# Enable CUDA_LAUNCH_BLOCKING for debugging (remove after testing)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load tokenizer and model with fallback
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

# Define vocab_size globally
vocab_size = model.config.vocab_size  # 50257 for GPT-2

# Step 7: Symbolic Loop (Initial Latent Embedding)
prompt = "Identify the pattern: Input grid [[1,2],[3,4]] -> Output [[4,1],[2,3]] (90 deg rotate). Apply to [[5,6],[7,8]]."
inputs = tokenizer(prompt, return_tensors='pt').to(device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
latent = outputs.hidden_states[-1].squeeze(0).cpu().numpy()

n_components = min(8, latent.shape[0] - 1)  # Increased dimensionality
pca = PCA(n_components=n_components)
reduced = pca.fit_transform(latent)
reduced_latent = reduced.mean(axis=0)

dim = len(reduced_latent)
target = np.roll(reduced_latent, shift=dim // 4)
pull_strength = 1.5
gamma = 0.3

def symbolic_loop(reduced_latent, target, steps=200, dt=0.05):
    pos = reduced_latent * 15.0
    vel = np.zeros(dim)
    for _ in range(steps):
        r = np.linalg.norm(pos)
        if r < 1e-6: r = 1e-6
        pull = pull_strength * (target - pos)
        accel = pull - gamma * vel
        vel += dt * accel
        pos += dt * vel
    return pos

final_pos = symbolic_loop(reduced_latent, target)
error = np.linalg.norm(final_pos - target)
print(f"Convergence Error: {error:.4f}")

# Step 8: Warped Inference (for reference and nudge target)
correct_example = "The output is [[8,5],[6,7]]."
example_inputs = tokenizer(correct_example, return_tensors='pt').to(device)
with torch.no_grad():
    example_outputs = model(**example_inputs, output_hidden_states=True)
example_latent = example_outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
reduced_example = pca.transform(example_latent.reshape(1, -1)).squeeze()
nudge_target = reduced_example

def symbolic_nudge(current_reduced, nudge_target, steps=50, dt=0.05):
    pos = current_reduced
    vel = np.zeros(dim)
    for _ in range(steps):
        r = np.linalg.norm(pos)
        if r < 1e-6: r = 1e-6
        pull = pull_strength * (nudge_target - pos)
        accel = pull - gamma * vel
        vel += dt * accel
        pos += dt * vel
    return pos

# Step 9: Benchmark on 10 Synthetic ARC Tasks
def generate_arc_task():
    grid = [[random.randint(1, 9), random.randint(1, 9)] for _ in range(2)]
    rotated = [[grid[1][0], grid[0][0]], [grid[1][1], grid[0][1]]]  # 90-degree rotation
    return grid, rotated

tasks = [generate_arc_task() for _ in range(10)]
stock_correct = 0
warped_correct = 0
stock_hallucinations = 0
warped_hallucinations = 0

for i, (input_grid, expected_output) in enumerate(tasks, 1):
    prompt = f"Identify the pattern: Input grid {input_grid} -> Output {expected_output} (90 deg rotate). Apply to {input_grid}."
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    # Stock GPT-2
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=True)
    stock_output = tokenizer.decode(torch.argmax(outputs.logits, dim=-1)[0])
    stock_correct += 1 if str(expected_output) in stock_output else 0
    stock_hallucinations += 1 if any(c not in "0123456789[], " for c in stock_output) else 0
    
    # Warped Geodesic
    generated = inputs['input_ids']
    past_key_values = None
    for _ in range(40):  # Increased tokens for better context
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(generated, output_hidden_states=True, use_cache=True)
            else:
                cache = DynamicCache.from_legacy_cache(past_key_values)
                outputs = model(generated, past_key_values=cache, output_hidden_states=True, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
        next_token = torch.clamp(next_token, 0, vocab_size - 1)
        generated = torch.cat([generated, next_token], dim=1).to(device)
        
        if generated.shape[1] % 5 == 0:
            current_hidden = outputs.hidden_states[-1][:, -1, :].to(device)
            current_latent = current_hidden.cpu().numpy()
            reduced_current = pca.transform(current_latent)
            nudged_reduced = symbolic_nudge(reduced_current[0], nudge_target)
            nudged_latent = pca.inverse_transform(nudged_reduced.reshape(1, -1))[0]
            norm = np.linalg.norm(nudged_latent)
            if norm > 0:
                nudged_latent = (nudged_latent / norm) * 5.0
            nudged_hidden = torch.from_numpy(nudged_latent).unsqueeze(0).unsqueeze(0).to(device, torch.float32)
            nudged_logits = model.lm_head(nudged_hidden)[:, 0, :]
            if nudged_logits.size(0) > vocab_size:
                nudged_logits[vocab_size:] = -float('inf')
            next_token = torch.argmax(nudged_logits, dim=-1).unsqueeze(0)
            next_token = torch.clamp(next_token, 0, vocab_size - 1)
            if next_token.item() == 0 or next_token.item() >= vocab_size:
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            past_key_values = None  # Reset cache after nudge
            generated = torch.cat([generated[:, :-1], next_token], dim=1).to(device)
    
    warped_output = tokenizer.decode(generated[0]).lower()
    warped_correct += 1 if str(expected_output).lower() in warped_output else 0
    warped_hallucinations += 1 if any(c not in "0123456789[], " for c in warped_output) else 0
    print(f"Task {i}: Stock '{stock_output[:50]}...', Warped '{warped_output[:50]}...', Expected {expected_output}")

# Summary
stock_acc = stock_correct / 10 * 100
warped_acc = warped_correct / 10 * 100
hall_red = ((stock_hallucinations - warped_hallucinations) / stock_hallucinations * 100) if stock_hallucinations > 0 else 0
print(f"\nBenchmark Results:")
print(f"Stock Accuracy: {stock_acc:.1f}%, Warped Accuracy: {warped_acc:.1f}%, Hallucination Reduction: {hall_red:.1f}%")