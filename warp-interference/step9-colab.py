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
# !apt-get update
# !apt-get install -y build-essential libatlas-base-dev gfortran
# !pip install --no-build-isolation --prefer-binary transformers==4.55.2 torch==2.8.0+cu126 numpy==2.0.2 scikit-learn==1.0.0

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np
from sklearn.decomposition import PCA

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
vocab_size = tokenizer.vocab_size

# Prompt and initial processing
prompt = "Identify the pattern: Input grid [[1,2],[3,4]] -> Output [[4,1],[2,3]] (90 deg rotate). Apply to [[5,6],[7,8]]."
inputs = tokenizer(prompt, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
latent = outputs.hidden_states[-1].squeeze(0).numpy()

# Step 1: Dimensionality Reduction with Fixed PCA
n_components = 2
pca = PCA(n_components=n_components)
reduced = pca.fit_transform(latent)
reduced_latent = reduced.mean(axis=0)
print(f"Selected n_components: {n_components}, Explained Variance: {pca.explained_variance_ratio_.sum():.4f}")

# Step 2: Symbolic Loop (Step 7) with Task-Specific Target
dim = len(reduced_latent)
correct_example = "The pattern is a 90-degree rotation. Applying this to [[5,6],[7,8]] gives [[8,5],[6,7]]."
correct_inputs = tokenizer(correct_example, return_tensors='pt')
with torch.no_grad():
    correct_outputs = model(**correct_inputs, output_hidden_states=True)
correct_latent = correct_outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()
# Normalize correct_latent
# correct_latent = correct_latent / np.linalg.norm(correct_latent)

task_target_reduced = pca.transform(correct_latent.reshape(1, -1)).squeeze()
print(f"Task Target Norm: {np.linalg.norm(task_target_reduced):.4f}")

def symbolic_loop(reduced_latent, target, steps=1100, dt=0.05, damping=0.2):
    pos = reduced_latent * 15.0
    vel = np.zeros(dim)
    for _ in range(steps):
        r = np.linalg.norm(pos)
        if r < 1e-6: r = 1e-6
        pull = pull_strength * (target - pos)
        accel = pull - gamma * vel - damping * vel
        vel += dt * accel
        pos += dt * vel
    return pos

pull_strength = 3.2  # Increased pull strength
gamma = 0.2
final_pos = symbolic_loop(reduced_latent, task_target_reduced)
error = np.linalg.norm(final_pos - task_target_reduced)
print(f"Convergence Error: {error:.4f}")

# Step 8: Symbolic Nudge with Expanded Training Set
train_examples = [
    "Identify the pattern: Input grid [[2,3],[4,5]] -> Output [[5,2],[3,4]] (90 deg rotate). Apply to [[2,3],[4,5]].",
    "Identify the pattern: Input grid [[1,1],[2,2]] -> Output [[2,1],[2,1]] (90 deg rotate). Apply to [[1,1],[2,2]].",
    "Identify the pattern: Input grid [[3,2],[1,4]] -> Output [[4,3],[2,1]] (90 deg rotate). Apply to [[3,2],[1,4]].",
    "Identify the pattern: Input grid [[5,6],[7,8]] -> Output [[8,5],[6,7]] (90 deg rotate). Apply to [[5,6],[7,8]].",
    "Identify the pattern: Input grid [[6,7],[8,9]] -> Output [[9,6],[7,8]] (90 deg rotate). Apply to [[6,7],[8,9]]."
]
anchor_reduced = np.zeros(dim)
for example in train_examples:
    train_inputs = tokenizer(example, return_tensors='pt')
    with torch.no_grad():
        train_outputs = model(**train_inputs, output_hidden_states=True)
    train_latent = train_outputs.hidden_states[-1].mean(dim=1).squeeze().numpy()
    anchor_reduced += pca.transform(train_latent.reshape(1, -1)).squeeze()
anchor_reduced /= len(train_examples)
nudge_target = anchor_reduced

def symbolic_nudge(current_reduced, nudge_target, steps=1100, dt=0.05, damping=0.2):
    pos = current_reduced
    vel = np.zeros(dim)
    for _ in range(steps):
        r = np.linalg.norm(pos)
        if r < 1e-6: r = 1e-6
        pull = pull_strength * (nudge_target - pos)
        accel = pull - gamma * vel - damping * vel
        vel += dt * accel
        pos += dt * vel
    pos = pos * np.linalg.norm(nudge_target) / (np.linalg.norm(pos) if np.linalg.norm(pos) > 0 else 1.0)
    return pos

# Generation loop with symbolic nudge
max_tokens = 50
generated = inputs['input_ids'].clone()
last_output = ""
for i in range(max_tokens):
    with torch.no_grad():
        outputs = model(generated, output_hidden_states=True)
        logits = outputs.logits[:, -1, :]
        if torch.any(logits.isnan()):
            print("NaN detected in logits")
            break
    next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
    if next_token.item() >= vocab_size:
        print(f"Invalid token ID: {next_token.item()}, clamping to 0")
        next_token = torch.tensor([[0]])
    generated = torch.cat([generated, next_token], dim=1)
    generated = torch.clamp(generated, 0, vocab_size - 1)
    
    if generated.shape[1] % 5 == 0:
        current_hidden = outputs.hidden_states[-1][:, -1, :]
        current_latent = current_hidden.numpy()
        reduced_current = pca.transform(current_latent.reshape(1, -1)).squeeze()
        nudged_reduced = symbolic_nudge(reduced_current, nudge_target)
        nudged_latent = pca.inverse_transform(nudged_reduced.reshape(1, -1)).squeeze()
        nudged_hidden = torch.from_numpy(nudged_latent).unsqueeze(0).unsqueeze(0).to(torch.float32)
        nudged_logits = model.lm_head(nudged_hidden)[:, 0, :]
        nudged_logits = torch.clamp(nudged_logits, min=-100.0, max=100.0)
        nudged_logits = torch.nn.functional.softmax(nudged_logits, dim=-1) * 100.0
        print(f"Nudge logits shape: {nudged_logits.shape}, min: {torch.min(nudged_logits)}, max: {torch.max(nudged_logits)}")
        next_token = torch.argmax(nudged_logits, dim=-1).unsqueeze(0)
        if next_token.item() >= vocab_size or torch.all(nudged_logits == nudged_logits[0, 0]):
            print(f"Invalid or uniform nudged token ID: {next_token.item()}, falling back to original logit")
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
        generated = torch.cat([generated[:, :-1], next_token], dim=1)
        print("Applied symbolic correction at step:", generated.shape[1])
    # Stop if last 10 tokens repeat
    current_output = tokenizer.decode(generated[0], skip_special_tokens=True)
    if last_output and current_output[-10:] == last_output[-10:] and len(current_output) > 20:
        print("Repetition detected in last 10 tokens, stopping early")
        break
    last_output = current_output

output = tokenizer.decode(generated[0], skip_special_tokens=True)
print("Stabilized Output:", output)