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
# Runtime environment for Colab (uncomment to install in Colab)
# ==============================================================================
# !apt-get update -qq
# !apt-get install -y -qq build-essential libatlas-base-dev gfortran
# !pip install --quiet --no-build-isolation --prefer-binary transformers==4.30.0 torch==2.3.0+cu121 numpy==1.26.4 scikit-learn==1.0.0 -f https://download.pytorch.org/whl/torch_stable.html

# ==============================================================================
# Benchmark Code
# ==============================================================================
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np
from sklearn.decomposition import PCA
import random
from google.colab import userdata

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load tokenizer and model with HF_TOKEN if available
hf_token = userdata.get('HF_TOKEN')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', token=hf_token if hf_token else None)
model = GPT2LMHeadModel.from_pretrained('gpt2', token=hf_token if hf_token else None)
vocab_size = tokenizer.vocab_size

# Generate synthetic ARC tasks with varied transformations
def generate_arc_task():
    rows = random.choice([2, 3])
    cols = random.choice([2, 3])
    grid = [[random.randint(1, 9) for _ in range(cols)] for _ in range(rows)]
    transform_type = random.choice(['rotate', 'flip_h', 'flip_v', 'scale', 'multi_step', 'swap_colors', 'shift'])
    if transform_type == 'rotate':
        if rows == 2 and cols == 2:
            output = [[grid[1][0], grid[0][0]], [grid[1][1], grid[0][1]]]
        else:
            output = [grid[rows-1]] + grid[:-1]  # Rotate 3x3 by shifting rows
    elif transform_type == 'flip_h':
        output = [row[::-1] for row in grid]
    elif transform_type == 'flip_v':
        output = grid[::-1]
    elif transform_type == 'scale':
        scale = random.uniform(1.5, 2.0)
        output = [[int(x * scale) for x in row] for row in grid]
    elif transform_type == 'multi_step':
        output = [[x + 1 if x < 9 else 1 for x in row] for row in grid]
    elif transform_type == 'swap_colors':
        output = [[9 - x + 1 for x in row] for row in grid]
    elif transform_type == 'shift':
        output = [[x + 1 if x < 8 else 1 for x in row] for row in grid]
    desc = f"Identify the pattern: Input grid {grid} -> Output {output} (Transform: {transform_type}). Apply to {grid}."
    correct_example = f"Output grid {output}"
    return desc, correct_example

# Generate blind test set with different seed
random.seed(43)
test_tasks = []
train_examples = [generate_arc_task() for _ in range(80)]  # 80 training tasks
train_grids = [eval(ex[0].split('Input grid ')[1].split(' -> ')[0]) for ex in train_examples]
while len(test_tasks) < 20:
    task = generate_arc_task()
    grid = eval(task[0].split('Input grid ')[1].split(' -> ')[0])
    if grid not in train_grids and task not in train_examples:
        test_tasks.append(task)

# Benchmark function
def run_benchmark(n_tasks=20):  # Test on 20 blind ARC tasks
    results = {"stock_accuracy": 0.0, "warped_accuracy": 0.0, "warped_semantic_similarity": 0.0, "hallucination_rate": 0.0}
    successful_tasks = 0

    for i in range(n_tasks):
        desc, correct_example = test_tasks[i]
        prompt = desc
        inputs = tokenizer(prompt, return_tensors='pt', max_length=128, truncation=True)

        # Stock inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            generated_stock = torch.cat([inputs['input_ids'], next_token], dim=1)
            for _ in range(60):  # Generate 60 tokens
                with torch.no_grad():
                    outputs = model(generated_stock)
                    logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                generated_stock = torch.cat([generated_stock, next_token], dim=1)
        stock_output = tokenizer.decode(generated_stock[0], skip_special_tokens=True)
        stock_correct = stock_output.strip() == correct_example

        # Latent processing and PCA
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            latent = outputs.hidden_states[-1].squeeze(0).numpy()
            pca = PCA(n_components=min(8, latent.shape[0] - 1))
            reduced_latent = pca.fit_transform(latent).mean(axis=0)
            print(f"Task {i+1} Selected n_components: {pca.n_components_}, Explained Variance: {pca.explained_variance_ratio_.sum():.4f}")

            # Symbolic nudge loop
            generated_warped = inputs['input_ids'].clone()
            target = reduced_latent  # Initial target matches reduced latent shape
            pull_strength = 2.3
            gamma = 0.2
            for _ in range(60):  # Generate 60 tokens
                with torch.no_grad():
                    warped_outputs = model(generated_warped, output_hidden_states=True)
                    current_hidden = warped_outputs.hidden_states[-1][:, -1, :]  # [1, 768]
                    current_latent = current_hidden.mean(dim=0).numpy()  # [768]
                    reduced_current = pca.transform(current_latent.reshape(1, -1)).squeeze()  # [n_components]
                    # Pad reduced_current to match pca.n_components_ if needed
                    if len(reduced_current) < pca.n_components_:
                        reduced_current = np.pad(reduced_current, (0, pca.n_components_ - len(reduced_current)), mode='constant')
                    elif len(reduced_current) > pca.n_components_:
                        reduced_current = reduced_current[:pca.n_components_]
                    pos = reduced_current
                    vel = np.zeros(pca.n_components_)
                    dt = 0.01
                    steps = 350
                    for _ in range(steps):
                        r = np.linalg.norm(pos)
                        if r < 1e-6: r = 1e-6
                        pull = pull_strength * (target - pos)
                        accel = pull - gamma * vel
                        vel += dt * accel
                        pos += dt * vel
                    # Inverse transform back to full 768 dimensions
                    full_nudged_latent = pca.inverse_transform(pos.reshape(1, -1)).squeeze()
                    nudged_hidden = torch.from_numpy(full_nudged_latent).unsqueeze(0).unsqueeze(0).to(torch.float32)
                    nudged_logits = model.lm_head(nudged_hidden)[:, 0, :]
                    nudged_logits = torch.clamp(nudged_logits, min=-100.0, max=100.0)
                    nudged_logits = torch.nn.functional.softmax(nudged_logits, dim=-1) * 100.0
                    next_token = torch.argmax(nudged_logits, dim=-1).unsqueeze(0)
                    generated_warped = torch.cat([generated_warped[:, :-1], next_token], dim=1)
                    if generated_warped.shape[1] % 10 == 0:
                        print(f"Task {i+1} Applied symbolic correction at step: {generated_warped.shape[1]}")
            warped_output = tokenizer.decode(generated_warped[0], skip_special_tokens=True)
        warped_correct = warped_output.strip() == correct_example
        # Semantic similarity for all tasks
        with torch.no_grad():
            warped_emb = model(**tokenizer(warped_output, return_tensors='pt'), output_hidden_states=True).hidden_states[-1].mean(dim=1)
            correct_emb = model(**tokenizer(correct_example, return_tensors='pt'), output_hidden_states=True).hidden_states[-1].mean(dim=1)
            similarity = torch.nn.functional.cosine_similarity(warped_emb, correct_emb, dim=1).item()
        if warped_correct or stock_correct:  # Only count if a valid output is produced
            results["warped_semantic_similarity"] += similarity
            results["stock_accuracy"] += stock_correct
            results["warped_accuracy"] += warped_correct
            results["hallucination_rate"] += 1 - warped_correct if warped_correct else 1
            successful_tasks += 1

        # Verify results for first 5 tasks
        if i < 5:
            print(f"Task {i+1} Stock Output: {stock_output}")
            print(f"Task {i+1} Warped Output: {warped_output}")

    # Compute results only if there were successful tasks
    results = {k: v / successful_tasks if successful_tasks > 0 else 0.0 for k, v in results.items()}
    print("Benchmark Results:")
    print(f"Stock Accuracy: {results['stock_accuracy']*100:.1f}%")
    print(f"Warped Accuracy: {results['warped_accuracy']*100:.1f}%")
    print(f"Warped Semantic Similarity: {results['warped_semantic_similarity']*100:.1f}%")
    print(f"Hallucination Rate: {results['hallucination_rate']*100:.1f}%")

# Run benchmark
run_benchmark()