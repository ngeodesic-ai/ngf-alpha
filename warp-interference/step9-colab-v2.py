# =====================================================================
# Setup
# =====================================================================
# numpy: 2.0.2, torch: 2.8.0+cu126, sklearn: 1.6.1, CUDA: True, transformers 4.55.2

import torch, numpy as np, re
from collections import Counter
from sklearn.decomposition import PCA
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================================
# Load model
# =====================================================================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()
vocab_size = tokenizer.vocab_size

# =====================================================================
# Prompt + hidden states -> PCA
# =====================================================================
prompt = ("Identify the pattern: Input grid [[1,2],[3,4]] -> Output [[4,1],[2,3]] (90 deg rotate). "
          "Apply to [[5,6],[7,8]]. Only output the resulting grid in the form [[a,b],[c,d]].")

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

latent = out.hidden_states[-1].squeeze(0).detach().cpu().numpy()

n_components = 2
pca = PCA(n_components=n_components)
reduced = pca.fit_transform(latent)
reduced_latent = reduced.mean(axis=0)
print(f"Selected n_components: {n_components}, Explained Variance: {pca.explained_variance_ratio_.sum():.4f}")

# =====================================================================
# Task target (from correct example)
# =====================================================================
correct_example = "The pattern is a 90-degree rotation. Applying this to [[5,6],[7,8]] gives [[8,5],[6,7]]."
corr_inputs = tokenizer(correct_example, return_tensors="pt").to(device)
with torch.no_grad():
    corr_out = model(**corr_inputs, output_hidden_states=True)
corr_latent = corr_out.hidden_states[-1].mean(dim=1).squeeze().detach().cpu().numpy()
corr_latent = corr_latent / (np.linalg.norm(corr_latent) + 1e-12)
task_target_reduced = pca.transform(corr_latent.reshape(1, -1)).squeeze()
print("Task target norm (reduced):", np.linalg.norm(task_target_reduced))

# =====================================================================
# Symbolic dynamics
# =====================================================================
dim = len(reduced_latent)
pull_strength = 3.7
gamma = 0.2

def symbolic_loop(reduced_latent, target, steps=1500, dt=0.05, damping=0.2):
    pos = reduced_latent * 15.0
    vel = np.zeros(dim)
    for _ in range(steps):
        pull = pull_strength * (target - pos)
        accel = pull - gamma * vel - damping * vel
        vel += dt * accel
        pos += dt * vel
    return pos

def symbolic_nudge(current_reduced, nudge_target, steps=1500, dt=0.05, damping=0.2):
    pos = current_reduced.copy()
    vel = np.zeros(dim)
    for _ in range(steps):
        pull = pull_strength * (nudge_target - pos)
        accel = pull - gamma * vel - damping * vel
        vel += dt * accel
        pos += dt * vel
    npos = np.linalg.norm(pos); ntgt = np.linalg.norm(nudge_target)
    if npos > 0: pos = pos * ntgt / npos
    return pos

final_pos = symbolic_loop(reduced_latent, task_target_reduced)
print("Convergence error:", np.linalg.norm(final_pos - task_target_reduced))

# =====================================================================
# Anchor from few-shot examples
# =====================================================================
train_examples = [
    "Identify the pattern: Input grid [[2,3],[4,5]] -> Output [[5,2],[3,4]] (90 deg rotate). Apply to [[2,3],[4,5]].",
    "Identify the pattern: Input grid [[1,1],[2,2]] -> Output [[2,1],[2,1]] (90 deg rotate). Apply to [[1,1],[2,2]].",
    "Identify the pattern: Input grid [[3,2],[1,4]] -> Output [[4,3],[2,1]] (90 deg rotate). Apply to [[3,2],[1,4]].",
    "Identify the pattern: Input grid [[5,6],[7,8]] -> Output [[8,5],[6,7]] (90 deg rotate). Apply to [[5,6],[7,8]].",
    "Identify the pattern: Input grid [[6,7],[8,9]] -> Output [[9,6],[7,8]] (90 deg rotate). Apply to [[6,7],[8,9]]."
]
anchor_reduced = np.zeros(dim)
for example in train_examples:
    ex_inp = tokenizer(example, return_tensors="pt").to(device)
    with torch.no_grad():
        ex_out = model(**ex_inp, output_hidden_states=True)
    ex_lat = ex_out.hidden_states[-1].mean(dim=1).squeeze().detach().cpu().numpy()
    anchor_reduced += pca.transform(ex_lat.reshape(1, -1)).squeeze()
anchor_reduced /= len(train_examples)
nudge_target = anchor_reduced

# =====================================================================
# Decoding helpers
# =====================================================================
def top_k_filter(logits, k):
    if k <= 0: return logits
    v, _ = torch.topk(logits, k)
    cutoff = v[..., -1, None]
    return torch.where(logits < cutoff, torch.tensor(float("-inf"), device=logits.device), logits)

def top_p_filter(logits, p):
    if p >= 1.0: return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    mask = cumprobs > p
    mask[..., 0] = False
    filtered = sorted_logits.masked_fill(mask, float("-inf"))
    inv = torch.empty_like(sorted_idx)
    inv.scatter_(1, sorted_idx, torch.arange(sorted_idx.size(1), device=logits.device).unsqueeze(0))
    return filtered.gather(1, inv)

def apply_frequency_penalty(logits, counts: Counter, penalty: float):
    if penalty <= 0 or not counts: return logits
    idx = torch.tensor(list(counts.keys()), device=logits.device, dtype=torch.long)
    vals = torch.tensor([counts[i] for i in idx.tolist()], device=logits.device, dtype=logits.dtype) * penalty
    out = logits.clone()
    out.index_put_((torch.arange(out.size(0), device=out.device).unsqueeze(1), idx.unsqueeze(0).expand(out.size(0), -1)),
                   -vals, accumulate=True)
    return out

def sample_from_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    logits = logits / max(temperature, 1e-6)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

ALLOWED_TOKENS = set()
for ch in list("[]0123456789, "):
    toks = tokenizer.encode(ch, add_special_tokens=False)
    for t in toks: ALLOWED_TOKENS.add(t)
for s in ["[[", "]]"]:
    for t in tokenizer.encode(s, add_special_tokens=False):
        ALLOWED_TOKENS.add(t)

def apply_token_mask(logits, allowed_ids):
    vocab = logits.size(1)
    mask = torch.ones(vocab, dtype=torch.bool, device=logits.device)
    ids = list(allowed_ids)
    if len(ids) > 0:
        mask[ids] = False
    return logits.masked_fill(mask, float("-inf"))

def cleanup(text: str) -> str:
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\[\s*,', '[', text)
    text = re.sub(r',\s*\]', ']', text)
    text = re.sub(r'\.\s*,', '.', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

# =====================================================================
# Generation with nudges + mask + early stop
# =====================================================================
nudge_interval = 10
alpha = 0.25
temperature = 0.7
top_k = 50
top_p = 0.85
freq_penalty = 0.6
max_tokens = 40
repetition_window = 6

generated = inputs["input_ids"].clone().to(device)
token_counts = Counter(generated[0].tolist())
last_tokens = []
trajectory = []

for step in range(max_tokens):
    with torch.no_grad():
        out = model(generated, output_hidden_states=True)
        base_logits = out.logits[:, -1, :]

    if (generated.shape[1] % nudge_interval) == 0 and step > 0:
        current_hidden = out.hidden_states[-1][:, -1, :]
        current_latent = current_hidden.detach().cpu().numpy().squeeze()
        reduced_current = pca.transform(current_latent.reshape(1, -1)).squeeze()
        nudged_reduced = symbolic_nudge(reduced_current, nudge_target)
        nudged_latent = pca.inverse_transform(nudged_reduced.reshape(1, -1)).squeeze()
        nudged_hidden = torch.from_numpy(nudged_latent).to(torch.float32).to(device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            nudged_logits = model.lm_head(nudged_hidden)[:, 0, :]
        logits = (1.0 - alpha) * base_logits + alpha * nudged_logits
        trajectory.append(float(np.linalg.norm(nudged_reduced)))
    else:
        logits = base_logits
        ch = out.hidden_states[-1][:, -1, :].detach().cpu().numpy().squeeze()
        rc = pca.transform(ch.reshape(1, -1)).squeeze()
        trajectory.append(float(np.linalg.norm(rc)))

    logits = apply_frequency_penalty(logits, token_counts, freq_penalty)
    logits = apply_token_mask(logits, ALLOWED_TOKENS)
    next_token = sample_from_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)

    generated = torch.cat([generated, next_token], dim=1)
    tid = int(next_token.item())
    token_counts[tid] += 1

    last_tokens.append(tid)
    if len(last_tokens) > repetition_window:
        last_tokens.pop(0)
        if len(set(last_tokens)) == 1:
            break

    close_tok = tokenizer.encode("]]", add_special_tokens=False)
    if len(close_tok) == 1 and generated[0, -1].item() == close_tok[0]:
        break
    elif len(close_tok) == 2 and generated.shape[1] >= 2:
        if generated[0, -2].item() == close_tok[0] and generated[0, -1].item() == close_tok[1]:
            break

text = tokenizer.decode(generated[0], skip_special_tokens=True)
m = re.search(r"\[\s*\[\s*\d+\s*,\s*\d+\s*\]\s*,\s*\[\s*\d+\s*,\s*\d+\s*\]\s*\]", text)
if m: text = m.group(0)
text = cleanup(text)


# --- decode ONLY the continuation, not the whole prompt ---
start_idx = inputs["input_ids"].shape[1]
continuation = tokenizer.decode(generated[0, start_idx:], skip_special_tokens=True)

# find the first valid 2x2 grid in the continuation
import re
grid_pat = r"\[\s*\[\s*\d+\s*,\s*\d+\s*\]\s*,\s*\[\s*\d+\s*,\s*\d+\s*\]\s*\]"
m = re.search(grid_pat, continuation)

if m:
    text = m.group(0)
else:
    # fallback: try the last grid anywhere in the full text
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    all_grids = re.findall(grid_pat, full_text)
    text = all_grids[-1] if all_grids else continuation

# optional cleanup
text = cleanup(text)
print("Stabilized Output:", text)

import matplotlib.pyplot as plt
plt.plot(trajectory)
plt.title("Latent Trajectory Norm over Steps")
plt.xlabel("Step")
plt.ylabel("||latent||")
plt.show()
