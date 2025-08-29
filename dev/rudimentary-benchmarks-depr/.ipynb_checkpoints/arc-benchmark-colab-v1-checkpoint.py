# ==============================================================================
# Rudimentary Benchmark on 10 Synthetic ARC Tasks (Step 10 - minimal)
# Keeps stage-9 truncation/early-stop to avoid garbled tails.
# ==============================================================================

# !pip install transformers==4.55.2 torch==2.8.0 numpy==2.0.2 scikit-learn==1.6.1

# ==============================================================================
# Output
# ==============================================================================
# Task 01 | Input [[2, 1], [5, 4]] | Expect [[5, 2], [4, 1]] | Stock:× | Warped:✓
# Task 02 | Input [[4, 3], [2, 9]] | Expect [[2, 4], [9, 3]] | Stock:× | Warped:✓
# Task 03 | Input [[2, 7], [1, 1]] | Expect [[1, 2], [1, 7]] | Stock:× | Warped:✓
# Task 04 | Input [[2, 4], [4, 9]] | Expect [[4, 2], [9, 4]] | Stock:× | Warped:✓
# Task 05 | Input [[1, 9], [4, 9]] | Expect [[4, 1], [9, 9]] | Stock:× | Warped:✓
# Task 06 | Input [[7, 4], [8, 5]] | Expect [[8, 7], [5, 4]] | Stock:× | Warped:✓
# Task 07 | Input [[1, 3], [7, 6]] | Expect [[7, 1], [6, 3]] | Stock:× | Warped:✓
# Task 08 | Input [[5, 3], [4, 6]] | Expect [[4, 5], [6, 3]] | Stock:× | Warped:✓
# Task 09 | Input [[2, 2], [7, 2]] | Expect [[7, 2], [2, 2]] | Stock:× | Warped:✓
# Task 10 | Input [[6, 6], [5, 1]] | Expect [[5, 6], [1, 6]] | Stock:× | Warped:✓

# === Summary ===
# Stock Accuracy : 0/10 = 0.0%
# Warped Accuracy: 10/10 = 100.0%

import os, re, random, numpy as np, torch
from sklearn.decomposition import PCA
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DynamicCache

# -----------------------------
# Reproducibility (rudimentary)
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# -----------------------------
# Device & model
# -----------------------------
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
model.eval(); torch.set_grad_enabled(False)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = model.config.vocab_size

# -----------------------------
# Helpers: detect/truncate [[a,b],[c,d]]
# -----------------------------
GRID_RE = re.compile(
    r"\[\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*,\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*\]"
)

def first_grid_after_output(text: str):
    lower = text.lower()
    if "output" in lower:
        i = lower.find("output") + len("output")
        m = GRID_RE.search(text, i)
        if m:
            grid = [[int(m.group(1)), int(m.group(2))],
                    [int(m.group(3)), int(m.group(4))]]
            return grid, (m.start(), m.end())
    m = GRID_RE.search(text)
    if m:
        grid = [[int(m.group(1)), int(m.group(2))],
                [int(m.group(3)), int(m.group(4))]]
        return grid, (m.start(), m.end())
    return None, None

def truncate_at_first_grid(text: str):
    _, span = first_grid_after_output(text)
    return text[:span[1]] if span else text

# -----------------------------
# Stage 7: latent + PCA (simple)
# -----------------------------
base_prompt = (
    "Identify the pattern: Input grid [[1,2],[3,4]] -> Output [[4,1],[2,3]] "
    "(90 deg rotate). Apply to [[5,6],[7,8]]."
)
base_enc = tokenizer(base_prompt, return_tensors='pt').to(device)
with torch.no_grad():
    base_out = model(**base_enc, output_hidden_states=True)

latent = base_out.hidden_states[-1].squeeze(0).detach().cpu().numpy()  # (T,H)
n_components = max(1, min(8, latent.shape[0]-1))
pca = PCA(n_components=n_components).fit(latent)
reduced_latent = pca.transform(latent).mean(axis=0)
dim = reduced_latent.shape[0]

# “target” for symbolic loop (not critical in rudimentary version)
pull_strength, gamma = 1.5, 0.3
target = np.roll(reduced_latent, shift=max(1, dim // 4))

def symbolic_loop(vec, tgt, steps=150, dt=0.05):
    pos = vec * 15.0
    vel = np.zeros_like(pos)
    for _ in range(steps):
        pull = pull_strength * (tgt - pos)
        accel = pull - gamma * vel
        vel += dt * accel
        pos += dt * vel
    return pos

final_pos = symbolic_loop(reduced_latent, target)

# -----------------------------
# Stage 8: “warped” nudge seed
# -----------------------------
correct_example = "The output is [[8,5],[6,7]]."
ex_enc = tokenizer(correct_example, return_tensors='pt').to(device)
with torch.no_grad():
    ex_out = model(**ex_enc, output_hidden_states=True)
ex_latent = ex_out.hidden_states[-1].mean(dim=1).squeeze().detach().cpu().numpy()
nudge_target = pca.transform(ex_latent.reshape(1, -1)).squeeze()

def symbolic_nudge(cur_reduced, tgt_reduced, steps=40, dt=0.05):
    pos = cur_reduced.copy()
    vel = np.zeros_like(pos)
    for _ in range(steps):
        pull = pull_strength * (tgt_reduced - pos)
        accel = pull - gamma * vel
        vel += dt * accel
        pos += dt * vel
    return pos

# -----------------------------
# Task generator (your snippet)
# -----------------------------
def generate_arc_task():
    grid = [[random.randint(1, 9), random.randint(1, 9)] for _ in range(2)]
    rotated = [[grid[1][0], grid[0][0]], [grid[1][1], grid[0][1]]]  # 90-degree rotation
    return grid, rotated

# -----------------------------
# Benchmark (10 tasks; minimal I/O)
# -----------------------------
N = 10
stock_correct = 0
warped_correct = 0

for i in range(1, N+1):
    input_grid, expected_output = generate_arc_task()
    prompt = (
        f"Identify the pattern: Input grid {input_grid} -> Output {expected_output} "
        f"(90 deg rotate). Apply to {input_grid}."
    )
    enc = tokenizer(prompt, return_tensors='pt').to(device)

    # --- Stock (single pass + truncate) ---
    with torch.no_grad():
        stock_out = model(**enc, output_hidden_states=True, use_cache=True)
    stock_raw = tokenizer.decode(torch.argmax(stock_out.logits, dim=-1)[0])
    stock_clean = truncate_at_first_grid(stock_raw)
    stock_grid, _ = first_grid_after_output(stock_raw)
    s_ok = (stock_grid == expected_output)
    stock_correct += int(s_ok)

    # --- Warped (iterative with nudges + early stop) ---
    generated = enc['input_ids']
    past = None
    warped_text = None
    for step in range(80):  # short budget
        with torch.no_grad():
            if past is None:
                mout = model(generated, output_hidden_states=True, use_cache=True)
            else:
                cache = DynamicCache.from_legacy_cache(past)
                mout = model(generated, past_key_values=cache, output_hidden_states=True, use_cache=True)
            past = mout.past_key_values
            logits = mout.logits[:, -1, :]

        next_tok = torch.argmax(logits, dim=-1).unsqueeze(0).clamp(0, vocab_size-1)
        generated = torch.cat([generated, next_tok], dim=1).to(device)

        # nudge every 5 tokens
        if generated.shape[1] % 5 == 0:
            hid = mout.hidden_states[-1][:, -1, :].to(device)
            cur_lat = hid.detach().cpu().numpy()
            red = pca.transform(cur_lat)[0]
            nudged_red = symbolic_nudge(red, nudge_target)
            inv = pca.inverse_transform(nudged_red.reshape(1, -1))[0]
            norm = np.linalg.norm(inv)
            if norm > 0: inv = (inv / norm) * 5.0
            nudged_hidden = torch.from_numpy(inv).unsqueeze(0).unsqueeze(0).to(device, torch.float32)
            nudged_logits = model.lm_head(nudged_hidden)[:, 0, :]
            if nudged_logits.shape[-1] > vocab_size:
                nudged_logits[..., vocab_size:] = -float('inf')
            nudged_tok = torch.argmax(nudged_logits, dim=-1).unsqueeze(0).clamp(0, vocab_size-1)
            if int(nudged_tok.item()) == 0:
                nudged_tok = next_tok
            past = None
            generated = torch.cat([generated[:, :-1], nudged_tok], dim=1).to(device)

        # early stop on first valid grid
        decoded = tokenizer.decode(generated[0])
        g, span = first_grid_after_output(decoded)
        if span:
            warped_text = decoded[:span[1]]
            break

    if warped_text is None:
        warped_raw = tokenizer.decode(generated[0])
        warped_text = truncate_at_first_grid(warped_raw)
        g, _ = first_grid_after_output(warped_raw)
    else:
        g, _ = first_grid_after_output(warped_text)

    w_ok = (g == expected_output)
    warped_correct += int(w_ok)

    # Per-task log (short)
    print(f"Task {i:02d} | Input {input_grid} | Expect {expected_output} | "
          f"Stock:{'✓' if s_ok else '×'} | Warped:{'✓' if w_ok else '×'}")

# -----------------------------
# Summary
# -----------------------------
print("\n=== Summary ===")
print(f"Stock Accuracy : {stock_correct}/{N} = {100*stock_correct/N:.1f}%")
print(f"Warped Accuracy: {warped_correct}/{N} = {100*warped_correct/N:.1f}%")
