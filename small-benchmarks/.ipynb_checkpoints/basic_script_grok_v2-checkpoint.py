import os
import json
import torch
import requests
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.nn import functional as F

# Set up directories and download HellaSwag validation set
DATA_CACHE_DIR = "hellaswag_data"
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
data_url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
data_filename = os.path.join(DATA_CACHE_DIR, "hellaswag_val.jsonl")

if not os.path.exists(data_filename):
    print(f"Downloading {data_url} to {data_filename}...")
    resp = requests.get(data_url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(data_filename, "wb") as file, tqdm(total=total, unit="iB", unit_scale=True) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

# Load model and tokenizer using Auto classes
model_type = "gpt2-medium"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load configuration
config = AutoConfig.from_pretrained(model_type)
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_type, config=config).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_type)

# Function to render HellaSwag example into tensors
def render_example(example):
    ctx = example["ctx"]
    label = int(example["label"])
    endings = example["endings"]

    ctx_tokens = tokenizer.encode(ctx, return_tensors="pt")[0]
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = tokenizer.encode(" " + end, return_tensors="pt")[0]
        tok_rows.append(torch.cat([ctx_tokens, end_tokens]))
        mask_rows.append(torch.cat([torch.zeros(len(ctx_tokens)), torch.ones(len(end_tokens))]))

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = tok_row
        mask[i, :len(mask_row)] = mask_row

    return tokens, mask, label

# Evaluation function (limited to 100 examples)
@torch.no_grad()
def evaluate(max_examples=100):
    model.eval()
    num_correct_norm = 0
    num_total = 0

    with open(data_filename, "r") as f:
        for i, line in enumerate(f):
            if num_total >= max_examples:  # Stop after 100 examples
                break
            example = json.loads(line)
            tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            # Get logits and compute loss
            logits = model(tokens).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_tokens = tokens[..., 1:].contiguous()
            shift_losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                         shift_tokens.view(-1), reduction="none")
            shift_losses = shift_losses.view(tokens.size(0), -1)
            shift_mask = mask[..., 1:].contiguous()
            masked_shift_losses = shift_losses * shift_mask
            sum_loss = masked_shift_losses.sum(dim=1)
            avg_loss = sum_loss / shift_mask.sum(dim=1)
            pred_norm = avg_loss.argmin().item()

            num_total += 1
            num_correct_norm += int(pred_norm == label)
            if num_total % 10 == 0:  # Update progress every 10 examples
                print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

    final_acc_norm = num_correct_norm / num_total
    print(f"Final HellaSwag acc_norm (100 examples): {final_acc_norm:.4f}")

# Run evaluation
evaluate(max_examples=100)