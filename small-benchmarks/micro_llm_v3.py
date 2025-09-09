import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import argparse

"""
python3 micro_llm_v3.py \
  --data latent_arc_20k.npz \
  --num_classes 5 --batch_size 128 \
  --epochs 6 --lr 1e-3 --weight_decay 1e-4 \
  --seed 42 \
  --save_ckpt micro_latent_arc.pth \
  --save_stats micro_stats.npz

python3 micro_llm_v3.py \
  --data latent_arc_20k.npz \
  --num_classes 5 --batch_size 256 \
  --epochs 0 --seed 42 --eval_only \
  --load_ckpt micro_latent_arc.pth \
  --load_stats micro_stats.npz

"""

# Mock synthetic traces (replace with real import from stage11_benchmark_latest.py)
def make_synthetic_traces(samples=1000, latent_dim=64, noise=0.05):
    np.random.seed(42)  # Reproducible
    latents = np.random.randn(samples, latent_dim) + noise * np.random.randn(samples, latent_dim)
    labels = np.random.randint(0, 3, samples)  # 0: flip_h, 1: flip_v, 2: rotate
    return latents, labels

# Dataset
class LatentARCDataset(Dataset):
    def __init__(self, latents, labels):
        self.latents = torch.tensor(latents, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.latents)
    def __getitem__(self, idx): return self.latents[idx], self.labels[idx]

# Ultra-light Micro-LLM
class MicroLLM(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=1, num_heads=2, num_classes=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.margin_loss = nn.MarginRankingLoss(margin=0.044)  # NGF (Appendix A Eq. 2)

    def forward(self, x):
        x = self.embedding(x.unsqueeze(1))  # [B, 1, H]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pool
        return self.fc(x)

# --- replace your train() with this ---
def train(model, dataloader, epochs=1, lr=1e-4, weight_decay=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        running = 0.0  # scalar accumulator
        n_batches = 0
        for latents, labels in dataloader:
            outputs = model(latents)
            loss = criterion(outputs, labels)
            # margin widening (top-2 gap)
            logits_sorted = outputs.sort(dim=1, descending=True).values
            margin_target = torch.ones_like(logits_sorted[:, 0])
            margin_loss = model.margin_loss(logits_sorted[:, 0], logits_sorted[:, 1], margin_target)
            total = loss + 0.5 * margin_loss

            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total.backward()
            optimizer.step()

            running += float(total.item())
            n_batches += 1
        print(f"Epoch {epoch+1}, Avg Loss: {running / max(1, n_batches):.4f}")
    return model



# Validation
def validate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for latents, labels in dataloader:
            outputs = model(latents)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="", help="NPZ from sim: (x0,x_star,label,name)")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--save_ckpt", type=str, default="", help="path to save trained model")
    parser.add_argument("--load_ckpt", type=str, default="", help="path to load model for eval")
    parser.add_argument("--save_stats", type=str, default="", help="path to save scaler/indices")
    parser.add_argument("--load_stats", type=str, default="", help="path to load scaler/indices")
    parser.add_argument("--eval_only", action="store_true", help="skip training; just eval")


    args = parser.parse_args()

    # seeding
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    # load data
    if args.data:
        D = np.load(args.data)
        X = D["x0"].astype(np.float32)          # or np.concatenate([D["x0"], D["x_star"]-D["x0"]], 1)
        y = D["label"].astype(np.int64)
    else:
        # fallback (dev only)
        X, y = make_synthetic_traces(args.samples, args.latent_dim)

    # after you build X, y
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    idx = np.random.permutation(len(X))
    k = int(len(X) * (1.0 - args.val_frac))
    train_idx, val_idx = idx[:k], idx[k:]
    
    # compute z-score on train only, persist
    mu = X[train_idx].mean(axis=0)
    sigma = X[train_idx].std(axis=0) + 1e-8
    Xz = (X - mu) / sigma
    
    if args.save_stats:
        np.savez(args.save_stats, mu=mu, sigma=sigma, train_idx=train_idx, val_idx=val_idx)
    if args.load_stats:
        S = np.load(args.load_stats)
        mu, sigma = S["mu"], S["sigma"]
        train_idx, val_idx = S["train_idx"], S["val_idx"]
        Xz = (X - mu) / sigma
    
    Xtr, Ytr = Xz[train_idx], y[train_idx]
    Xva, Yva = Xz[val_idx], y[val_idx]

    # Generate data
    latents, labels = make_synthetic_traces(args.samples, args.latent_dim)
    dataset = LatentARCDataset(latents, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # dataloaders (use args.batch_size)
    train_loader = DataLoader(LatentARCDataset(Xtr, Ytr), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(LatentARCDataset(Xva, Yva), batch_size=args.batch_size, shuffle=False, num_workers=0)

    # model (use args.num_classes)
    model = MicroLLM(input_dim=Xz.shape[1], num_classes=args.num_classes)
    
    # training (use args.lr/weight_decay)
    model = train(model, train_loader, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)


    if not args.eval_only and args.epochs > 0:
        model = train(model, train_loader, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)
        if args.save_ckpt:
            torch.save({"state_dict": model.state_dict()}, args.save_ckpt)
    
    # If evaluating, load checkpoint if provided
    if args.load_ckpt:
        sd = torch.load(args.load_ckpt, map_location="cpu")
        model.load_state_dict(sd["state_dict"])
    

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            trues.append(yb.cpu().numpy())
    yhat = np.concatenate(preds); ytrue = np.concatenate(trues)
    acc = float(accuracy_score(ytrue, yhat))
    P,R,F1,_ = precision_recall_fscore_support(ytrue, yhat, average="macro", zero_division=0)
    print("[EVAL]", {"accuracy_exact": acc, "precision": float(P), "recall": float(R), "f1": float(F1)})



    # validation with metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    model.eval(); preds=[]; trues=[]
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            preds.append(torch.argmax(logits,1).cpu().numpy()); trues.append(yb.cpu().numpy())
    import json
    yhat = np.concatenate(preds); ytrue = np.concatenate(trues)
    acc = float(accuracy_score(ytrue, yhat))
    P,R,F1,_ = precision_recall_fscore_support(ytrue, yhat, average="macro", zero_division=0)
    json.dump(dict(accuracy_exact=acc, precision=float(P), recall=float(R), f1=float(F1)),
              open("micro_metrics.json","w"), indent=2)
    
    # save checkpoint + class names if present
    torch.save(dict(state_dict=model.state_dict(),
                    class_names=(D["name"] if args.data else None)),
                   "micro_llm_8gb_cpu.pth")