import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import argparse

# Mock synthetic traces (replace with real import from stage11_benchmark_latest.py)
def make_synthetic_traces(samples=5000, latent_dim=128, noise=0.05):
    np.random.seed(42)  # Reproducible
    latents = np.random.randn(samples, latent_dim) + noise * np.random.randn(samples, latent_dim)
    labels = np.random.randint(0, 3, samples)  # 0: flip_h, 1: flip_v, 2: rotate
    return latents, labels

# Dataset for CPU efficiency
class LatentARCDataset(Dataset):
    def __init__(self, latents, labels):
        self.latents = torch.tensor(latents, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.latents)
    def __getitem__(self, idx): return self.latents[idx], self.labels[idx]

# Micro-LLM: Optimized for CPU
class MicroLLM(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, num_heads=2, num_classes=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.margin_loss = nn.MarginRankingLoss(margin=0.044)  # NGF-inspired (Appendix A Eq. 2)

    def forward(self, x):
        x = self.embedding(x.unsqueeze(1))  # [B, 1, H]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pool
        return self.fc(x)

# Training loop for CPU
def train(model, dataloader, epochs=1, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for latents, labels in dataloader:
            outputs = model(latents)
            loss = criterion(outputs, labels)
            logits_sorted = outputs.sort(dim=1, descending=True).values
            margin_target = torch.ones_like(logits_sorted[:, 0])
            margin_loss = model.margin_loss(logits_sorted[:, 0], logits_sorted[:, 1], margin_target)
            total_loss = loss + 0.5 * margin_loss
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # CPU stability
            total_loss.backward()
            optimizer.step()
            total_loss += total_loss.item()
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(dataloader):.4f}")
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

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    # Generate and prepare data
    latents, labels = make_synthetic_traces(args.samples, args.latent_dim)
    dataset = LatentARCDataset(latents, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)  # CPU-safe

    # Train micro-LLM
    model = MicroLLM(input_dim=args.latent_dim, num_classes=3)
    model = train(model, dataloader, epochs=args.epochs)

    # Validate on held-out set
    val_latents, val_labels = make_synthetic_traces(500, args.latent_dim)
    val_dataset = LatentARCDataset(val_latents, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=8)
    accuracy = validate(model, val_loader)
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # Save
    torch.save(model.state_dict(), "micro_llm_cpu.pth")