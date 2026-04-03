# wellmetric_ft.py
import math, argparse, contextlib
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from datasets import load_dataset

"""

python3 wellmetric_ft_v2.py --model_name gpt2 --layer_idx 9 --max_steps 1200

# HellaSwag
python3 ngf_benchmark.py --dataset hellaswag --mode stock \
  --model gpt2 --split validation --n 1000 --max_length 128 --device auto \
  --out_json results/hella_stock.json

python3 ngf_benchmark.py --dataset hellaswag --mode stock \
  --model gpt2 --split validation --n 1000 --max_length 128 --device auto \
  --wellmetric_ckpt wellmetric_gpt2_layer9.pt \
  --out_json results/hella_warp.json

# HellaSwag
python3 ngf_benchmark.py --dataset hellaswag --mode stock \
  --model gpt2 --split validation --n 1000 --max_length 128 --device auto \
  --out_json results/hella_stock.json

python3 ngf_benchmark.py --dataset hellaswag --mode stock \
  --model gpt2 --split validation --n 1000 --max_length 128 --device auto \
  --wellmetric_ckpt wellmetric_gpt2_layer9.pt \
  --out_json results/hella_warp.json

# Boolq (same pattern)
python3 ngf_benchmark.py --dataset boolq --mode stock \
  --model gpt2 --split validation --n 1200 --max_length 512 --device auto \
  --out_json results/wg_stock.json

python3 ngf_benchmark.py --dataset boolq --mode stock \
  --model gpt2 --split validation --n 1200 --max_length 512 --device auto \
  --wellmetric_ckpt wellmetric_gpt2_layer9.pt \
  --out_json results/wg_warp.json

# commonsenseqa (same pattern)
python3 ngf_benchmark.py --dataset commonsenseqa --mode stock \
  --model gpt2 --split validation --n 1200 --max_length 512 --device auto \
  --out_json results/wg_stock.json

python3 ngf_benchmark.py --dataset commonsenseqa --mode stock \
  --model gpt2 --split validation --n 1200 --max_length 512 --device auto \
  --wellmetric_ckpt wellmetric_gpt2_layer9.pt \
  --out_json results/wg_warp.json


"""

# ---------------------------
# 0) Angle-preserving warp
# ---------------------------
class WellMetric(nn.Module):
    def __init__(self, d_model: int, alpha=1.0, beta=0.5):
        super().__init__()
        self.center = nn.Parameter(torch.zeros(1, 1, d_model))
        self.alpha  = nn.Parameter(torch.tensor(float(alpha)))
        self.beta   = nn.Parameter(torch.tensor(float(beta)))
        self.eps    = 1e-8
        self.enabled = True
        self.register_buffer("_cm_stack", torch.tensor(0))  # placeholder

    @contextlib.contextmanager
    def disabled(self):
        prev = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = prev

    def forward(self, h):  # h: [B,T,D] or [T,D]
        if not self.enabled:
            return h

        # Make center shape match h (handle 2D vs 3D cleanly)
        if h.dim() == 3:             # [B,T,D]
            c = self.center.to(h.dtype).to(h.device)    # [1,1,D]
        elif h.dim() == 2:           # [T,D]
            c = self.center.to(h.dtype).to(h.device).squeeze(0)  # [1,D]
        else:
            raise ValueError(f"Unsupported h.dim()={h.dim()} (expected 2 or 3)")

        v = h - c
        r = torch.linalg.vector_norm(v, dim=-1, keepdim=True)              # [...,1]
        a0 = F.softplus(self.alpha).to(h.dtype)                             # max warp
        b  = F.softplus(self.beta).to(h.dtype)                              # slope
        # radius-conditioned alpha: small near the apex, capped globally
        alpha_r = torch.clamp(a0 * (1.0 - torch.exp(-b * r)), 0.0, 0.25)   # ≤ 0.25
        # apply toward center (angle-preserving)
        z = c + (1.0 - alpha_r) * v

        # Final NaN guard (shouldn’t trigger, but belt & suspenders)
        z = torch.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)
        return z


# ---------------------------
# 1) Hook to apply WellMetric at a transformer block
# ---------------------------
def attach_wellmetric(model, layer_idx: int):
    d = model.config.n_embd
    wm = WellMetric(d)
    model.add_module("wellmetric", wm)

    cache = {"pre": None}

    def hook_apply(_module, _inp, out):
        # GPT-2 blocks often return a tuple: (hidden_states, present, ...)
        if isinstance(out, (tuple, list)):
            hs = out[0]
        else:
            hs = out

        cache["pre"] = hs.detach() if hs.requires_grad else hs
        hs_warp = model.wellmetric(hs) if model.wellmetric.enabled else hs

        # Return same structure with hs replaced
        if isinstance(out, tuple):
            return (hs_warp,) + tuple(out[1:])
        elif isinstance(out, list):
            return [hs_warp] + list(out[1:])
        else:
            return hs_warp

    handle = model.transformer.h[layer_idx].register_forward_hook(
        lambda m, i, o: hook_apply(m, i, o)
    )
    return wm, cache, handle



# ---------------------------
# 2) Post-hoc target: whiten → radial funnel → unwhiten
# ---------------------------
@torch.no_grad()
def fit_whitener(H: torch.Tensor, q: int = 256):
    """
    H: [N, D] sample of pre-warp activations (float32)
    Returns mu [1,D], W [D,D], Winv [D,D] (ZCA-like in feature space).
    """
    mu = H.mean(0, keepdim=True)
    X  = H - mu                               # [N,D]
    # pca_lowrank returns: U [N,q], S [q], V [D,q]
    U, S, V = torch.pca_lowrank(X, q=min(q, H.size(1)))
    eps = 1e-5
    Sinv = torch.diag(1.0 / (S + eps))        # [q,q]
    Vq   = V[:, :S.numel()]                   # [D,q]
    W    = Vq @ Sinv @ Vq.T                   # [D,D]  (whitener)
    Winv = Vq @ torch.diag(S) @ Vq.T          # [D,D]  (approx inverse)
    return mu, W, Winv


@torch.no_grad()
def posthoc_funnel(H: torch.Tensor, mu, W, Winv, alpha=1.0, beta=0.5):
    """
    Applies analytic Stage-11-style funnel to a batch of activations.
    H: [N,D] or [B,T,D]
    """
    orig_shape = H.shape
    Z = (H.reshape(-1, H.size(-1)) - mu) @ W
    r = Z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    Udir = Z / r
    g = alpha * torch.tanh(beta * r)
    Zp = Udir * g
    Hp = (Zp @ Winv) + mu
    return Hp.reshape(orig_shape)

# ---------------------------
# 3) KL(pre, post) on logits for behavior guard
# ---------------------------
def kl_pre_post(logits_pre, logits_post, T=2.0):
    p = F.log_softmax(logits_pre / T, dim=-1)
    q = F.log_softmax(logits_post / T, dim=-1)
    p_exp = p.exp()
    kl = (p_exp * (p - q)).sum(dim=-1)   # per token
    return kl.mean()

# ---------------------------
# 4) Training loop (Phase A: backbone frozen)
# ---------------------------
def phaseA_distill(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tok.pad_token = tok.eos_token

    # Small text dataset (feel free to swap)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:2%]")
    def enc(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)
    ds = ds.map(enc, batched=True, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=default_data_collator)

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    # Attach the WellMetric
    wm, cache, handle = attach_wellmetric(model, layer_idx=args.layer_idx)

    # Freeze backbone
    for p in model.parameters(): p.requires_grad_(False)
    for p in wm.parameters():    p.requires_grad_(True)

    opt = torch.optim.AdamW(wm.parameters(), lr=args.lr)

    # --- Warmup: fit whitener from a few batches of pre-warp activations
    samples = []
    with torch.no_grad(), wm.disabled():  # disable warp to sample raw layer outputs
        for i, batch in enumerate(dl):
            if i >= args.whiten_batches: break
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        output_hidden_states=False)
            H = cache["pre"]  # [B,T,D]
            samples.append(H.reshape(-1, H.size(-1)).float().cpu())
    Hs = torch.cat(samples, dim=0)
    mu, W, Winv = fit_whitener(Hs, q=min(128, Hs.size(1)))
    mu, W, Winv = mu.to(device), W.to(device), Winv.to(device)
    # initialize center from measured μ
    with torch.no_grad():
        wm.center.copy_(mu.view(1,1,-1))
    # (optional) freeze the center during Phase-A for stability
    wm.center.requires_grad_(False)

    # --- Phase A: distill shape + logit consistency
    for step, batch in enumerate(dl, start=1):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        # PRE (no warp): logits_pre and cached H_raw
        with wm.disabled():
            out_pre = model(input_ids=input_ids, attention_mask=attn)
            logits_pre = out_pre.logits  # [B,T,V]
            H_raw = cache["pre"].detach()  # [B,T,D]

        # Target = posthoc funnel on H_raw (analytic Stage-11)
        H_tgt = posthoc_funnel(H_raw, mu, W, Winv,
                               alpha=args.funnel_alpha, beta=args.funnel_beta).detach()

        # POST (warp on): logits_post
        out_post = model(input_ids=input_ids, attention_mask=attn)
        logits_post = out_post.logits  # [B,T,V]

        # Distill: make WellMetric(H_raw) ≈ H_tgt
        H_warp = wm(H_raw)  # apply module explicitly to the same H_raw
        loss_mse = F.mse_loss(H_warp, H_tgt)
        # direction alignment (cosine), guard tiny norms
        eps = 1e-6
        v_w = H_warp - wm.center; v_t = H_tgt - wm.center
        cos = F.cosine_similarity(v_w, v_t, dim=-1)
        loss_dir = (1.0 - cos).mean()



        # Guard: keep behavior similar
        # mask: only score positions that predict the next token
        shift_p = logits_pre[:, :-1]; shift_q = logits_post[:, :-1]
        mask    = attn[:, 1:].float()
        p = F.log_softmax(shift_p / args.kl_T, dim=-1).exp()
        q = F.log_softmax(shift_q / args.kl_T, dim=-1)
        kl_tok = (p * (p.log() - q)).sum(dim=-1) * mask
        loss_kl = kl_tok.sum() / (mask.sum() + 1e-6)
        # KL warmup (prevents early over-regularization)
        kl_w = min(1.0, step / max(50, args.log_every)) * args.kl_lambda
        loss = loss_mse + 0.1 * loss_dir + kl_w * loss_kl

        loss = loss_mse + 0.1 * loss_dir + args.kl_lambda * loss_kl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                r_raw = (H_raw - wm.center).norm(dim=-1).mean().item()
                r_warp = (H_warp - wm.center).norm(dim=-1).mean().item()
                print(f"[{step:05d}] loss={loss.item():.4f}  mse={loss_mse.item():.4f}  kl={loss_kl.item():.4f}  "
                      f"⟨r_raw⟩={r_raw:.3f} → ⟨r_warp⟩={r_warp:.3f}")

        if step >= args.max_steps:
            break

    handle.remove()
    # Save just the tiny module (and center)
    torch.save({"state_dict": wm.state_dict(),
                "layer_idx": args.layer_idx,
                "model_name": args.model_name,
                "mu": wm.center.detach().cpu().squeeze(0).squeeze(0),   # [D]
                "alpha": float(F.softplus(wm.alpha).item()),
                "beta":  float(F.softplus(wm.beta).item())},
                args.out_path)
    print(f"[✓] Saved WellMetric to {args.out_path}")

# ---------------------------
# 5) CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="gpt2")
    ap.add_argument("--layer_idx", type=int, default=9, help="inject after this transformer.h[idx]")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--whiten_batches", type=int, default=8, help="batches to fit whitener")
    ap.add_argument("--funnel_alpha", type=float, default=1.0)
    ap.add_argument("--funnel_beta", type=float, default=0.5)
    ap.add_argument("--kl_lambda", type=float, default=0.05)
    ap.add_argument("--kl_T", type=float, default=2.0)
    ap.add_argument("--out_path", type=str, default="wellmetric_gpt2_layer9.pt")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    phaseA_distill(args)
