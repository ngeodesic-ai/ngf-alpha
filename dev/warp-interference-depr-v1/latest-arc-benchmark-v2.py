
# latest-arc-benchmark-v2.py
# Stage-10 (upgraded v2): PCA -> LDA space, multi-layer GPT-2 latent, prototype anchors,
# Mahalanobis (shared-cov) classifier, trajectory voting, CPU defaults.

import argparse, random, numpy as np
from numpy.linalg import inv
from dataclasses import dataclass

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from flip_fix_v3 import FlipAwareClassifier

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# -------------------------
# Defaults
# -------------------------
DEFAULTS = dict(
    target_var=0.99,
    pca_min_dims=8,
    pca_max_dims=24,
    lam=0.32,
    dt=0.02,
    steps=700,
    gamma=0.05,
    seed=42,
    mode="geodesic",
    tau=1.8,
    mass_scale=4.0,
    early_stop_window=15,
    vote_tail=20,   # majority vote over last N steps
)

EPS = 1e-6

def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def pick_device():
    return torch.device("cpu")

# -------------------------
# Latent extractor (multi-layer)
# -------------------------
class LatentExtractor:
    def __init__(self, device):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.model.eval()

    @torch.no_grad()
    def get(self, prompt:str, mode="mean_last4") -> np.ndarray:
        toks = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model(**toks, output_hidden_states=True)
        h = [hs[0].cpu().numpy() for hs in out.hidden_states[-4:]]  # last 4 layers
        if mode == "mean_last4":
            # mean over layers, then mean over sequence
            H = np.mean(h, axis=0)  # (seq, hidden)
            return H.mean(axis=0)
        elif mode == "eot_last4":
            H = np.mean(h, axis=0)
            return H[-1]  # last token
        else:
            H = out.hidden_states[-1][0].cpu().numpy()
            return H.mean(axis=0)

# -------------------------
# PCA -> LDA transform
# -------------------------
def pca_fit_transform(vectors, target_var=0.99, MIN_DIMS=8, MAX_DIMS=24, verbose=True):
    X = np.asarray(vectors)
    n_samples, n_features = X.shape
    n_cap = min(n_samples, n_features)
    probe_n = min(n_cap, MAX_DIMS)
    probe = PCA(n_components=probe_n, whiten=False, svd_solver="full").fit(X)
    cum = np.cumsum(probe.explained_variance_ratio_)
    k = int(np.searchsorted(cum, target_var) + 1)
    n = max(MIN_DIMS, min(MAX_DIMS, k))
    n = min(n, n_cap)
    whiten = (n >= 2)
    pca = PCA(n_components=n, whiten=whiten, svd_solver="full").fit(X)
    Z = pca.transform(X)
    if verbose:
        print(f"[PCA] samples={n_samples} feat={n_features} cap={n_cap} -> n={pca.n_components_} "
              f"(cum var={cum[min(n-1, len(cum)-1)]:.4f}, whiten={whiten})")
    return pca, Z

def lda_fit_transform(Z, labels, verbose=True):
    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels])
    # LDA components <= n_classes-1
    lda = LDA(n_components=min(len(classes)-1, Z.shape[1]))
    Z2 = lda.fit_transform(Z, y)
    if verbose:
        print(f"[LDA] projected to {Z2.shape[1]}D (classes={classes})")
    return lda, Z2, classes

# -------------------------
# Geometry
# -------------------------
def potential(x, centers, masses):
    diffs = x[None,:]-centers
    d = np.linalg.norm(diffs, axis=1) + EPS
    return -np.sum(masses / d)

def grad_potential(x, centers, masses):
    diffs = x[None,:]-centers
    d = np.linalg.norm(diffs, axis=1) + EPS
    terms = masses[:,None]*diffs/(d**3)[:,None]
    return -np.sum(terms, axis=0)

def lnphi_and_grad(x, centers, masses, lam):
    V = potential(x, centers, masses); gV = grad_potential(x, centers, masses)
    return lam*V, lam*gV

def rk4_step(x, v, dt, centers, masses, lam, gamma):
    _, grad_lnphi = lnphi_and_grad(x, centers, masses, lam)
    def acc(x_, v_):
        _, g = lnphi_and_grad(x_, centers, masses, lam)
        v_dot = float(np.dot(v_, g)); v_norm2 = float(np.dot(v_, v_))
        Gamma_vv = 2.0*v_*v_dot - v_norm2*g
        return -Gamma_vv - gamma*v_
    k1x = v; k1v = acc(x, v)
    k2x = v + 0.5*dt*k1v; k2v = acc(x + 0.5*dt*k1x, v + 0.5*dt*k1v)
    k3x = v + 0.5*dt*k2v; k3v = acc(x + 0.5*dt*k2x, v + 0.5*dt*k2v)
    k4x = v + dt*k3v;     k4v = acc(x + dt*k3x, v + dt*k3v)
    x_new = x + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    v_new = v + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
    return x_new, v_new

def integrate_with_voting(x0, v0, centers, masses, lam, gamma, dt, steps, vote_tail=20, window=15):
    x, v = x0.copy(), v0.copy()
    picks = []
    tail = []
    for t in range(steps):
        x, v = rk4_step(x, v, dt, centers, masses, lam, gamma)
        d = np.linalg.norm(centers - x, axis=1)
        pick = int(np.argmin(d))
        picks.append(pick)
        tail.append(pick)
        if len(tail) > vote_tail: tail.pop(0)
        if len(picks) > window and len(set(picks[-window:])) == 1:
            break
    # majority vote over last tail
    if tail:
        vals, counts = np.unique(tail, return_counts=True)
        voted = int(vals[np.argmax(counts)])
    else:
        voted = picks[-1]
    return x, voted

def integrate_nudge(x0, centers, masses, steps=350, dt=0.05, k=2.0, gamma=0.2, tau=0.8):
    x = x0.copy(); v = np.zeros_like(x)
    for _ in range(steps):
        d = np.linalg.norm(centers - x, axis=1)
        score = np.exp(-(d - d.min()) / max(1e-9, tau))
        w = (score * (masses + 1e-9)); w = w / (w.sum() + 1e-9)
        target = (w[:,None]*centers).sum(axis=0)
        a = k*(target - x) - gamma*v
        v = v + dt*a; x = x + dt*v
    return x

# -------------------------
# Anchors & classifier
# -------------------------
def prototype_prompts():
    # short canonical prototypes to build cleaner anchors (reduce lexical noise)
    return {
        "rotate":  ["Rotate 90° clockwise the grid."],
        "flip_h":  ["Flip the grid horizontally (mirror left-right)."],
        "flip_v":  ["Flip the grid vertically (mirror top-bottom)."],
    }

def compute_prototype_anchors(extractor, pca, lda, labels_in_order):
    protos = prototype_prompts()
    centers = []
    for lab in labels_in_order:
        plist = protos.get(lab, [f"{lab} transform."])
        lat = np.stack([extractor.get(p) for p in plist], axis=0)
        z = pca.transform(lat); z2 = lda.transform(z)
        c = z2.mean(axis=0)
        centers.append(c)
    centers = np.stack(centers, axis=0)
    # masses equal initially (will rescale later)
    masses = np.ones(len(labels_in_order), dtype=float)
    masses /= masses.sum()
    return centers, masses

def shared_cov_stats(Z2, labels, order, ridge=1e-3, shrink=0.2):
    mus, resids = {}, []
    for u in order:
        idx = [i for i,l in enumerate(labels) if l==u]
        G = Z2[idx]; mu = G.mean(axis=0)
        mus[u] = mu; resids.append(G - mu)
    R = np.concatenate(resids, axis=0)
    S = (R.T @ R) / max(1, R.shape[0]-1)
    D = np.diag(np.diag(S))
    S = (1-shrink)*S + shrink*D
    S += ridge*np.eye(S.shape[0])
    invS = inv(S)
    return {u: {"mu": mus[u], "invS": invS} for u in order}

def mahalanobis_shared(z, stats, order):
    invS = next(iter(stats.values()))["invS"]
    ds = []
    for u in order:
        mu = stats[u]["mu"]; d = z - mu
        ds.append(float(np.sqrt(d @ invS @ d)))
    return np.array(ds)

def anchor_confidence(distances, tau=1.8):
    d = np.asarray(distances); s = np.exp(-(d - d.min())/max(1e-9,tau))
    p = s/(s.sum()+1e-9); j = int(np.argmin(d))
    return j, float(p[j]), p

# -------------------------
# ARC pieces
# -------------------------
def rot90_cw(grid):
    g = np.array(grid); return np.rot90(g, k=3).tolist()
def flip_h(grid):
    g = np.array(grid); return np.flip(g, axis=1).tolist()
def flip_v(grid):
    g = np.array(grid); return np.flip(g, axis=0).tolist()

TRANSFORM_BY_LABEL = {"rotate": rot90_cw, "flip_h": flip_h, "flip_v": flip_v}

def grid_to_prompt(grid, choices):
    # Do NOT embed the choices words; keep label tokens out to reduce lexical bias
    return f"Identify the grid transformation pattern. Input grid {grid}. What is the output grid?"

def training_prompts():
    # 10 per class, non-symmetric
    rotate = [
        "Rotate 90° cw: [[1,2],[3,4]] → [[3,1],[4,2]].",
        "Rotate 90° cw: [[5,6],[7,8]] → [[7,5],[8,6]].",
        "Rotate 90° cw: [[2,1],[4,3]] → [[4,2],[3,1]].",
        "Rotate 90° cw: [[0,1],[2,3]] → [[2,0],[3,1]].",
        "Rotate 90° cw: [[9,8],[7,6]] → [[7,9],[6,8]].",
        "Rotate 90° cw (3×3): [[1,2,3],[4,5,6],[7,8,9]] → [[7,4,1],[8,5,2],[9,6,3]].",
        "Rotate 90° cw: [[3,0],[0,1]] → [[0,3],[1,0]].",
        "Rotate 90° cw: [[1,2,0],[0,3,4],[0,0,5]] → [[0,0,1],[0,3,2],[5,4,0]].",
        "Rotate 90° cw: [[2,0],[5,7]] → [[5,2],[7,0]].",
        "Rotate 90° cw: [[2,0,1],[0,1,0],[1,0,2]] → [[1,0,2],[0,1,0],[2,0,1]].",
    ]
    flip_h = [
        "Flip horizontally: [[1,2],[3,4]] → [[2,1],[4,3]].",
        "Flip horizontally: [[5,6],[7,8]] → [[6,5],[8,7]].",
        "Flip horizontally: [[0,1],[2,3]] → [[1,0],[3,2]].",
        "Flip horizontally: [[1,2,4],[4,5,6],[7,8,9]] → [[4,2,1],[6,5,4],[9,8,7]].",
        "Flip horizontally: [[2,1],[4,3]] → [[1,2],[3,4]].",
        "Flip horizontally: [[0,2],[5,7]] → [[2,0],[7,5]].",
        "Flip horizontally: [[3,0],[0,1]] → [[0,3],[1,0]].",
        "Flip horizontally: [[1,0],[0,2]] → [[0,1],[2,0]].",
        "Flip horizontally: [[1,0,2],[0,1,0],[2,0,1]] → [[2,0,1],[0,1,0],[1,0,2]].",
        "Flip horizontally: [[9,8],[7,6]] → [[8,9],[6,7]].",
    ]
    flip_v = [
        "Flip vertically: [[1,2],[3,4]] → [[3,4],[1,2]].",
        "Flip vertically: [[5,6],[7,8]] → [[7,8],[5,6]].",
        "Flip vertically: [[0,1],[2,3]] → [[2,3],[0,1]].",
        "Flip vertically: [[1,2,4],[4,5,6],[7,8,9]] → [[7,8,9],[4,5,6],[4,2,1]].",
        "Flip vertically: [[2,1],[4,3]] → [[4,3],[2,1]].",
        "Flip vertically: [[0,2],[5,7]] → [[5,7],[0,2]].",
        "Flip vertically: [[3,0],[0,1]] → [[0,1],[3,0]].",
        "Flip vertically: [[1,0],[0,2]] → [[0,2],[1,0]].",
        "Flip vertically: [[1,0,2],[0,1,0],[2,0,1]] → [[2,0,1],[0,1,0],[1,0,2]].",
        "Flip vertically: [[9,8],[7,6]] → [[7,6],[9,8]].",
    ]
    prompts = rotate + flip_h + flip_v
    labels = (["rotate"]*10 + ["flip_h"]*10 + ["flip_v"]*10)
    return prompts, labels

def make_cases():
    cases = [
        {"grid": [[1,2],[3,4]], "label": "rotate"},
        {"grid": [[5,6],[7,8]], "label": "rotate"},
        {"grid": [[1,0,2],[0,1,0],[2,0,1]], "label": "rotate"},
        {"grid": [[2,0,1],[0,1,0],[1,0,2]], "label": "rotate"},
        {"grid": [[1,2],[3,4]], "label": "flip_h"},
        {"grid": [[5,6],[7,8]], "label": "flip_h"},
        {"grid": [[1,0,2],[0,1,0],[2,0,1]], "label": "flip_h"},
        {"grid": [[1,2,4],[4,5,6],[7,8,9]], "label": "flip_h"},
        {"grid": [[1,2],[3,4]], "label": "flip_v"},
        {"grid": [[5,6],[7,8]], "label": "flip_v"},
        {"grid": [[1,0,2],[0,1,0],[2,0,1]], "label": "flip_v"},
        {"grid": [[1,2,4],[4,5,6],[7,8,9]], "label": "flip_v"},
    ]
    out = []
    for c in cases:
        t = TRANSFORM_BY_LABEL[c["label"]]; out.append({**c, "target": t(c["grid"])})
    return out

# -------------------------
# Context and pipeline
# -------------------------
@dataclass
class ARCContext:
    pca: PCA
    lda: LDA
    centers: np.ndarray
    masses: np.ndarray
    labels_in_order: list
    cls_stats: dict
    extractor: LatentExtractor
    args: argparse.Namespace

def anchor_conf(distances, tau):
    return anchor_confidence(distances, tau)

def arc_latent_for_grid(grid, ctx: ARCContext):
    prompt = grid_to_prompt(grid, ctx.labels_in_order)
    z_full = ctx.extractor.get(prompt, mode="mean_last4")
    z = ctx.pca.transform(z_full.reshape(1,-1))
    z2 = ctx.lda.transform(z)[0]
    return z2

def run_warp_interference(z2, ctx: ARCContext):
    _, g_lnphi = lnphi_and_grad(z2, ctx.centers, ctx.masses, ctx.args.lam)
    g_norm = float(np.linalg.norm(g_lnphi))
    if g_norm < 1e-8:
        rng = np.random.default_rng(ctx.args.seed)
        v0 = rng.normal(size=z2.shape).astype(float)
        v0 /= (np.linalg.norm(v0)+1e-9); v0 *= 0.05
    else:
        v0 = -g_lnphi/(g_norm+1e-9)*0.1

    if ctx.args.mode == "geodesic":
        zf, voted = integrate_with_voting(
            z2, v0, ctx.centers, ctx.masses, ctx.args.lam, ctx.args.gamma,
            ctx.args.dt, ctx.args.steps, vote_tail=ctx.args.vote_tail, window=ctx.args.early_stop_window
        )
        return zf, voted
    else:
        zf = integrate_nudge(z2, ctx.centers, ctx.masses)
        d = np.linalg.norm(ctx.centers - zf, axis=1)
        return zf, int(np.argmin(d))

def classify_maha(z2, ctx: ARCContext):
    d = mahalanobis_shared(z2, ctx.cls_stats, ctx.labels_in_order)
    j = int(np.argmin(d))
    return j, float(d[j]), d

# -------------------------
# Main benchmark
# -------------------------
def run_arc_benchmark_12(ctx: ARCContext):
    cases = make_cases()
    correct=0; confs=[]; details=[]
    for idx, case in enumerate(cases, 1):
        z0 = arc_latent_for_grid(case["grid"], ctx)
        zf, voted = run_warp_interference(z0, ctx)
        # Use Mahalanobis for probs/conf, but consider voted as tiebreaker
        j, dmin, dists = classify_maha(zf, ctx)
        j_hat, conf, probs = anchor_conf(dists, ctx.args.tau)
        # combine: if voted != j_hat but conf < 0.6, trust vote
        final_idx = voted if (j_hat!=voted and conf < 0.6) else j_hat
        pred_label = ctx.labels_in_order[final_idx]
        pred_grid = TRANSFORM_BY_LABEL[pred_label](case["grid"])
        ok = (pred_label == case["label"]) and (pred_grid == case["target"])
        correct += int(ok); confs.append(conf)
        print(f"[{idx:02d}] true={case['label']:7s} pred={pred_label:7s} ok={ok} "
              f"vote={ctx.labels_in_order[voted]} d*={dmin:.3f} conf={conf:.3f} "
              f"probs={np.array_str(probs, precision=3)}")
        details.append(dict(idx=idx, true=case["label"], pred=pred_label, ok=ok, conf=conf, dmin=dmin))
    acc = correct/len(cases); mean_conf = float(np.mean(confs)) if confs else 0.0
    print(f"\n[ARC-12] Accuracy: {acc*100:.1f}% | Mean confidence: {mean_conf:.3f} | Mode={ctx.args.mode}")
    # Sanity distances from a demo latent (no traversal)
    demo = [[1,2],[3,4]]
    z_demo = arc_latent_for_grid(demo, ctx)
    base_d = np.linalg.norm(ctx.centers - z_demo, axis=1)
    print("[Sanity] anchor order:", ctx.labels_in_order)
    print("[Sanity] distances:", np.array_str(base_d, precision=3))
    print("[Sanity] chosen:", ctx.labels_in_order[int(np.argmin(base_d))])
    return dict(accuracy=acc, mean_confidence=mean_conf, details=details)

# -------------------------
# CLI
# -------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Stage-10 Upgraded ARC Benchmark v2 (PCA→LDA, prototypes, voting)")
    for k,v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    return p.parse_args()

def main():
    args = parse_args()
    seed_all(args.seed)
    device = pick_device()
    print(f"[Init] mode={args.mode} seed={args.seed} device={device.type}")
    print(f"[Geom] λ={args.lam} γ={args.gamma} dt={args.dt} steps={args.steps} mass_scale={args.mass_scale}")

    # Build training latents
    extractor = LatentExtractor(device)
    prompts, labels = training_prompts()
    latents = np.stack([extractor.get(p, mode="mean_last4") for p in prompts], axis=0)

    # PCA then LDA
    pca, Z = pca_fit_transform(latents, target_var=args.target_var,
                               MIN_DIMS=args.pca_min_dims, MAX_DIMS=args.pca_max_dims, verbose=True)
    lda, Z2, labels_in_order = lda_fit_transform(Z, labels, verbose=True)



    # Classifier stats in LDA space
    cls_stats = shared_cov_stats(Z2, labels, labels_in_order, ridge=1e-3, shrink=0.2)

    # Prototype anchors in LDA space
    centers, masses = compute_prototype_anchors(extractor, pca, lda, labels_in_order)
    # Scale masses by class tightness from training to reflect separation
    spreads = []
    for i, lab in enumerate(labels_in_order):
        idx = [k for k,l in enumerate(labels) if l==lab]
        G = Z2[idx]; c = G.mean(axis=0)
        spreads.append(float(np.mean(np.linalg.norm(G - c, axis=1)) + 1e-6))
    masses = 1.0/np.array(spreads); masses = masses/(masses.sum()+1e-9)
    masses *= args.mass_scale

    print("[Anchors] order:", labels_in_order)

    ctx = ARCContext(pca=pca, lda=lda, centers=centers, masses=masses,
                     labels_in_order=labels_in_order, cls_stats=cls_stats,
                     extractor=extractor, args=args)

    run_arc_benchmark_12(ctx)

if __name__ == "__main__":
    main()
