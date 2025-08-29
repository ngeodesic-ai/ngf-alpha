# Noetic Geodesic Framework (Alpha)
🚀 **The Noetic Geodesic Framework (NGF)** is a geometric approach to deterministic AI reasoning. It reframes reasoning in latent space as geodesic traversals through warped manifolds, where semantic structure is enforced by energy wells. This allows us to suppress hallucinations and enforce stable, truth-aligned reasoning.

NGF builds on two key pillars:
* Latent Vector Embeddings — high-dimensional representations (used across modern AI, including LLMs).
* Warp → Detect → Denoise Doctrine (Stage 11) — our pipeline that shapes these embeddings into deterministic geodesic trajectories.

### LLMs vs Vector Embeddings
* LLMs (Large Language Models): Sequence models that operate on tokens, typically built on transformer architectures. They internally rely on vector embeddings (hidden states) but expose only the text interface.
* Vector Embeddings: High-dimensional vectors that encode semantic meaning. These can be obtained independently of an LLM (e.g., sentence embeddings, ARC synthetic embeddings) and are directly manipulable.

**NGF operates at the embedding level**, not at the text level. This means NGF methods are pluggable into any LLM or embedding model. Instead of manipulating prompts or fine-tuning weights, NGF directly reshapes latent trajectories in vector space.

## What’s This?
NGF enhances GPT-2 by adjusting its latent space using PCA and a symbolic nudge, which is linear approximation to geodesics (see [technical draft](https://github.com/ngeodesic-ai/ngf-alpha/blob/main/docs/article_latest.pdf)), reducing hallucinations on synthetic ARC patterns and MMLU tasks in alpha testing. It’s an alpha release—early, exciting, and open for collaboration!

## Research Plan Stages
The NGF follows a 12-step research plan, with 10 completed stages posted here. Step 10 is the public rollout with NVIDIA A100 results. Steps 11-12 (milestone reports) are in progress.

| Stage | Description | Phase | Hardware | Folder/Code |
|-------|-------------|-------------|-------------|-------------|
| 1 | [Toy Example](toy-example/stage1.ipynb) | Toy Example $R^4$ | CPU | toy-example/ |
| 2 | [Embed Grid Intelligently](embed-grid/stage2.ipynb) | Toy Example $R^4$ | CPU | embed-grid/ |
| 3 | [Rotation Matrix Integration](rotation-matrix/stage3.ipynb) | Toy Example $R^4$ | CPU | rotation-matrix/ |
| 4 | [Simulate Pattern Completion](pattern-completion/stage4.ipynb) | Toy Example $R^4$ | CPU | pattern-completion/ |
| 5 | [Higher-Dim Embeddings](higher-dim-embeddings/stage5.ipynb) | Higher Dim $R^9$ | CPU | higher-dim-embeddings/ |
| 6 | [Integrate Dynamic Intelligence](dynamic-intelligence/stage6.ipynb) | Higher Dim $R^9$ | CPU | dynamic-intelligence/ |
| 7 | [ARC Question](rudimentary-arc/stage7.ipynb) | Higher Dim $R^9$ | CPU | rudimentary-arc/ |
| 8 | [LLM Latent Embedding](llm-latent-embedding/stage8.ipynb) | LLM System | CPU | llm-latent-embedding/ |
| 9 | [Warp LLM Interference](warp-interference/stage9.py) | LLM System | CPU | warp-interference/ |
| 10 | [Rudimentary Benchmarks](rudimentary-benchmarks/stage10-benchmark-latest.py) | LLM System*  | CPU | rudimentary-interference/ |
| 11 | [Small Benchmarks](small-benchmarks/stage11-benchmark-latest.py) | LLM System*   | CPU | small-benchmarks/ |
| 12 | Large Benchmark (coming) | LLM System | A100 | milestone-benchmark/ |

(*) **Note**: working on vector embeddings only

## Illustration: NGF Warped vs Flat Paths (Re: Stage 5)

This animation shows how warped paths converge to correct answers in high-dimensional semantic space:

![NGF Warped vs Flat Paths](higher-dim-embeddings/ngf_warped_geodesic_contour.gif)

## Requirements
- Python 3.x
- `transformers==4.55.2`
- `torch==2.8.0`
- `numpy==2.0.2`
- `scikit-learn==1.6.1`
- NVIDIA A100 GPU (e.g., Colab Pro+)

## Setup
Install dependencies:
```bash
!pip install transformers==4.55.2 torch==2.8.0 numpy==2.0.2 scikit-learn==1.6.1
```

## Stage-11 (Current): Warp → Detect → Denoise
Stage-11 introduced the breakthrough:
* **Warp:** Embed latents into PCA(3) space, warp into a single dominant well.
* **Detect:** Use matched filters with null calibration to identify the true well.
* **Denoise:** Apply smoothing, phantom guards, and jitter averaging to suppress false wells.

#### Results
On Latent ARC (n=100):
* Stock Parser: 49% exact
* Geodesic Parser (Stage-10): 64% exact
* Stage-11 Denoiser: 100% exact; hallucination ≈ 0.5%, omission ≈ 0.2%
This is deterministic reasoning: geodesic paths converge to truth-aligned endpoints.

#### How This Relates to LLMs
* NGF is not a new LLM. It is a geometry-on-latents module.
* You can integrate NGF with any embedding-producing model (LLMs, encoders, diffusion models).
* **Example**: an LLM provides hidden states → NGF warps them → trajectories follow deterministic geodesics instead of drifting probabilistically.
This separation is critical: LLMs handle language; NGF handles geometry.

#### Run latest benchmark:
```bash
python -u arc-benchmark-latest.py \
      --samples 100 --seed 42 \
      --latent_arc --latent_dim 64 --latent_arc_noise 0.05 \
      --denoise_mode hybrid --ema_decay 0.85 --median_k 3 \
      --probe_k 5 --probe_eps 0.02 --conf_gate 0.65 --noise_floor 0.03 \
      --seed_jitter 2 --log INFO \
      --out_json latent_arc_denoise_100.json --out_csv latent_arc_denoise_100.csv
```

## Hardware Notes
Results validated on an NVIDIA A100 GPU (Colab Pro+ & Grok Expert). Testing on NVIDIA T4 and CPU currently inprogress. Performance may vary; A100 recommended for optimal results.

## Technical Paper - WORK IN PROGRESS
A draft paper is included as [Noetic Geodesic Framework: A Geometric Approach to Deterministic AI Reasoning](docs/article_latest.pdf). **Disclaimer**: This is a preliminary alpha-stage document (August 20, 2025), subject to change. Feedback is welcome! Provisional patents filed as #63/864,726 and #63/865,437.

## References
- Moore, I. C. (2025). *Warped Semantic Manifolds: A Geometric Framework for Deterministic AI Reasoning (Preliminary Memo)*. Zenodo. https://zenodo.org/records/16908227 (DOI: 10.5281/zenodo.16730759); see [code](toy-example/step1.ipynb)

## Medium Articles
 * **Toy Example in $R^3$**: [Warped Semantic Manifolds: A Geometric Approach to AI Reasoning](https://medium.com/@icmoore/warped-semantic-manifolds-a-new-path-to-flawless-ai-reasoning-d2328c91d920)
 * **Higher Dimensional Embeddings in $R^9$**: [How Semantic Mass Shapes AI Reasoning in R^9](https://medium.com/@icmoore/how-semantic-mass-warps-ai-thoughts-to-flawless-convergence-879e2f6f3373)
 * **Warping LLM Interference**: [Warping LLM Interference with Geodesic Nudges for Deterministic AI Reasoning](https://medium.com/@icmoore/warping-llm-interference-with-geodesic-nudges-for-deterministic-ai-reasoning-38663bbc1609)


## Onboarding
As these techniques are uncommon in AI, onboarding requires some prerequisites background knowledge in general relativity and differential geometry, coupled with a good understanding of the objectives behind each of the steps in the [12-staged research plan](#research-plan-stages). If this interests you, please see the **[onboarding docs](docs/onboarding.md)** for further detail.

## Contribute
This is alpha software! Help us refine prompts, test on other hardware, or improve the nudge. **Contributors must sign the [CLA](CLA.md) and email it to ngeodesic@gmail.com before submitting pull requests.**

If you find this helpful, please leave a ⭐!
