# Noetic Geodesic Framework (Alpha)
🚀 **The Noetic Geodesic Framework (NGF)** is a pioneering alpha approach to enhance AI reasoning, tackling hallucinations via geodesics. Much of modern AI follows the mantra “we know it works, but don’t know why.” NGF shifts this by applying physics tools—geometry, geodesics, and symmetry—to uncover hidden structures in models, treating them as systems with discoverable laws, with promising results on synthetic tasks.

## What’s This?
NGF enhances GPT-2 by adjusting its latent space using PCA and a symbolic nudge, which is linear approximation to geodesics (see [technical draft](https://github.com/ngeodesic-ai/ngf-alpha/blob/main/docs/article_latest.pdf)), reducing hallucinations on synthetic ARC patterns and MMLU tasks in alpha testing. It’s an alpha release—early, exciting, and open for collaboration!

## Research Plan Stages
The NGF follows a 12-step research plan, with 10 completed stages posted here. Step 10 is the public rollout with NVIDIA A100 results. Steps 11-12 (milestone reports) are in progress.

| Stage | Description | Phase | Hardware | Folder/Code |
|-------|-------------|-------------|-------------|-------------|
| 1 | [Toy Example](toy-example/step1.ipynb) | Toy Example $R^4$ | CPU | toy-example/ |
| 2 | [Embed Grid Intelligently](embed-grid/step2.ipynb) | Toy Example $R^4$ | CPU | embed-grid/ |
| 3 | [Rotation Matrix Integration](rotation-matrix/step3.ipynb) | Toy Example $R^4$ | CPU | rotation-matrix/ |
| 4 | [Simulate Pattern Completion](pattern-completion/step4.ipynb) | Toy Example $R^4$ | CPU | pattern-completion/ |
| 5 | [Higher-Dim Embeddings](higher-dim-embeddings/step5.ipynb) | Higher Dim $R^9$ | CPU | higher-dim-embeddings/ |
| 6 | [Integrate Dynamic Intelligence](dynamic-intelligence/step6.ipynb) | Higher Dim $R^9$ | CPU | dynamic-intelligence/ |
| 7 | [ARC Question](rudimentary-arc/step7.ipynb) | Higher Dim $R^9$ | CPU | rudimentary-arc/ |
| 8 | [LLM Latent Embedding with DR](llm-latent-embedding/step8.ipynb) | LLM System | CPU | llm-latent-embedding/ |
| 9 | [Warp LLM Interference](warp-interference/step9.py) | LLM System | A100 | warp-interference/ |
| 10 | [ARC](latest-arc-benchmark.py) / [MMLU](latest-mmlu-benchmark.py) Benchmarks | LLM System | A100 | small-benchmarks/ |
| 11 | Large Benchmark (coming) | LLM System | A100 | large-benchmarks/ |
| 12 | Milestone Benchmark (coming) | LLM System | A100 | milestone-benchmark/ |


## Illustration: NGF Warped vs Flat Paths (Re: Stage 5)

This animation shows how warped paths converge to correct answers in high-dimensional semantic space:

![NGF Warped vs Flat Paths](higher-dim-embeddings/ngf_warped_geodesic_contour.gif)

## Requirements
- Python 3.x
- `transformers==4.30.0`
- `torch==2.4.1`
- `numpy==1.26.4`
- `scikit-learn==1.0.0`
- NVIDIA A100 GPU (e.g., Colab Pro+)

## Setup
Install dependencies:
```bash
pip install transformers==4.30.0 torch==2.4.1 numpy==1.26.4 scikit-learn==1.0.0
```

#### Run latest ARC benchmark:
```bash
python latest-arc-benchmark.py
```

#### Benchmark results - ARC test
```
Stock Accuracy: 65.0%
Warped Accuracy: 100.0%
Warped Semantic Similarity: 93.7%
Hallucination Rate: 0.0%
```

#### Run latest MMLU benchmark:
```bash
python latest-mmlu-benchmark.py
```

#### Benchmark results - MMLU test
```
Stock Accuracy: 60.0%
Warped Accuracy: 100.0%
Warped Semantic Similarity: 92.1%   
Hallucination Rate: 0.0%
```

#### Self-Evaluation (latest)
- Tests are fully blind; the separate seed (43) and filtering ensured no overlap with the training set
- Possible tuning bias that needs to be investigated
- Small synthetic test set; next step is to transition to larger sample of ARC/MMLU tasks for validation (eg, from datasets python package)
- Using nudge approach to approximate geodesic path; this is good for now on synthetic tasks, as per [simulations](small-benchmarks/benchmark-findings.ipynb); a hybrid approach using ODEs could address bias/variance issues
- 350 nudge steps per correction seems excessive
- Tested on NVIDIA A100, need to test on NVIDIA T4
- For ARC, the stock 65.0% and semantic similarity 93.7% are realistic
- For MMLU, the stock 60.0% and semantic similarity 92.1% are realistic

## Hardware Notes
Results validated on an NVIDIA A100 GPU (Colab Pro+ & Grok Expert). Testing on NVIDIA T4 and CPU currently inprogress. Performance may vary; A100 recommended for optimal results.

## Technical Paper - WORK IN PROGRESS
A draft paper is included as [Noetic Geodesic Framework: A Geometric Approach to Deterministic AI Reasoning](docs/article_latest.pdf). **Disclaimer**: This is a preliminary alpha-stage document (August 20, 2025), subject to change. Feedback is welcome! Provisional patents filed as #63/864,726 and #63/865,437.

## References
- Moore, I. C. (2025). *Warped Semantic Manifolds: A Geometric Framework for Deterministic AI Reasoning (Preliminary Memo)*. Zenodo. https://zenodo.org/records/16908227 (DOI: 10.5281/zenodo.16730759); see [code](toy-example/step1.ipynb)

## Medium Articles
 * **Toy Example in $R^3$**: [Warped Semantic Manifolds: A Geometric Approach to AI Reasoning](https://medium.com/@icmoore/warped-semantic-manifolds-a-new-path-to-flawless-ai-reasoning-d2328c91d920)
 * **Higher Dimensional Embeddings in $R^9$**: [How Semantic Mass Shapes AI Reasoning in R^9](https://medium.com/@icmoore/how-semantic-mass-warps-ai-thoughts-to-flawless-convergence-879e2f6f3373) 


## Onboarding
As these techniques are uncommon in AI, onboarding requires some prerequisites background knowledge in physics and differential geometry, coupled with a good understanding of the objectives behind each of the steps in the [12-staged research plan](#research-plan-stages). If this interests you, please see the **[onboarding docs](docs/onboarding.md)** for further detail.

## Contribute
This is alpha software! Help us refine prompts, test on other hardware, or improve the nudge. **Contributors must sign the [CLA](CLA.md) and email it to ngeodesic@gmail.com before submitting pull requests.**

If you find this helpful, please leave a ⭐!
