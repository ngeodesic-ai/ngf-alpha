# Noetic Geodesic Framework (Alpha)
🚀 **This is the Noetic Geodesic Framework (NGF)**, a pioneering approach to stabilize AI reasoning to address the hallucination problem via geodisics.

## What’s This?
NGF enhances GPT-2 by adjusting its latent space using PCA and a symbolic nudge, which is linear approximation to geodesics (see [technical draft](https://github.com/ngeodesic-ai/ngf-alpha/blob/main/docs/article_latest.pdf)), reducing hallucinations on synthetic ARC patterns and MMLU tasks in alpha testing. It’s an alpha release—early, exciting, and open for collaboration!

## Research Plan Stages
The NGF follows a 12-step research plan, with 10 completed stages posted here. Step 10 is the public rollout with NVIDIA A100 results. Steps 11-12 (milestone reports) are in progress.

| Stage | Description | Hardware | Folder/Code |
|-------|-------------|-------------|-------------|
| 1 | [Toy Example](toy-example/step1.ipynb) | CPU | toy-example/ |
| 2 | [Embed Grid Intelligently](embed-grid/step2.ipynb) | CPU | embed-grid/ |
| 3 | [Rotation Matrix Integration](rotation-matrix/step3.ipynb) | CPU | rotation-matrix/ |
| 4 | [Simulate Pattern Completion](pattern-completion/step4.ipynb) | CPU | pattern-completion/ |
| 5 | [Higher-Dim Embeddings](higher-dim-embeddings/step5.ipynb) | CPU | higher-dim-embeddings/ |
| 6 | [Integrate Dynamic Intelligence](dynamic-intelligence/step6.ipynb) | CPU | dynamic-intelligence/ |
| 7 | [Rudimentary ARC](rudimentary-arc/step7.ipynb) | CPU | rudimentary-arc/ |
| 8 | [LLM Latent Embedding with DR](llm-latent-embedding/step8.ipynb) | CPU | llm-latent-embedding/ |
| 9 | [Warp LLM Interference](warp-interference/step9.py) | A100 | warp-interference/ |
| 10 | [ARC](latest-arc-benchmark.py) / [MMLU](latest-mmlu-benchmark.py) Benchmarks | A100 | small-benchmarks/ (small benchmark) |
| 11 | Large Benchmark (coming) | A100 | large-benchmarks/ (large benchmark) |
| 12 | Milestone Benchmark (coming) | A100 | milestone-benchmark/ (milestone) |


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
- 350 nudge steps per correction seems excessive
- Small test set; next step is to transition to larger sample of ARC/MMLU tasks for validation (eg, from datasets python package)
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


## Contribute
This is alpha software! Help us refine prompts, test on other hardware, or improve the nudge. **Contributors must sign the [CLA](CLA.md) and email it to ngeodesic@gmail.com before submitting pull requests.**

If you find this helpful, please leave a ⭐!
