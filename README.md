# Noetic Geodesic Framework (Alpha)
🚀 **This is the Noetic Geodesic Framework (NGF)**, a pioneering approach to boost GPT-2's reasoning with symbolic nudging. Achieved 100.0% accuracy on 100 ARC and 100 MMLU tasks when nudged, with a 65.0% stock baseline on an A100 GPU.

## What’s This?
NGF enhances GPT-2 by adjusting its latent space using PCA and a symbolic nudge, tackling synthetic ARC patterns and MMLU challenges. It’s an alpha release—early, exciting, and open for collaboration!

## Research Plan Stages
The NGF follows a 10-step research plan, with 9 completed stages posted here. Step 9 is the public rollout with A100 results. Step 10 (milestone report) is in progress.

| Stage | Description | Folder/Code |
|-------|-------------|-------------|
| 1 | [Refine Toy Example](toy-example/step1.ipynb) | toy-example/ |
| 2 | [Embed Grid Intelligently](embed-grid/step2.ipynb) | embed-grid/ |
| 3 | [Rotation Matrix Integration](rotation-matrix/step3.ipynb) | rotation-matrix/ |
| 4 | [Simulate Pattern Completion](pattern-completion/step4.ipynb) | pattern-completion/ |
| 5 | [Higher-Dim Embeddings](higher-dim-embeddings/step5.ipynb) | higher-dim-embeddings/ |
| 6 | [Integrate Dynamic Intelligence](dynamic-intelligence/step6.ipynb) | dynamic-intelligence/ |
| 7 | [Rudimentary ARC](rudimentary-arc/step7.ipynb) | rudimentary-arc/ |
| 8 | [LLM Latent Embedding with DR](llm-latent-embedding/step8.ipynb) | llm-latent-embedding/ |
| 9 | [Warp LLM Interference](warp-interference/step9.ipynb) | warp-interference/ |
| 10 | [Public Validation](main/main.ipynb) | main/ (full benchmark) |


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
!pip install transformers==4.30.0 torch==2.4.1 numpy==1.26.4 scikit-learn==1.0.0
```

## Hardware Notes
Results validated on an NVIDIA A100 GPU (Colab Pro+). Testing on NVIDIA T4 (Colab Free) showed [insert results], and CPU runtime was [insert results]. Performance may vary; A100 recommended for optimal results.

## Technical Paper
A draft paper is included as [Noetic Geodesic Framework: A Geometric Approach to Deterministic AI Reasoning](docs/article_v8.pdf). **Disclaimer**: This is a preliminary alpha-stage document (August 15, 2025), subject to change. Feedback is welcome! Provisional patent filed as #63/864,726.

## Patent Status
A provisional patent was filed on August 15, 2025, with application number 63/864,726. Refinements will follow for a utility patent.

## Contribute
This is alpha software! Help us refine prompts, test on other hardware, or improve the nudge. **Contributors must sign the [CLA](CLA.md) and email it to ic3moore@gmail.com before submitting pull requests.**

If you find this helpful, please leave a ⭐!
