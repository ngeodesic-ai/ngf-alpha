# Noetic Geodesic Framework (Alpha)
🚀 **This is the Noetic Geodesic Framework (NGF)**, a pioneering approach to boost GPT-2's reasoning with symbolic nudging. Achieved 100.0% accuracy on 100 ARC and 100 MMLU tasks when nudged, with a 65.0% stock baseline on an A100 GPU.

## What’s This?
NGF enhances GPT-2 by adjusting its latent space using PCA and a symbolic nudge, tackling synthetic ARC patterns and MMLU challenges. It’s an alpha release—early, exciting, and open for collaboration!

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
This is alpha software! Help us refine prompts, test on other hardware, or improve the nudge. **Contributors must sign the [CLA](CLA.md) and email it to ic3moor@gmail.com before submitting pull requests.**

If you find this helpful, please leave a ⭐!
