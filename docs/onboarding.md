# Onboarding Guide for NGF-alpha

Welcome to the NGF-alpha project! This onboarding document is designed to help new contributors quickly get up to speed. Since the techniques here blend uncommon concepts from differential geometry, general relativity (GR), and AI latent spaces, we'll start with essential background knowledge. Then, we'll walk through the project's conceptual progression across its 10 completed stages, grouped into three overarching phases. This structure reflects how the idea evolved: from foundational proofs-of-concept to scalable embeddings and finally to LLM integration and validation.

The project aims to reduce AI hallucinations by warping semantic manifolds with physics-inspired curvature, guiding reasoning along deterministic "noetic geodesics" to truth-aligned outcomes. If you're familiar with GR's spacetime warping or geometric deep learning, you'll catch on fast—otherwise, the resources below will help.

## Prerequisites and Background Knowledge
To contribute effectively, you'll need a basic grasp of these areas. We've kept it concise with recommended resources (free where possible). Aim to spend 4-8 hours reviewing if you're new to any.

### 1. **Differential Geometry Basics**
   - **Why?** NGF treats latent spaces as curved manifolds, using geodesics (shortest paths in curved space) and metrics (e.g., Christoffel symbols) to simulate warping.
   - **Key Concepts:** Manifolds, Riemannian metrics, geodesics, curvature (e.g., Ricci scalar), Christoffel symbols for ODE-based path solving.
   - **Resources:**
     - [Differential Geometry | MIT OpenCourseWare](https://ocw.mit.edu/courses/18-950-differential-geometry-fall-2008/) (MIT notes; free PDF).
     - Youtube Videos:
         - [Differential Geometry in Under 15 Minutes](https://www.youtube.com/watch?v=oH0XZfnAbxQ)
         - [What is a manifold?](https://www.youtube.com/watch?v=zIjBArHTPZ4)
         - [Riemannian Manifolds in 12 Minutes](https://www.youtube.com/watch?v=jpjt08HkOzA)
         - [Conceptualizing the Christoffel Symbols](https://www.youtube.com/watch?v=TvFvL_sMg4g)
     - Book: [Introduction to Smooth Manifolds](https://julianchaidez.net/materials/reu/lee_smooth_manifolds.pdf) by John M. Lee (skim chapters 1-3; PDF often available via academic libraries).

### 2. **General Relativity (GR) Fundamentals**
   - **Why?** The core analogy: Semantic mass warps latent spaces like mass-energy curves spacetime, making wrong paths "geometrically unstable."
   - **Key Concepts:** Spacetime as a 4D manifold, Schwarzschild metric (for black hole-like wells), geodesic equations, gravitational time dilation, worldlines.
   - **Resources:**
     - Free notes: [Sean Carroll's GR Notes](https://archive.org/details/arxiv-gr-qc9712019) (skim intro and geodesic sections).
     - Youtube Videos:
         - [Einstein's General Relativity Explained](https://www.youtube.com/watch?v=AwhKZ3fd9JA) (PBS Space Time video; 15 min).
         - [Gravity Visualized](https://www.youtube.com/watch?v=MTY1Kje0yLg) (rubber sheet analogy; 10 min).
         - [The 4th Dimension in Relativity isn't Time - it's Space](https://www.youtube.com/watch?v=JFEMgaAusic)
         - [What Causes Gravitational Time Dilation?](https://www.youtube.com/watch?v=DjwQsKMh2v8)
         - [The Maths of General Relativity (1/8) - Spacetime and Worldlines](https://www.youtube.com/watch?v=xodtfM1r9FA)
         - [The Maths of General Relativity (3/8) - Geodesics](https://www.youtube.com/watch?v=3NnZzRb7L58)
         - [The Mathematics Behind the Schwarzschild Solution](https://www.youtube.com/watch?v=D-RTa_LCEvI)
         - [The Schwarzschild Metric: Complete Derivation | General Relativity](https://www.youtube.com/watch?v=6cSYZMM0wU4)

### 3. **AI Latent Spaces and Embeddings**
   - **Why?** NGF warps high-dimensional embeddings (e.g., R⁴ for 2x2 grids) to enforce determinism in models like GPT-2.
   - **Key Concepts:** Latent representations in transformers/LLMs, PCA for dimensionality reduction, hallucinations as probabilistic drift, symbolic nudging.
   - **Resources:**
     - [Latent Space in Deep Learning - Baeldung on Computer Science](https://www.baeldung.com/cs/dl-latent-space) (article; 10 min).
     - Hugging Face docs on [Transformers Embeddings](https://huggingface.co/docs/transformers/model_doc/gpt2) (focus on GPT-2).
     - YouTube Videos
         - [What are Word Embeddings?](https://www.youtube.com/watch?v=wgfSDrqYMJ4)
         - [Autoencoders | Deep Learning Animated](https://www.youtube.com/watch?v=hZ4a4NgM3u0)
         - [AI Hallucinations Explained - The Probability Problem](https://www.youtube.com/watch?v=u8tjByJtFrg)
     - Paper: [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913) (arXiv:1712.09913; skim for geometry in AI).

### 4. **Python and Tools Setup**
   - **Why?** All code is in Python; you'll run notebooks for simulations.
   - **Requirements:** Python 3.x, transformers==4.30.0, torch==2.4.1, numpy==1.26.4, scikit-learn==1.0.0. NVIDIA GPU (A100 recommended) for benchmarks.
   - **Setup:** `pip install -r requirements.txt` (create one if needed). Clone the repo and run `jupyter notebook` in each stage directory.
   - **Resources:** If new to Jupyter, [Quickstart Guide](https://jupyter.org/install).

### 5. **Project-Specific Reading**
   - Read the repo's README.md and /docs/article_latest.pdf (draft paper).
   - Zenodo memo: "Warped Semantic Manifolds" for conceptual overview.

Once comfortable, dive into the code—start with step1.ipynb and build up.

## Project Progression: Three Overarching Phases
The 12-step research plan (stages 1-10 completed; 11-12 planned for large benchmarks) evolved organically. It began with simple GR-inspired toys to prove warping reduces drift, then scaled to structured data and LLMs. We've grouped the 10 stages into three phases for clarity:

### Phase 1: Foundational Concepts and Toy Models (Stages 1-4)
This phase establishes the core physics-AI analogy: Warping flat spaces creates stable geodesics. Focus: Basic simulations in low dimensions (e.g., radial to R⁴ for 2x2 grids), proving wrong paths become unstable.

- **Stage 1: Toy Example** (toy-example/step1.ipynb)
  - Thinking: Start with radial geodesics in a Schwarzschild-like metric to simulate basic convergence.
  - Key: Mass M warps space; flat paths drift (hallucinate), warped spiral to singularity.
  - Dim: Effective radial (visualized in 3D); precursor to R⁴.

- **Stage 2: Embed Grid** (embed-grid/step2.ipynb)
  - Thinking: Embed simple structures (2x2 grids) into R⁴ vectors; test warping on real data.
  - Key: PCA projection to R³; logarithmic potential creates wells; geodesics guide to correct embeddings.
  - Evolution: Builds on stage 1 by adding grid features (sums, stats).

- **Stage 3: Rotation Matrix** (rotation-matrix/step3.ipynb)
  - Thinking: Introduce transformations (90° rotations) along geodesics.
  - Key: Position-dependent mass M_eff; stabilizes against noise, like GR frame-dragging.
  - Evolution: Applies stage 2 embeddings to dynamic ops.

- **Stage 4: Pattern Completion** (pattern-completion/step4.ipynb)
  - Thinking: Use geodesics for inference (complete partial patterns).
  - Key: Damping (gamma) and target pull; R⁴ traversals correct errors.
  - Evolution: Scales to 3x3/2x2 tasks; first hallucination reduction demo (~99% accuracy).

### Phase 2: Scaling to Higher Dimensions and Dynamics (Stages 5-7)
Here, the idea matures: Generalize to higher dims (R⁹+), add intelligence-like behaviors, and tackle rudimentary reasoning tasks. Focus: Prove scalability without losing determinism.

- **Stage 5: Higher-Dim Embeddings** (higher-dim-embeddings/step5.ipynb)
  - Thinking: Handle complex grids (3x3 in R⁹); use autoencoders for structure preservation.
  - Key: 9D geodesics; inverse square semantic mass; GIFs show warped vs. flat paths.
  - Evolution: Extends R⁴ from phase 1; multi-question setups for robustness.

- **Stage 6: Dynamic Intelligence** (dynamic-intelligence/step6.ipynb)
  - Thinking: Introduce adaptive warping for "intelligent" responses.
  - Key: Real-time mass adjustments; simulates cognitive flexibility.
  - Evolution: Builds on stage 5 dims with ODE tweaks for dynamics.

- **Stage 7: Rudimentary ARC** (rudimentary-arc/step7.ipynb)
  - Thinking: Apply to Abstraction and Reasoning Corpus (ARC)-like puzzles.
  - Key: Grid transformations in higher dims; benchmarks show 100% warped accuracy.
  - Evolution: First real task integration; ties back to phase 1 toys.

### Phase 3: LLM Integration and Validation (Stages 8-10)
Final phase: Bridge to real AI models (GPT-2), handle interferences, and benchmark. Focus: Empirical proof on synthetics, setting up for full-scale tests.

- **Stage 8: LLM Latent Embedding** (llm-latent-embedding/step8.ipynb)
  - Thinking: Embed LLM outputs (e.g., GPT-2 tokens) into warped manifolds.
  - Key: Symbolic nudges approximate geodesics; reduces probabilistic drift.
  - Evolution: Applies phase 2 scaling to NLP; first hallucination tests.

- **Stage 9: Warp Interference** (warp-interference/step9.py)
  - Thinking: Manage multiple warps (e.g., conflicting masses).
  - Key: Interference resolution via hybrid ODEs; ensures stability.
  - Evolution: Addresses real-world complexity in LLM spaces.

- **Stage 10: Small Benchmarks** (small-benchmarks/latest-arc-benchmark.py & latest-mmlu-benchmark.py)
  - Thinking: Validate on synthetic ARC/MMLU tasks.
  - Key: 100% warped accuracy, 0% hallucinations; compares stock vs. warped GPT-2.
  - Evolution: Culminates prior phases; blind tests highlight bias mitigation.

## Next Steps for Contributors
1. **Run the Code:** Start with phase 1 notebooks on Colab (A100 GPU for speed).
2. **Contribute:** Sign the CLA (CLA.md) and email to ngeodesic@gmail.com. Focus on stages 11-12 (large benchmarks) or extensions (e.g., R¹⁶ for 4x4 grids).
3. **Feedback:** Join discussions via issues/PRs. Questions? Post in the repo or email.
4. **Future Phases:** Stages 11-12: Full ARC/MMLU; potential neuroscience ties.

Thanks for your interest—let's warp AI reasoning together! 🚀