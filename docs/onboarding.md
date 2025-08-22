# Onboarding Guide for NGF-alpha

Welcome to the NGF-alpha project! This onboarding document is designed to help new contributors quickly get up to speed. Since the techniques here blend uncommon concepts from differential geometry, general relativity (GR), and AI latent spaces, we'll start with essential background knowledge. Then, we'll walk through the project's conceptual progression across its 10 completed stages, grouped into three overarching phases. This structure reflects how the idea evolved: from foundational proofs-of-concept to scalable embeddings and finally to LLM integration and validation.

The project aims to reduce AI hallucinations by warping semantic manifolds with physics-inspired curvature, guiding reasoning along deterministic "noetic geodesics" to truth-aligned outcomes. If you're familiar with GR's spacetime warping or geometric deep learning, you'll catch on fast; otherwise, the resources provided below will help.

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
   - Read the repo's README.md and [/docs/article_latest.pdf](/docs/article_latest.pdf) (latest draft paper).
   - Zenodo memo: [Warped Semantic Manifolds](https://zenodo.org/records/16731201) for conceptual overview.

Once comfortable, dive into the code—start with step1.ipynb and build up.

## Project Progression: Three Overarching Phases
The 12-step research plan (stages 1-10 completed; 11-12 planned for large benchmarks) evolved organically. It began with simple GR-inspired toys to prove warping reduces drift, then scaled to structured data and LLMs. We've grouped the 12 stages into three phases for clarity:

### Phase 1: Foundational Concepts and Toy Models (Stages 1-4)
This phase establishes the core physics-AI analogy using the minimal number of dimensions, R⁴. The focus: basic simulations in the lowest dimension (e.g., radial to R⁴ for 2x2 grids). The main finding; we establish that warping the latent space with semantic mass creates stable, deterministic geodesic paths that significantly reduce errors (akin to hallucinations) compared to flat space, where paths are prone to probabilistic drift and instability.

- **Stage 1: Toy Example** (toy-example/step1.ipynb)
  - Thinking: Start with radial geodesics in a Schwarzschild-like metric to simulate basic convergence.
  - Key: Mass M warps space; flat paths drift (hallucinate), warped spiral to singularity.
  - Dim: Effective radial (visualized in 3D); precursor to R⁴.
  - Main Finding: Warping a latent space with a physics-inspired, Schwarzschild-like metric creates stable, deterministic geodesic paths that converge to a central point (akin to a noetic singularity), while flat (unwarped) spaces allow chaotic, drifting paths analogous to AI hallucinations. This finding validates the project’s foundational hypothesis: introducing GR-inspired curvature (semantic mass) into latent spaces can eliminate erratic reasoning paths, setting the stage for scaling to R⁴ embeddings (step2) and LLM integration (step8).

- **Stage 2: Embed Grid** (embed-grid/step2.ipynb)
  - Thinking: Embed simple structures (2x2 grids) into R⁴ vectors; test warping on real data.
  - Key: PCA projection to R³; logarithmic potential creates wells; geodesics guide to correct embeddings.
  - Evolution: Stage 2 introduces real data (2x2 grids, like ARC puzzles), unlike Stage 1’s abstract radial setup.
  - Main Finding: Warping R⁴ latent space with semantic mass (inspired by GR’s curvature) guides grid transformations (e.g., rotations, completions) along noetic geodesics, achieving ~75-100% accuracy on synthetic tasks, compared to ~75% in flat space. The use of structured grid data and feature embeddings allows the warped manifold to capture meaningful patterns, making correct paths more stable than in Stage 1’s abstract radial model.

- **Stage 3: Rotation Matrix** (rotation-matrix/step3.ipynb)
  - Thinking: Introduce transformations (90° rotations) along geodesics.
  - Key: Position-dependent mass M_eff; stabilizes against noise, like GR frame-dragging.
  - Evolution: Stage 3 introduces active operations (90° rotations) to manipulate grid embeddings, unlike Stage 2’s static mappings.
  - Main Finding: Incorporating dynamic transformations (rotations) into the warped R⁴ latent space significantly improves the robustness of reasoning tasks, as the position-dependent M_eff stabilizes paths against noise (e.g., rotational errors), achieving near-perfect accuracy (~99-100%) compared to Stage 2’s ~75-100%. The warping now dynamically adjusts to the transformation’s context, akin to GR’s frame-dragging, where rotating masses (like black holes) influence nearby geodesics.

- **Stage 4: Pattern Completion** (pattern-completion/step4.ipynb)
  - Thinking: Use geodesics for inference (complete partial patterns).
  - Key: Damping (gamma) and target pull; R⁴ traversals correct errors.
  - Evolution: Stage 4 shifts to inferring missing grid elements, a more complex reasoning task than Stage 3’s application of known rotations.
  - Main Finding: Stage 4’s finding—that warped R⁴ spaces with damping and target pull enable near-perfect pattern completion—demonstrates the framework’s ability to handle generative reasoning tasks, a significant step beyond Stage 3’s transformation focus; first hallucination reduction demo with ~99% accuracy.

### Phase 2: Scaling to Higher Dimensions and Dynamics (Stages 5-7)
Here, the idea matures where we generalize to higher dims (R⁹+), add intelligence-like behaviors, and tackle rudimentary reasoning tasks with the first real task integration (ie, ARC-like tasks). The focus: prove scalability without losing determinism. This means demonstrating that the framework’s core mechanism—warping high-dimensional latent spaces with semantic mass to guide reasoning along deterministic noetic geodesics—remains effective and reliable as the dimensionality and complexity of tasks increase (ie, from R⁴ in Phase 1 to R⁹ in Phase 2),

- **Stage 5: Higher-Dim Embeddings** (higher-dim-embeddings/step5.ipynb)
  - Thinking: Handle complex grids (3x3 in R⁹); use autoencoders for structure preservation.
  - Key: 9D geodesics; inverse square semantic mass; GIFs show warped vs. flat paths.
  - Evolution: Extends R⁴ from phase 1; multi-question setups for complex tasks, enhancing robustness and generalization while maintaining determinism.
  - Main Finding: Warping R⁹ spaces with autoencoders and multi-question setups maintains ~99-100% accuracy and zero hallucinations; this is critical, as it demonstrates the framework’s scalability to higher-dimensional, complex tasks (a critical step toward Phase 3’s LLM integration in step8).

- **Stage 6: Dynamic Intelligence** (dynamic-intelligence/step6.ipynb)
  - Thinking: Introduce adaptive warping for "intelligent" responses.
  - Key: Real-time mass adjustments; simulates cognitive flexibility.
  - Evolution: Enhances the ODE system with adaptive mass terms that update based on input context; this ensures geodesics remain stable across dynamic tasks, reducing potential drift.
  - Main Finding: Introduce real-time adaptive warping of the R⁹ latent space, through dynamic adjustments to semantic mass; this enhances the framework's ability to simulate cognitive flexibility, thus achieving robust convergence across varied tasks while maintaining near-perfect accuracy (~99-100%) and zero hallucination rates.

- **Stage 7: Rudimentary ARC** (rudimentary-arc/step7.ipynb)
  - Thinking: Apply to Abstraction and Reasoning Corpus (ARC)-like puzzles.
  - Key: Grid transformations in higher dims; benchmarks show 100% warped accuracy.
  - Evolution: First real task integration (eg. ARC task); ties back to phase 1 toys.
  - Main Finding: Warping R⁹ spaces achieves 100% accuracy and zero hallucinations on ARC-like puzzles—demonstrates the framework’s practical applicability to complex, reasoning-intensive AI tasks, a significant step beyond Stage 6’s abstract flexibility.

### Phase 3: LLM Integration and Validation (Stages 8-12)
In the final phase, we bridge to real AI models (GPT-2), handle interferences, and benchmark tasks like ARC and MMLU questions. Here we transition NGF from theoretical scalability (Phase 2) to practical LLM applications, proving that warping latent spaces can reduce hallucinations in real AI systems like GPT-2. The focus: empirical proof on synthetics, setting up for full-scale tests. 

- **Stage 8: LLM Latent Embedding** (llm-latent-embedding/step8.ipynb)
  - Thinking: Embed LLM outputs (e.g., GPT-2 tokens) into warped manifolds.
  - Key: Symbolic nudges approximate geodesics; reduces probabilistic drift.
  - Evolution: Transitioning from structured, abstract reasoning tasks (ARC puzzles) to integrating the warped manifold framework with real-world LLMs, adapting the warping mechanism to handle probabilistic, unstructured data like text.
  - Main Finding: Warping GPT-2 embeddings with symbolic nudges reduces hallucinations in NLP—demonstrates the framework’s practical applicability to real-world language models; this represents a major leap from Stage 7’s abstract puzzles

- **Stage 9: Warp Interference** (warp-interference/step9.py)
  - Thinking: Manage multiple warps (e.g., conflicting masses).
  - Key: Interference resolution via hybrid ODEs; ensures stability.
  - Evolution: Transitioning from single-concept LLM warping to managing complex, multi-concept interactions by synthesizing multiple geodesic paths into one, thus handling multiple semantic masses at once.
  - Main Finding: Hybrid ODEs resolve multi-mass interference while maintaining ~99-100% accuracy and near-zero hallucinations—demonstrates the framework’s ability to handle real-world NLP complexities.

- **Stage 10: Small Benchmarks** (small-benchmarks/latest-arc-benchmark.py & latest-mmlu-benchmark.py)
  - Thinking: Validate on synthetic ARC/MMLU tasks.
  - Key: 100% warped accuracy, 0% hallucinations; compares stock vs. warped GPT-2.
  - Evolution: Transition from interference resolution in abstract NLP tasks to validating the framework on standardized AI benchmarks, ensuring empirical robustness and generalizability, while intriducing blind testing.
  - Main Finding: Warping achieves 100% accuracy and zero hallucinations on synthetic ARC/MMLU tasks, thus validating the framework’s real-world applicability and trigging the public rollout.
 
- **Stage 11: Large Benchmarks (Coming)** (large-benchmarks/)
  - Thinking: Scaling the framework to real-world, large-scale datasets.
  - Evolution: Address the limitation on small synthetic datasets.
  - Expected Finding: Warping higher-dimensional latent spaces (e.g., R¹⁶ or beyond) with hybrid ODEs achieves near-perfect accuracy (~95-100%) and near-zero hallucination rates on full-scale ARC and MMLU datasets, significantly outperforming stock LLMs (e.g., ~60-70% for GPT-3 on ARC, ~80% on MMLU).
 
- **Stage 12: Milestone Benchmarks (Coming)** (milestone-benchmarks/)
  - Thinking: Final culmination of the project’s 12-step research plan.
  - Evolution: Moving from large-scale validation on standardized datasets (ARC, MMLU) to achieving a milestone benchmark using more advanced models.
  - Expected Finding: Warping high-dimensional latent spaces (e.g., R⁶⁴ or R¹²⁸) with hybrid ODEs achieves near-perfect accuracy (~95-100%) and near-zero hallucination rates across diverse AI benchmarks (e.g., BIG-Bench, CommonsenseQA). 

## Next Steps for Contributors
1. **Run the Code:** Start with phase 1/step 1 and progress through to phase 3/step 10 using notebooks on Colab (A100 GPU for speed).
2. **Contribute:** Sign the CLA (CLA.md) and email to ngeodesic@gmail.com. Focus on stages 11-12 (large benchmarks) or extensions (e.g., R¹⁶ for 4x4 grids).
3. **Feedback:** Join discussions via issues/PRs. Questions? Post in the repo or email.
4. **Future Phases:** Stages 11-12: Full ARC/MMLU; refine the math

Thanks for your interest—let's warp AI reasoning together! 🚀