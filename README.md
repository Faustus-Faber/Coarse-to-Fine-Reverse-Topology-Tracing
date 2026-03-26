# Coarse-to-Fine Video Frame Interpolation (VFI)

This repository contains the codebase, evaluation metrics, and testing scenarios for an advanced, AI-free Coarse-to-Fine Video Frame Interpolation pipeline. The algorithm relies strictly on geometric structural ghost geolocation, pristine pixel gathering, and Farneback Optical Flow magnitude cross-fading to generate temporally coherent sub-millisecond frames without neural network hallucinations.

## 📂 Repository Structure

The workspace has been rigorously organized to separate generation logic, continuous integration testing, and formal academic evaluation metrics.

* **`/src`**: Core pipeline source code. 
  * `interpolator_core.py` - The primary $O(N)$ VFI mathematical algorithm featuring static HUD pinning.
  * `pipeline_orchestrator.py` - End-to-end framework for batched processing.
  * `dataset_builder.py` - Scripts to compile the raw input datasets.
  * `figure_generator.py` - Matplotlib graphing logic for the LaTeX paper results.

* **`/evaluation`**: Benchmarking, statistical aggregations, and visual integrity code utilized to generate the paper's primary claims.
  * `eval_core_metrics.py` - Parses basic PSNR and SSIM.
  * `metric_lpips_calculator.py` - Computes LPIPS perceptual similarity.
  * `baseline_comparator.py` - Formal comparison engine versus baseline interpolation models.
  * *Other specific scenario evaluators for multi-frame coherence and 1080p scaling.*

* **`/testing`**: Proof of concept simulators and isolated scenario generators.
  * `sim_base_interpolation.py` - The testbed simulation for standard interpolation mathematically.
  * `sim_fantasy_occlusion.py` - Isolates complex geometric boundary occlusions (e.g., dragons breaking topology).
  * `test_scenario_engine.py` - Core test harness for the data permutations.

## 🚀 Quick Start

To run the foundational interpolator daemon natively:
```bash
# General frame interpolation (6x multiplier)
python src/interpolator_core.py "dataset_shmup_30fps_in.mp4" 6 "dataset_shmup_output_x6.mp4"

# Real-time C++ Game Engine IPC mode
python src/interpolator_core.py --ipc 1280 720
```

## ⚖️ License & Compilation
This codebase acts as the technical supplementary material for this paper's submission. Raw `.mp4` datasets and LaTeX `.pdf` compilation artifacts are explicitly omitted via `.gitignore` to maintain repository speed and compliance limits.
