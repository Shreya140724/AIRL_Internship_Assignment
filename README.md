# AIRL Internship Assignment — README

## Project Title
AIRL Internship Assignment — Vision Transformer (from-scratch) & Text-driven Segmentation (CLIPSeg / GroundingDINO → SAM2)

---

## Introduction
This repository contains two Colab-ready notebooks for the AIRL internship assignment:

- `q1.ipynb` — Vision Transformer (ViT) implemented in PyTorch from scratch. Trains on CIFAR-10 and saves the best checkpoint.
- `q2.ipynb` — Text-driven segmentation pipeline combining CLIPSeg / GroundingDINO with SAM 2. Colab-ready instructions and example prompts included.

The goal is to provide reproducible notebooks that run on Colab GPU and demonstrate (1) building/training a ViT and (2) a prompt-driven segmentation pipeline using state-of-the-art modules.

---

## Table of Contents
1. [Quickstart (Colab)](#quickstart-colab)  
2. [Notebooks](#notebooks)  
3. [Installation / Dependencies](#installation--dependencies)  
4. [How to run](#how-to-run)  
5. [Best Q1 config (example)](#best-q1-config-example)  
6. [Results](#results)  
7. [Configuration options](#configuration-options)  
8. [Documentation & Notes](#documentation--notes)  
9. [Examples](#examples)  
10. [Troubleshooting](#troubleshooting)  
11. [Contributors](#contributors)  
12. [License](#license)

---

## Quickstart (Colab)
1. Open a new Google Colab notebook.
2. Upload `q1.ipynb` or `q2.ipynb` (or open directly from GitHub).
3. Runtime → Change runtime type → Hardware accelerator → **GPU**.
4. Run cells top-to-bottom. The notebooks are organized so cells execute sequentially.

---

## Notebooks
- `q1.ipynb` — Vision Transformer (PyTorch, scratch)
  - Build ViT model (patch embedding, transformer encoder, classification head)
  - Data loading and CIFAR-10 preprocessing/augmentation
  - Training loop with checkpointing (saves best validation/test checkpoint)
  - Logging of loss/accuracy metrics
- `q2.ipynb` — Text-driven segmentation
  - Integrates text localization (GroundingDINO / CLIPSeg style) with SAM 2 for mask generation
  - Colab-ready installation steps for required packages and model weights
  - Example prompts and visualization of segmentation masks

---

## Installation / Dependencies
Notebooks are Colab-ready. Below is a typical local / Colab dependency list (versions are examples — use compatible/latest as needed):

- Python 3.8+
- PyTorch (CUDA-enabled in Colab): `torch`, `torchvision`
- numpy, matplotlib, tqdm, scikit-learn
- For `q2.ipynb`: `transformers`, `diffusers` (if needed), `opencv-python`, `Pillow`
- Model-specific repos / pip packages for GroundingDINO, CLIPSeg, and SAM 2 (see `q2.ipynb` for exact install commands)

Example pip installs (run in a notebook cell):
```bash
# core
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tqdm scikit-learn

# text-to-mask pipeline (example)
pip install transformers
pip install opencv-python pillow
```

> See `q2.ipynb` for the precise install steps — some SAM 2 builds require specific wheel installs or patched CUDA compatibility on Colab.

---

## How to run
### q1.ipynb (ViT)
1. Ensure GPU runtime is selected.
2. Open `q1.ipynb`.
3. Edit hyperparameters at the top cell if you want to change architecture or training settings.
4. Run all cells. The notebook:
   - Downloads CIFAR-10 automatically.
   - Builds the ViT with configurable `patch_size`, `embed_dim`, `depth`, `num_heads`.
   - Trains and evaluates, saving the **best checkpoint** to the Colab filesystem.

### q2.ipynb (Text-driven segmentation)
1. Ensure GPU runtime is selected (for heavy models).
2. Open `q2.ipynb`.
3. Run install cells (these may clone repos and download weights — large files).
4. Provide text prompts (example cells included) and run inference cells.
5. Visualize masks and optionally export results.

---

## Best Q1 config (example)
```text
patch_size: 4
embed_dim: 256
depth: 12
num_heads: 8
epochs: 50
batch_size: 128
```

---

## Results
- CIFAR-10 test accuracy (best run): **81.41%**  

---

## Configuration options
- `patch_size`, `embed_dim`, `depth`, `num_heads`
- Optimizer / scheduler: learning rate, weight decay, warmup steps
- Training: `epochs`, `batch_size`, `mixup`/`cutmix`
- Augmentation: `RandomCrop`, `Flip`, `ColorJitter`

---

## Documentation & Notes
- **Patch size:** Smaller patch (e.g., 4) increases token count and captures local detail.
- **Depth vs Width:** Increasing `embed_dim` improves representations; deeper models need LR warmup.
- **Augmentations:** `RandomCrop`, `Flip`, `ColorJitter`, `MixUp`, `CutMix`.
- **SAM 2 pipeline:** Large weights; ambiguous prompts reduce quality — clearer prompts help.

---

## Examples
### ViT Training
```python
config = {
  "patch_size": 4,
  "embed_dim": 256,
  "depth": 12,
  "num_heads": 8,
  "epochs": 50,
  "batch_size": 128,
  "lr": 3e-4,
}
```

## Troubleshooting
- **OOM on Colab:** Reduce `batch_size` or `embed_dim`, increase `patch_size`.
- **Weights download errors:** Check internet connection and Colab permissions.
- **SAM 2 CUDA issues:** Use CPU or alternative runtime.
- **Ambiguous prompts:** Be more descriptive.

---

## Contributors
- Original assignment: AIRL Internship Assignment
- Notebooks: (Shreya Sidabache)

---

## License
MIT License

Copyright (c) 2025 <Shreya Sidabache>
