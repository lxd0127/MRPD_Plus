# SAGPA

Code for **SAGPA: Semantic and Geometric Prompt Adaptation for Adversarially Robust 3D Point Cloud Recognition**.

SAGPA combines discrete and continuous text prompts with Post-FPS Point Prompts (PFPP) and Post-FPS Geometric Reprogramming (PFGR). The implementation covers Uni3D and ULIP-2 on ModelNet40 and ScanObjectNN.

## Repository Structure

- `train/`: Training scripts. Files ending in `_improved.py` implement SAGPA; `_old.py` files contain earlier implementations.
- `test/`: Adversarial attack and evaluation scripts.
- `demo/`: Language-assisted object-selection demonstration.

## Preparation

Prepare the required backbone code, dependencies, datasets, and pretrained weights. Replace `XXX` placeholders with the corresponding resource paths before running the scripts.
