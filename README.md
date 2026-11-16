# MRPD+

Code for **MRPD+: Semantic and Geometric Prompt Adaptation for Adversarially Robust 3D Point Cloud Recognition**.

MRPD+ combines discrete and continuous text prompts with Post-FPS Point Prompts (PFPP) and Post-FPS Geometric Reprogramming (PFGR). The implementation covers Uni3D and ULIP-2 on ModelNet40 and ScanObjectNN.

## Repository Structure

- `train/`: Training scripts. Files ending in `_improved.py` implement MRPD+; `_old.py` files contain earlier implementations.
- `test/`: Adversarial attack and evaluation scripts.
- `demo/`: Language-assisted object-selection demonstration.

## Preparation

Prepare the required backbone code, dependencies, datasets, and pretrained weights. Replace `XXX` placeholders with the corresponding resource paths before running the scripts.
