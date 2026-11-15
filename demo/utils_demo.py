import sys
import os
import torch
from pathlib import Path
import importlib.util
import copy
import numpy as np
from torchvision import transforms

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import utils.open_clip as open_clip

def get_transform():
    return transforms.Compose([
        lambda pc: torch.tensor(pc, dtype=torch.float32)
    ])

def ensure_xyzrgb_torch(pc: torch.Tensor) -> torch.Tensor:

    if not torch.is_tensor(pc):
        raise TypeError(f"pc must be torch.Tensor, got {type(pc)}")
    if pc.dim() not in (2, 3):
        raise ValueError(f"Point cloud must be 2D or 3D tensor, got shape={tuple(pc.shape)}")
    if pc.shape[-1] < 3:
        raise ValueError(f"Point cloud last dim must be >=3 for XYZ, got shape={tuple(pc.shape)}")
    xyz = pc[..., :3]
    rgb = torch.full(xyz.shape[:-1] + (3,), 0.5, device=pc.device, dtype=pc.dtype)
    return torch.cat([xyz, rgb], dim=-1)

def load_demo_model(args, device='cuda:0'):
    train_script_path = parent_dir / 'XXX'
    if not train_script_path.exists():
        raise FileNotFoundError(f"Training script not found: {train_script_path}")

    spec = importlib.util.spec_from_file_location("ulip2_text_point_xyzrgb_train", str(train_script_path))
    train_module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load training script module: {train_script_path}")
    spec.loader.exec_module(train_module)

    get_ulip2_pointbert_student = train_module.get_ulip2_pointbert_student
    PointPrompt = train_module.PointPrompt

    print(f">>> [Model Loader] Initializing CLIP (Text Teacher)...")
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='XXX')
    clip_model.to(device)

    print(f">>> [Model Loader] Initializing ULIP-2 Student...")
    baseline_student_model = get_ulip2_pointbert_student(args, clip_model)
    baseline_student_model.to(device)
    baseline_student_model.eval()

    student_model_for_prompt = copy.deepcopy(baseline_student_model)
    student_model_for_prompt.to(device)
    student_model_for_prompt.eval()

    classnames_path = 'XXX'
    if not os.path.exists(classnames_path):
        raise FileNotFoundError(f"Class names file not found: {classnames_path}")

    with open(classnames_path, 'r') as f:
        classnames = [line.strip() for line in f.readlines()]

    print(f">>> [Model Loader] Loading trained best prompt...")
    prompt_dir = Path("XXX")
    if not prompt_dir.exists() or not prompt_dir.is_dir():
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

    prompt_candidates = sorted(
        [
            p for p in prompt_dir.glob("*best_prompt*.pt")
            if "_ctx_" not in p.name and "_point_prompt" not in p.name and "_tokens" not in p.name
        ]
    )
    if len(prompt_candidates) == 0:
        raise FileNotFoundError(f"Best prompt main file not found in directory: {prompt_dir}")

    prompt_path = prompt_candidates[0]
    print(f">>> [Model Loader] Using best prompt: {prompt_path}")

    model = PointPrompt(
        prompt_len=10,
        feat_dim=384,
        student_model=student_model_for_prompt,
        device=device,
        classnames=classnames,
        text_prompt_size=7,
        soft_prompt_size=3,
        clip_model_text=clip_model,
        ckpt_path=str(prompt_path)
    )
    model.baseline_student_model = baseline_student_model
    model.eval()
    return model, classnames
