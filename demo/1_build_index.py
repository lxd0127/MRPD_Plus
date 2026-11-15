import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from utils_demo import load_demo_model, ensure_xyzrgb_torch
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if m > 0: pc = pc / m
    return pc
def build_index():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=1280)
    parser.add_argument("--pc_encoder_dim", type=int, default=512)
    parser.add_argument("--evaluate_3d", action="store_true", default=True)
    parser.add_argument("--num_group", type=int, default=512)
    parser.add_argument("--group_size", type=int, default=64)
    args, _ = parser.parse_known_args()
    device = "cuda:0"
    model, classnames = load_demo_model(args, device)
    baseline_student = getattr(model, "baseline_student_model", model.student_model)
    DATA_PATH = "XXX"
    print(f">>> [Build] Building database from attack data: {DATA_PATH}")
    data = np.load(DATA_PATH)
    if 'points' in data: points_all = data['points']
    elif 'adv_points' in data: points_all = data['adv_points']
    else: raise KeyError("Cannot find points data")
    labels_all = data['labels']
    if points_all.shape[1] < points_all.shape[2]:
        points_all = points_all.transpose(0, 2, 1)
    if points_all.shape[-1] > 3:
        points_all = points_all[:, :, :3]
    print(f"    Data size: {len(points_all)}")
    feats_base = []
    feats_mrpd = []
    file_paths = []
    labels_str = []
    print(">>> [Build] Extracting dual-path features (Baseline vs MRPD)...")
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(points_all), batch_size)):
            batch_end = min(i+batch_size, len(points_all))
            batch_np = points_all[i:batch_end]
            batch_pts = []
            for j, p in enumerate(batch_np):
                batch_pts.append(torch.tensor(pc_normalize(p), dtype=torch.float32))
                idx = i + j
                file_paths.append(f"attack_data_idx_{idx}")
                labels_str.append(classnames[int(labels_all[idx])])
            batch_pts = torch.stack(batch_pts).to(device)
            batch_pts_6ch = ensure_xyzrgb_torch(batch_pts)
            f_base = baseline_student.encode_pc(batch_pts_6ch)
            f_base = f_base / f_base.norm(dim=-1, keepdim=True)
            feats_base.append(f_base.cpu().numpy())
            f_mrpd, _ = model(batch_pts_6ch)
            f_mrpd = f_mrpd / f_mrpd.norm(dim=-1, keepdim=True)
            feats_mrpd.append(f_mrpd.cpu().numpy())
    save_dir = Path(__file__).resolve().parent / 'XXX'
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / 'XXX', np.concatenate(feats_base))
    np.save(save_dir / 'XXX', np.concatenate(feats_mrpd))
    metadata = {
        "paths": file_paths,
        "labels": labels_str,
        "classnames": classnames,
        "source": "attack_npz"
    }
    with open(save_dir / 'XXX', "w") as f:
        json.dump(metadata, f)
    print(f">>> [Build] Database build complete!")
    print(f"    Baseline DB: {save_dir / 'XXX'}")
    print(f"    MRPD DB:     {save_dir / 'XXX'}")
if __name__ == "__main__":
    build_index()
