import sys
import os
from pathlib import Path
import torch
import numpy as np
import argparse
import json
from tqdm import tqdm
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
from utils.tokenizer import SimpleTokenizer
from utils_demo import load_demo_model, ensure_xyzrgb_torch
DATA_PATH = "XXX"
SAVE_JSON_PATH = "XXX"
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=1280)
    parser.add_argument("--pc_encoder_dim", type=int, default=512)
    parser.add_argument("--evaluate_3d", action="store_true", default=True)
    parser.add_argument("--num_group", type=int, default=512)
    parser.add_argument("--group_size", type=int, default=64)
    args, _ = parser.parse_known_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f">>> [1/4] Loading model (Baseline)...")
    model_wrapper, classnames = load_demo_model(args, device)
    student_model = getattr(model_wrapper, "baseline_student_model", model_wrapper.student_model)
    student_model.eval()
    try:
        logit_scale = model_wrapper.clip_model_text.logit_scale.exp().item()
    except:
        print("    [Info] Cannot get logit_scale directly, using default value 100.0")
        logit_scale = 100.0
    print(f"    [Info] Using Temperature Scaling (Logit Scale): {logit_scale:.2f}")
    tokenizer = SimpleTokenizer()
    print(f">>> [2/4] Generating standard text classification head...")
    text_prompts = [f"A point cloud of a {name}." for name in classnames]
    text_tokens = tokenizer(text_prompts).to(device)
    with torch.no_grad():
        text_features = model_wrapper.clip_model_text.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    print(f">>> [3/4] Loading attack data: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found: {DATA_PATH}")
    data = np.load(DATA_PATH)
    if 'points' in data: points_all = data['points']
    elif 'adv_points' in data: points_all = data['adv_points']
    else: raise KeyError("Cannot find points data")
    labels_all = data['labels']
    if points_all.shape[1] < points_all.shape[2]:
        points_all = points_all.transpose(0, 2, 1)
    if points_all.shape[-1] > 3:
        points_all = points_all[:, :, :3]
    print(f"    Processed data shape: {points_all.shape}")
    print(f"\n>>> [4/4] Start filtering (looking for High Confidence Mistakes)...")
    batch_size = 32
    candidates = []
    num_samples = len(points_all)
    for i in tqdm(range(0, num_samples, batch_size)):
        batch_end = min(i + batch_size, num_samples)
        batch_pts_np = points_all[i:batch_end]
        batch_pts_norm = []
        for pc in batch_pts_np:
            pc_norm = pc_normalize(pc)
            batch_pts_norm.append(torch.tensor(pc_norm, dtype=torch.float32))
        batch_pts = torch.stack(batch_pts_norm).to(device)
        batch_pts_6ch = ensure_xyzrgb_torch(batch_pts)
        with torch.no_grad():
            pc_feat = student_model.encode_pc(batch_pts_6ch)
            pc_feat = pc_feat / pc_feat.norm(dim=-1, keepdim=True)
            sim = pc_feat @ text_features.t()
            logits = sim * logit_scale
            probs = logits.softmax(dim=-1)
            vals, preds = torch.topk(probs, 1)
        for j in range(len(batch_pts)):
            global_idx = i + j
            pred_idx = preds[j].item()
            true_idx = int(labels_all[global_idx])
            confidence = vals[j].item()
            pred_name = classnames[pred_idx]
            true_name = classnames[true_idx]
            if pred_idx != true_idx and confidence > 0.5:
                item = {
                    "data_index": int(global_idx),
                    "true_class": true_name,
                    "baseline_prediction": pred_name,
                    "baseline_confidence": round(confidence, 4),
                    "suggested_query": f"Find the {true_name}"
                }
                candidates.append(item)
    if candidates:
        candidates.sort(key=lambda x: x['baseline_confidence'], reverse=True)
        with open(SAVE_JSON_PATH, "w", encoding='utf-8') as f:
            json.dump(candidates, f, indent=4, ensure_ascii=False)
        print(f"\n============================================================")
        print(f"Filtering complete!")
        print(f"Found {len(candidates)} high confidence misclassification samples.")
        print(f"Results saved to: {os.path.abspath(SAVE_JSON_PATH)}")
        print(f"============================================================")
        print("Top 3 most recommended demo cases (Conf should be high):")
        print(json.dumps(candidates[:3], indent=4))
    else:
        print("No high confidence misclassifications found. Logit Scale might be incorrect or model is too indecisive.")
if __name__ == "__main__":
    main()
