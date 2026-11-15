import torch
import numpy as np
import json
import os
from tqdm import tqdm
import argparse
import torch.nn as nn
from utils_demo import load_demo_model
from utils.tokenizer import SimpleTokenizer
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'XXX')
BASE_DB_PATH = os.path.join(BASE_DIR, 'XXX')
MRPD_DB_PATH = os.path.join(BASE_DIR, 'XXX')
META_PATH = os.path.join(BASE_DIR, 'XXX')
SAVE_PATH = "XXX"
def encode_text_manual(clip_model, text):
    if text.dim() == 1:
        text = text.unsqueeze(0)
    cast_dtype = clip_model.transformer.get_cast_dtype()
    x = clip_model.token_embedding(text).to(cast_dtype)
    x = x + clip_model.positional_embedding.to(cast_dtype)
    mask = clip_model.attn_mask
    if mask is not None and mask.device != x.device:
        mask = mask.to(x.device)
    x = clip_model.transformer(x, attn_mask=mask)
    x = clip_model.ln_final(x)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
    if clip_model.text_projection is not None:
        if isinstance(clip_model.text_projection, nn.Linear):
            x = clip_model.text_projection(x)
        else:
            x = x @ clip_model.text_projection
    return x
def main():
    print(f">>> Filtering 'golden samples' (Baseline | MRPD)...")
    if not os.path.exists(BASE_DB_PATH):
        raise FileNotFoundError("Database file not found, please run 1_build_index.py first")
    db_base = torch.tensor(np.load(BASE_DB_PATH)).cuda()
    db_mrpd = torch.tensor(np.load(MRPD_DB_PATH)).cuda()
    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    labels = meta['labels']
    paths = meta['paths']
    classnames = meta['classnames']
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=1280)
    parser.add_argument("--pc_encoder_dim", type=int, default=512)
    parser.add_argument("--evaluate_3d", action="store_true", default=True)
    parser.add_argument("--num_group", type=int, default=512)
    parser.add_argument("--group_size", type=int, default=64)
    args, _ = parser.parse_known_args()
    model, _ = load_demo_model(args, 'cuda:0')
    tokenizer = SimpleTokenizer()
    try:
        logit_scale = model.clip_model_text.logit_scale.exp().item()
    except:
        logit_scale = 100.0
    print(">>> Batch calculating full database classification results...")
    all_texts = [f"A point cloud of a {c}." for c in classnames]
    text_tokens = tokenizer(all_texts).cuda()
    with torch.no_grad():
        all_text_feats = encode_text_manual(model.clip_model_text, text_tokens)
        all_text_feats = all_text_feats / all_text_feats.norm(dim=-1, keepdim=True)
    logits_base = (logit_scale * db_base @ all_text_feats.t())
    preds_base_idx = logits_base.argmax(dim=1).cpu().numpy()
    with torch.no_grad():
        mrpd_text_feats = model.get_text_feat()
        mrpd_text_feats = mrpd_text_feats / mrpd_text_feats.norm(dim=-1, keepdim=True)
    logits_mrpd = (logit_scale * db_mrpd @ mrpd_text_feats.t())
    preds_mrpd_idx = logits_mrpd.argmax(dim=1).cpu().numpy()
    golden_candidates = {}
    count = 0
    for i in tqdm(range(len(labels))):
        true_label_str = labels[i]
        base_pred_str = classnames[preds_base_idx[i]]
        mrpd_pred_str = classnames[preds_mrpd_idx[i]]
        if base_pred_str != true_label_str and mrpd_pred_str == true_label_str:
            score_mrpd = logits_mrpd[i, preds_mrpd_idx[i]].item()
            score_base_wrong = logits_base[i, preds_base_idx[i]].item()
            if true_label_str not in golden_candidates:
                golden_candidates[true_label_str] = []
            golden_candidates[true_label_str].append({
                "data_index": i,
                "true_class": true_label_str,
                "baseline_prediction": base_pred_str,
                "mrpd_prediction": mrpd_pred_str,
                "score_mrpd": score_mrpd,
                "baseline_wrong_conf": score_base_wrong
            })
            count += 1
    print(f"Found {count} golden adversarial samples meeting requirements, covering {len(golden_candidates)} categories.")
    final_list = []
    for k, v in golden_candidates.items():
        v.sort(key=lambda x: x['baseline_wrong_conf'] + x['score_mrpd'], reverse=True)
        best_case = v[0]
        final_list.append(best_case)
    with open(SAVE_PATH, 'w') as f:
        json.dump(final_list, f, indent=4)
    print(f"Best demo samples saved to {SAVE_PATH}")
    print("Sample content (Top 3):")
    print(json.dumps(final_list[:3], indent=4))
if __name__ == "__main__":
    main()
