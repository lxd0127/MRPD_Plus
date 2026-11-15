import os
import sys
import argparse
import gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from loguru import logger
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
try:
    import utils.open_clip as open_clip
    from models.ULIP_models import ULIP2_WITH_OPENCLIP
    from models.pointbert.point_encoder import PointTransformer_Colored
    from utils.utils import cfg_from_yaml_file
    from utils.tokenizer import SimpleTokenizer
    from models.point_encoder import fps6d
    from AOF import CWAOF
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)
CONF = {
    "batch_size_eval": 32,
    "batch_size_gen": 16,
    "num_workers": 4,
    "embed_dim": 1280,
    "best_prompt_path": "XXX",
    "clip_model_path": "XXX",
    "whitebox_dir": "XXX",
    "whitebox_file": 'XXX',
    "modelnet_root": "XXX",
    "clean_npz": "XXX",
    "blackbox_path": "XXX",
    "student_ckpt": "XXX",
    "pointbert_config": "XXX",
}
Path(CONF["whitebox_dir"]).mkdir(parents=True, exist_ok=True)
WHITEBOX_PATH = os.path.join(CONF["whitebox_dir"], CONF["whitebox_file"])
class PFGR_Module(nn.Module):
    def __init__(self, original_layer, lambda_scale=1.0):
        super().__init__()
        self.original_layer = original_layer
        self.lambda_scale = lambda_scale
        self._is_reparameterized = False
        for param in self.original_layer.parameters():
            param.requires_grad = False
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        self.pfgr_branch = nn.Linear(in_features, out_features, bias=False)
    def forward(self, x):
        if self._is_reparameterized or self.lambda_scale == 0.0:
            return self.original_layer(x)
        return self.original_layer(x) + self.lambda_scale * self.pfgr_branch(x)
def _set_submodule(root: nn.Module, path: str, new_module: nn.Module):
    parts = path.split('.')
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)
class PostFPSPointPrompt(nn.Module):
    def __init__(self, prompt_xyz: torch.Tensor):
        super().__init__()
        self.prompt_xyz = nn.Parameter(prompt_xyz)
        self.hook_handle = None
    def forward_hook(self, module, inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            return output
        neighborhood = output[0]
        center = output[1]
        B = center.shape[0]
        prompt_xyz_batch = self.prompt_xyz.to(device=center.device, dtype=center.dtype).expand(B, -1, -1)
        P = self.prompt_xyz.shape[1]
        S = neighborhood.shape[2]
        C = neighborhood.shape[3]
        prompt_neighborhood = torch.zeros((B, P, S, C), device=neighborhood.device, dtype=neighborhood.dtype)
        neighborhood_new = torch.cat([neighborhood, prompt_neighborhood], dim=1)
        center_new = torch.cat([center, prompt_xyz_batch], dim=1)
        new_out = list(output)
        new_out[0] = neighborhood_new
        new_out[1] = center_new
        return tuple(new_out)
    def attach(self, point_encoder: nn.Module):
        group_module = None
        if hasattr(point_encoder, 'group_divider'):
            group_module = getattr(point_encoder, 'group_divider')
        else:
            for name, m in point_encoder.named_modules():
                if 'group_divider' in name.lower() or m.__class__.__name__.lower() == 'group':
                    group_module = m
                    break
        if group_module is None:
            logger.warning("Could not find GroupDivider to attach Post-FPS prompt.")
            return
        self.hook_handle = group_module.register_forward_hook(self.forward_hook)
    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
tokenizer = SimpleTokenizer()
class PromptBundle:
    def __init__(self):
        self.ctx_soft: torch.Tensor = None
        self.ctx_cont: torch.Tensor = None
        self.ctx_tokens: torch.Tensor = None
        self.ctx_positions: torch.Tensor = None
        self.n_soft: int = 0
        self.n_hard: int = 0
        self.pfgr_weight: dict = None
        self.point_prompt_xyz: torch.Tensor = None
def load_prompt_bundle(prompt_ckpt: str, clip_model, tokenizer, classnames, device):
    if not os.path.exists(prompt_ckpt):
        raise FileNotFoundError(f"Missing best prompt file: {prompt_ckpt}")
    prefix = prompt_ckpt.replace(".pt", "")
    ctx_soft_path = prefix + "_ctx_soft.pt"
    ctx_cont_path = prefix + "_ctx_cont.pt"
    ctx_tokens_path = prefix + "_ctx_tokens.pt"
    point_prompt_path = prefix + "_point_prompt.pt"
    b = PromptBundle()
    b.ctx_soft = torch.load(ctx_soft_path, map_location="cpu").to(device).float()
    if os.path.exists(ctx_tokens_path):
        b.ctx_tokens = torch.load(ctx_tokens_path, map_location="cpu")
        if not isinstance(b.ctx_tokens, torch.Tensor):
            b.ctx_tokens = torch.tensor(b.ctx_tokens, dtype=torch.long)
        b.ctx_tokens = b.ctx_tokens.to(device).long()
    elif os.path.exists(ctx_cont_path):
        b.ctx_cont = torch.load(ctx_cont_path, map_location="cpu").to(device).float()
    b.n_soft = int(b.ctx_soft.shape[0])
    if b.ctx_tokens is not None:
        b.n_hard = int(b.ctx_tokens.shape[0])
    else:
        b.n_hard = int(b.ctx_cont.shape[0])
    soft_prefix = " ".join(["X"] * b.n_soft)
    hard_prefix = " ".join(["X"] * b.n_hard)
    prompts = [f"{soft_prefix} A depth picture of a {hard_prefix} {name}." for name in classnames]
    tokenized_list = tokenizer(prompts)
    if isinstance(tokenized_list, list):
        tokenized = torch.stack(tokenized_list).to(device)
    else:
        tokenized = tokenized_list.to(device)
    def _get_single_token_id(word, add_space):
        text = (" " + word) if add_space else word
        ids = tokenizer(text)
        if isinstance(ids, torch.Tensor) and ids.ndim==2: ids=ids[0]
        eot_pos = int(ids.argmax().item())
        content = ids[1:eot_pos]
        content = content[content!=0]
        return int(content.item()) if content.numel() == 1 else None
    x_id_space = _get_single_token_id("X", True)
    x_id_nospace = _get_single_token_id("X", False)
    first = tokenized[0]
    positions = []
    if x_id_space is not None:
        positions.append((first == x_id_space).nonzero(as_tuple=False).squeeze(-1))
    if x_id_nospace is not None and x_id_nospace != x_id_space:
        positions.append((first == x_id_nospace).nonzero(as_tuple=False).squeeze(-1))
    positions = torch.cat(positions, dim=0)
    positions = torch.unique(positions).sort()[0]
    b.ctx_positions = positions[:(b.n_soft + b.n_hard)].to(device)
    b.pfgr_weight = torch.load(prompt_ckpt, map_location="cpu")
    if os.path.exists(point_prompt_path):
        b.point_prompt_xyz = torch.load(point_prompt_path, map_location="cpu").to(device)
    else:
        npy_path = point_prompt_path.replace(".pt", ".npy")
        if os.path.exists(npy_path):
            xyz_np = np.load(npy_path)
            b.point_prompt_xyz = torch.from_numpy(xyz_np).to(device).float()
    logger.info(f"Loaded Prompt Bundle from {prompt_ckpt}")
    return b
def install_pfgr(model: nn.Module, device: torch.device):
    point_encoder = model.point_encoder
    target_layer = None
    target_name = None
    for name in ['reduce_dim', 'fc', 'proj', 'head']:
        if hasattr(point_encoder, name):
            layer = getattr(point_encoder, name)
            if isinstance(layer, nn.Linear):
                target_layer, target_name = layer, name
                break
    if target_layer is None:
        for name, module in point_encoder.named_modules():
            if isinstance(module, nn.Linear) and 'reduce' in name.lower():
                target_layer, target_name = module, name
                break
    if target_layer is None:
        raise RuntimeError("Could not find linear layer for PFGR installation")
    pfgr = PFGR_Module(target_layer, lambda_scale=0.0).to(device)
    _set_submodule(point_encoder, target_name, pfgr)
    logger.info(f"PFGR installed at {target_name}")
    return pfgr
def toggle_point_prompt(model, pfgr_module, post_fps_prompt, enable: bool):
    if enable:
        pfgr_module.lambda_scale = 1.0
        if post_fps_prompt is not None and post_fps_prompt.hook_handle is None:
            post_fps_prompt.attach(model.point_encoder)
    else:
        pfgr_module.lambda_scale = 0.0
        if post_fps_prompt is not None:
            post_fps_prompt.remove()
def encode_text_with_prompt(clip_model, prompt_ctx, tokenized_prompts, ctx_positions):
    transformer = clip_model.transformer
    cast_dtype = transformer.get_cast_dtype() if hasattr(transformer, "get_cast_dtype") else prompt_ctx.dtype
    if prompt_ctx.dim() == 2:
        prompt_ctx = prompt_ctx.unsqueeze(0).expand(tokenized_prompts.shape[0], -1, -1)
    prompt_ctx = prompt_ctx.to(device=tokenized_prompts.device, dtype=cast_dtype)
    with torch.no_grad():
        x = clip_model.token_embedding(tokenized_prompts).to(device=tokenized_prompts.device, dtype=cast_dtype)
    x[:, ctx_positions, :] = prompt_ctx
    seq_len = x.shape[1]
    positional = clip_model.positional_embedding[:seq_len].to(device=tokenized_prompts.device, dtype=cast_dtype)
    x = x + positional
    attn_mask = getattr(clip_model, "attn_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask[:seq_len, :seq_len].to(device=tokenized_prompts.device, dtype=cast_dtype)
    x = transformer(x, attn_mask=attn_mask)
    x = clip_model.ln_final(x)
    pooled = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
    proj = getattr(clip_model, "text_projection", None)
    if proj is not None:
        pooled = proj(pooled) if isinstance(proj, nn.Linear) else pooled @ proj
    return pooled
def build_std_text_features(clip_model, tokenizer, classnames, device, batch_size=4):
    all_feats = []
    logger.info(f"Building std text features in batches (size={batch_size})...")
    for i in range(0, len(classnames), batch_size):
        batch_names = classnames[i : i + batch_size]
        prompts = [f"X X X A depth picture of {name}." for name in batch_names]
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            feats = clip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
        torch.cuda.empty_cache()
    return torch.cat(all_feats, dim=0).to(device)
def build_prompted_text_features(clip_model, tokenizer, classnames, bundle, device, batch_size=4):
    soft_prefix = " ".join(["X"] * bundle.n_soft)
    hard_prefix = " ".join(["X"] * bundle.n_hard)
    if bundle.ctx_tokens is not None:
        hard_emb = clip_model.token_embedding.weight.index_select(0, bundle.ctx_tokens).float().to(device)
        full_ctx = torch.cat([bundle.ctx_soft, hard_emb], dim=0)
    elif bundle.ctx_cont is not None:
        full_ctx = torch.cat([bundle.ctx_soft, bundle.ctx_cont], dim=0)
    else:
        raise ValueError("No hard context available")
    all_feats = []
    logger.info(f"Building prompted text features in batches (size={batch_size})...")
    for i in range(0, len(classnames), batch_size):
        batch_names = classnames[i : i + batch_size]
        prompts = [f"{soft_prefix} A depth picture of a {hard_prefix} {name}." for name in batch_names]
        tokenized_list = tokenizer(prompts)
        if isinstance(tokenized_list, list):
             tokenized = torch.stack(tokenized_list).to(device)
        else:
             tokenized = tokenized_list.to(device)
        with torch.no_grad():
            feats = encode_text_with_prompt(clip_model, full_ctx, tokenized, ctx_positions=bundle.ctx_positions)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
        torch.cuda.empty_cache()
    return torch.cat(all_feats, dim=0).to(device)
class NpzPointCloudDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        if "points" in data:
            pts = data["points"]
        elif "test_pc" in data:
            pts = data["test_pc"]
        else:
            pts = data[list(data.keys())[0]]
        if "labels" in data:
            lbs = data["labels"]
        elif "test_label" in data:
            lbs = data["test_label"]
        else:
            keys = list(data.keys())
            cand = [k for k in keys if "label" in k.lower()]
            if len(cand) > 0:
                lbs = data[cand[0]]
            else:
                lbs = data[keys[1]]
        pts = pts.astype(np.float32)
        lbs = lbs.astype(np.int64)
        if pts.shape[-1] == 3:
            rgb = np.ones((pts.shape[0], pts.shape[1], 3), dtype=np.float32) * 0.5
            pts = np.concatenate([pts, rgb], axis=-1)
        self.points = torch.from_numpy(pts).float()
        self.labels = torch.from_numpy(lbs).long()
    def __len__(self):
        return int(self.points.shape[0])
    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]
class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split="modelnet40_test"):
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.classnames = []
        classnames_path = os.path.join(root_dir, 'XXX')
        with open(classnames_path, "r") as f:
            self.classnames = [line.strip() for line in f.readlines()]
        split_file = os.path.join(root_dir, f"{split}.txt")
        with open(split_file, "r") as f:
            file_list = [line.strip() for line in f.readlines()]
        for item in file_list:
            category = item.rsplit("_", 1)[0]
            file_path = os.path.join(root_dir, category, f"{item}.txt")
            if os.path.exists(file_path):
                self.data.append(file_path)
                self.labels.append(self.classnames.index(category))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        pc_path = self.data[idx]
        label = self.labels[idx]
        points = np.loadtxt(pc_path, delimiter=",").astype(np.float32)[:, 0:3]
        colors = np.full(points.shape, 0.5, dtype=np.float32)
        points = np.hstack((points, colors))
        return points, label
class UntargetedLogitsAdvLoss(nn.Module):
    def __init__(self, kappa=0.):
        super().__init__()
        self.kappa = kappa
    def forward(self, logits, targets):
        B, K = logits.shape
        if targets.ndim == 1:
            targets = targets.view(-1, 1)
        targets = targets.long()
        one_hot = torch.zeros(B, K, device=logits.device).scatter_(1, targets, 1.0)
        real = torch.sum(one_hot * logits, dim=1)
        other = torch.max((1. - one_hot) * logits - one_hot * 10000., dim=1)[0]
        loss = torch.clamp(real - other + self.kappa, min=0.)
        return loss.mean()
class L2Dist(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, adv_pc, ori_pc, weights=None, batch_avg=True):
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,), device=adv_pc.device)
        if adv_pc.shape[-1] == 6:
            adv_pc = adv_pc[..., :3]
        if ori_pc.shape[-1] == 6:
            ori_pc = ori_pc[..., :3]
        dist = torch.sqrt(torch.sum((adv_pc - ori_pc) ** 2, dim=[1, 2]))
        dist = dist * weights
        return dist.mean() if batch_avg else dist
class ClipPointsLinf(nn.Module):
    def __init__(self, budget):
        super().__init__()
        self.budget = budget
    def forward(self, pc, ori_pc):
        with torch.no_grad():
            xyz = pc[..., :3]
            ori_xyz = ori_pc[..., :3]
            diff = xyz - ori_xyz
            norm = torch.sum(diff ** 2, dim=1) ** 0.5
            scale = self.budget / (norm + 1e-9)
            scale = torch.clamp(scale, max=1.)
            diff = diff * scale[:, None, :]
            new_xyz = ori_xyz + diff
            if pc.shape[-1] == 6:
                pc = torch.cat([new_xyz, pc[..., 3:]], dim=-1)
            else:
                pc = new_xyz
        return pc
def load_clip_text_model():
    logger.info("Loading CLIP Text Encoder...")
    try:
        model, _, _ = open_clip.create_model_and_transforms_text("ViT-bigG-14", pretrained=None)
    except Exception:
        model, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained=None)
    try:
        from safetensors.torch import load_file
        sd = load_file(CONF["clip_model_path"])
        model.load_state_dict(sd, strict=False)
    except Exception as e:
        logger.warning(f"Failed to load CLIP safetensors: {e}")
    return model
def load_student_model(args, clip_model_text):
    logger.info("Loading ULIP-2 Student model...")
    config = cfg_from_yaml_file(CONF["pointbert_config"])
    point_encoder = PointTransformer_Colored(config.model, args=args)
    model = ULIP2_WITH_OPENCLIP(
        open_clip_model=clip_model_text,
        point_encoder=point_encoder,
        pc_feat_dims=768,
        embed_dim=args.embed_dim
    )
    ckpt = CONF["student_ckpt"]
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Student weights not found: {ckpt}")
    checkpoint = torch.load(ckpt, map_location="cpu")
    sd = checkpoint.get("state_dict", checkpoint)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model_state = model.state_dict()
    filtered = {}
    for k, v in sd.items():
        if "pc_projection" in k:
             if k in model_state:
                 if v.shape == model_state[k].shape:
                     filtered[k] = v
                 elif v.ndim == 2 and v.t().shape == model_state[k].shape:
                     filtered[k] = v.t()
             else:
                 base = "pc_projection"
                 if base in model_state:
                     if v.shape == model_state[base].shape:
                         filtered[base] = v
                     elif v.ndim==2 and v.t().shape == model_state[base].shape:
                         filtered[base] = v.t()
        elif k in model_state:
             if v.shape == model_state[k].shape:
                 filtered[k] = v
    model.load_state_dict(filtered, strict=False)
    logger.info("Student model weights loaded.")
    return model
def eval_accuracy(model, loader, text_features, device=DEVICE):
    model.eval()
    correct, total = 0, 0
    text_features = text_features.to(device)
    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)
            if points.shape[1] > 1024:
                points = fps6d(points, 1024)
            point_features = model.encode_pc(points)
            point_features = F.normalize(point_features, dim=-1)
            logits = torch.matmul(point_features, text_features.t())
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += int(labels.numel())
    return correct / max(total, 1)
def run_ablation(title, loader, model, std_text_features, prompted_text_features, pfgr_module, post_fps_prompt):
    print(f"\n==================== {title} ====================")
    toggle_point_prompt(model, pfgr_module, post_fps_prompt, False)
    acc_clean = eval_accuracy(model, loader, std_text_features, DEVICE)
    acc_text = eval_accuracy(model, loader, prompted_text_features, DEVICE)
    toggle_point_prompt(model, pfgr_module, post_fps_prompt, True)
    acc_point = eval_accuracy(model, loader, std_text_features, DEVICE)
    acc_both = eval_accuracy(model, loader, prompted_text_features, DEVICE)
    toggle_point_prompt(model, pfgr_module, post_fps_prompt, False)
    print(f"acc_clean      = {acc_clean:.4f}")
    print(f"acc_text       = {acc_text:.4f}")
    print(f"acc_point      = {acc_point:.4f}")
    print(f"acc_text+point = {acc_both:.4f}")
def generate_whitebox_aof(model, text_features):
    save_path = WHITEBOX_PATH
    if os.path.exists(save_path):
        logger.info(f"AOF whitebox data already exists: {save_path}")
        return save_path
    logger.info("Start generating AOF whitebox data...")
    dataset = ModelNet40Dataset(root_dir=CONF["modelnet_root"], split="modelnet40_test")
    loader = DataLoader(dataset, batch_size=CONF["batch_size_gen"], shuffle=False, num_workers=CONF["num_workers"])
    clip_func = ClipPointsLinf(budget=0.18)
    adv_func = UntargetedLogitsAdvLoss(kappa=30.)
    dist_func = L2Dist()
    attacker = CWAOF(
        model, adv_func, dist_func,
        attack_lr=1e-2,
        binary_step=2,
        num_iter=200,
        GAMMA=0.25,
        low_pass=100,
        clip_func=clip_func,
        text_features=text_features
    )
    all_adv, all_lbl = [], []
    model.eval()
    for i, (points, labels) in enumerate(tqdm(loader)):
        points = points.to(DEVICE).float()
        labels = labels.to(DEVICE).long()
        if points.shape[1] > 1024:
            points = fps6d(points, 1024)
        _, adv_pc_numpy, _ = attacker.attack(points, labels)
        all_adv.append(adv_pc_numpy)
        all_lbl.append(labels.detach().cpu().numpy())
        torch.cuda.empty_cache()
        gc.collect()
    all_adv = np.concatenate(all_adv, axis=0)
    all_lbl = np.concatenate(all_lbl, axis=0)
    np.savez(save_path, points=all_adv, labels=all_lbl)
    logger.info(f"Saved: {save_path}")
    return save_path
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=CONF["embed_dim"])
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--evaluate_3d", action="store_true", default=True)
    args = parser.parse_args()
    clip_model_text = load_clip_text_model().to(DEVICE).eval()
    student_model = load_student_model(args, clip_model_text).to(DEVICE).eval()
    helper_ds = ModelNet40Dataset(root_dir=CONF["modelnet_root"], split="modelnet40_test")
    classnames = [name.replace("_", " ") for name in helper_ds.classnames]
    logger.info(f"Loading Best Prompt: {CONF['best_prompt_path']}")
    bundle = load_prompt_bundle(CONF["best_prompt_path"], clip_model_text, tokenizer, classnames, DEVICE)
    pfgr_module = install_pfgr(student_model, DEVICE)
    if bundle.pfgr_weight is not None:
        pfgr_module.pfgr_branch.load_state_dict(bundle.pfgr_weight, strict=False)
        logger.info("PFGR Weights loaded.")
    post_fps_prompt = None
    if bundle.point_prompt_xyz is not None:
        post_fps_prompt = PostFPSPointPrompt(bundle.point_prompt_xyz).to(DEVICE)
        logger.info("Post-FPS Prompt created.")
    torch.cuda.empty_cache()
    gc.collect()
    std_text_features = build_std_text_features(clip_model_text, tokenizer, classnames, DEVICE, batch_size=2)
    prompted_text_features = build_prompted_text_features(clip_model_text, tokenizer, classnames, bundle, DEVICE, batch_size=2)
    torch.cuda.empty_cache()
    toggle_point_prompt(student_model, pfgr_module, post_fps_prompt, enable=False)
    if os.path.exists(CONF["clean_npz"]):
        ds_clean = NpzPointCloudDataset(CONF["clean_npz"])
        loader_clean = DataLoader(ds_clean, batch_size=CONF["batch_size_eval"], shuffle=False, num_workers=CONF["num_workers"])
        run_ablation("clean", loader_clean, student_model, std_text_features, prompted_text_features, pfgr_module, post_fps_prompt)
    else:
        logger.warning(f"clean npz does not exist: {CONF['clean_npz']}")
    whitebox_path = generate_whitebox_aof(student_model, std_text_features)
    ds_white = NpzPointCloudDataset(whitebox_path)
    loader_white = DataLoader(ds_white, batch_size=CONF["batch_size_eval"], shuffle=False, num_workers=CONF["num_workers"])
    run_ablation("AOF whitebox", loader_white, student_model, std_text_features, prompted_text_features, pfgr_module, post_fps_prompt)
    if os.path.exists(CONF["blackbox_path"]):
        ds_black = NpzPointCloudDataset(CONF["blackbox_path"])
        loader_black = DataLoader(ds_black, batch_size=CONF["batch_size_eval"], shuffle=False, num_workers=CONF["num_workers"])
        run_ablation("AOF blackbox", loader_black, student_model, std_text_features, prompted_text_features, pfgr_module, post_fps_prompt)
    else:
        logger.warning(f"blackbox data does not exist: {CONF['blackbox_path']}")
    logger.info("All tests completed.")
if __name__ == "__main__":
    main()
