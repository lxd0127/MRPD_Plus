import os
import sys
from pathlib import Path
import argparse
import random
from typing import Optional, List
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from loguru import logger
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT_DIR = "XXX"
DATA_ROOT = os.path.join(ROOT_DIR, 'XXX')
SAVE_DIR_WHITEBOX = os.path.join(ROOT_DIR, 'XXX')
WHITEBOX_PATH = os.path.join(SAVE_DIR_WHITEBOX, 'XXX')
CLEAN_NPZ_PATH = os.path.join(SAVE_DIR_WHITEBOX, 'XXX')
BLACKBOX_DATA_PATH = os.path.join(
    ROOT_DIR,
    "XXX",
)
CLIP_MODEL_PATH = os.path.join(
    ROOT_DIR,
    'XXX',
)
ULIP2_CKPT_PATH = os.path.join(
    ROOT_DIR,
    'XXX',
)
POINTBERT_CFG = os.path.join(
    ROOT_DIR,
    'XXX',
)
PROMPT_CKPT = "XXX"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from utils.tokenizer import SimpleTokenizer
import utils.open_clip as open_clip
from utils.utils import cfg_from_yaml_file
from models.ULIP_models import ULIP2_WITH_OPENCLIP
from models.pointbert.point_encoder import PointTransformer_Colored
from models.point_encoder import fps6d
try:
    from CWPerturb import CWPerturb
except Exception:
    CWPerturb = None
NUM_POINTS = 1024
NUM_CLASSES = 40
BATCH_SIZE_EVAL = 32
BATCH_SIZE_GEN_WHITEBOX = 16
SEED = 0
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def ensure_6ch(points: torch.Tensor) -> torch.Tensor:
    if points.shape[-1] == 3:
        colors = torch.full((points.shape[0], points.shape[1], 3), 0.5, device=points.device, dtype=points.dtype)
        points = torch.cat([points, colors], dim=-1)
    return points
def ensure_1024(points: torch.Tensor) -> torch.Tensor:
    if points.shape[1] != NUM_POINTS:
        points = fps6d(points, NUM_POINTS)
    return points
class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split="modelnet40_test", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.classnames = []
        classnames_path = os.path.join(root_dir, 'XXX')
        if not os.path.exists(classnames_path):
            raise FileNotFoundError(f"Missing classnames: {classnames_path}")
        with open(classnames_path, "r") as f:
            self.classnames = [line.strip() for line in f.readlines()]
        split_file = os.path.join(root_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")
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
        if self.transform:
            points = self.transform(points)
        return points, label
class NpzPointCloudDataset(Dataset):
    def __init__(self, npz_path: str):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing npz: {npz_path}")
        data = np.load(npz_path)
        if "points" in data: pts = data["points"]
        elif "test_pc" in data: pts = data["test_pc"]
        else: pts = data[list(data.keys())[0]]
        if "labels" in data: lbs = data["labels"]
        elif "test_label" in data: lbs = data["test_label"]
        else: lbs = data[list(data.keys())[1]]
        self.points = torch.tensor(pts, dtype=torch.float32)
        self.labels = torch.tensor(lbs, dtype=torch.long)
        if self.points.ndim == 3 and self.points.shape[-1] == 3:
            colors = torch.full((self.points.shape[0], self.points.shape[1], 3), 0.5, dtype=torch.float32)
            self.points = torch.cat([self.points, colors], dim=-1)
    def __len__(self):
        return int(self.points.shape[0])
    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]
transform = transforms.Compose([
    lambda pc: np.hstack((pc[:, :3], np.full((pc.shape[0], 3), 0.5, dtype=np.float32))),
    lambda pc: torch.tensor(pc, dtype=torch.float32),
])
def build_ulip2_student(open_clip_model: nn.Module) -> nn.Module:
    config = cfg_from_yaml_file(POINTBERT_CFG)
    dummy_args = argparse.Namespace(embed_dim=1280, num_group=512, group_size=64)
    point_encoder = PointTransformer_Colored(config.model, args=dummy_args)
    model = ULIP2_WITH_OPENCLIP(
        open_clip_model=open_clip_model,
        point_encoder=point_encoder,
        pc_feat_dims=768,
        embed_dim=1280,
    )
    if not os.path.exists(ULIP2_CKPT_PATH):
        raise FileNotFoundError(f"Missing ULIP-2 weights: {ULIP2_CKPT_PATH}")
    checkpoint = torch.load(ULIP2_CKPT_PATH, map_location="cpu")
    sd = checkpoint.get("state_dict", checkpoint)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model_state = model.state_dict()
    filtered_sd = {}
    for k, v in sd.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_sd[k] = v
            elif "pc_projection" in k and tuple(reversed(v.shape)) == model_state[k].shape:
                filtered_sd[k] = v.t()
            elif "first_conv" in k and v.ndim >= 2 and v.shape[1] == 6 and model_state[k].shape[1] == 3:
                filtered_sd[k] = v[:, :3, ...]
    msg = model.load_state_dict(filtered_sd, strict=False)
    logger.info(f"ULIP-2 Student loaded. {msg}")
    return model
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
def load_new_best_prompt(prompt_ckpt: str, clip_model: nn.Module, tokenizer: SimpleTokenizer, classnames: List[str], device: torch.device) -> PromptBundle:
    if not os.path.exists(prompt_ckpt):
        raise FileNotFoundError(f"Missing best prompt file: {prompt_ckpt}")
    prefix = prompt_ckpt.replace(".pt", "")
    ctx_soft_path = prefix + "_ctx_soft.pt"
    ctx_cont_path = prefix + "_ctx_cont.pt"
    ctx_tokens_path = prefix + "_ctx_tokens.pt"
    point_prompt_path = prefix + "_point_prompt.pt"
    b = PromptBundle()
    b.ctx_soft = torch.load(ctx_soft_path, map_location="cpu").to(device).float()
    b.ctx_cont = torch.load(ctx_cont_path, map_location="cpu").to(device).float()
    ctx_tokens = torch.load(ctx_tokens_path, map_location="cpu")
    if not isinstance(ctx_tokens, torch.Tensor):
        ctx_tokens = torch.tensor(ctx_tokens, dtype=torch.long)
    b.ctx_tokens = ctx_tokens.to(device).long()
    b.n_soft = int(b.ctx_soft.shape[0])
    b.n_hard = int(b.ctx_cont.shape[0])
    soft_prefix = " ".join(["X"] * b.n_soft)
    hard_prefix = " ".join(["X"] * b.n_hard)
    prompts = [f"{soft_prefix} A depth picture of a {hard_prefix} {name}." for name in classnames]
    tokenized = tokenize_texts(tokenizer, prompts).to(device)
    def _tokenize_single(text):
        ids = tokenizer(text)
        return ids[0] if isinstance(ids, torch.Tensor) and ids.ndim==2 else ids
    def _get_single_token_id(word, add_space):
        text = (" " + word) if add_space else word
        ids = _tokenize_single(text)
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
def tokenize_texts(tokenizer, texts):
    if isinstance(texts, str): texts = [texts]
    out = tokenizer(texts)
    return out if isinstance(out, torch.Tensor) else torch.stack(out, dim=0)
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
    hard_emb = clip_model.token_embedding.weight.index_select(0, bundle.ctx_tokens).float().to(device)
    full_ctx = torch.cat([bundle.ctx_soft, hard_emb], dim=0)
    all_feats = []
    logger.info(f"Building prompted text features in batches (size={batch_size})...")
    for i in range(0, len(classnames), batch_size):
        batch_names = classnames[i : i + batch_size]
        prompts = [f"{soft_prefix} A depth picture of a {hard_prefix} {name}." for name in batch_names]
        tokenized = tokenize_texts(tokenizer, prompts).to(device)
        with torch.no_grad():
            feats = encode_text_with_prompt(clip_model, full_ctx, tokenized, ctx_positions=bundle.ctx_positions)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
        torch.cuda.empty_cache()
    return torch.cat(all_feats, dim=0).to(device)
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
@torch.no_grad()
def eval_accuracy(model, loader, text_features, device) -> float:
    model.eval()
    correct, total = 0, 0
    for points, labels in loader:
        points = points.to(device)
        labels = labels.to(device)
        points = ensure_6ch(points)
        points = ensure_1024(points)
        pc_feat = model.encode_pc(points)
        pc_feat = F.normalize(pc_feat, dim=-1)
        logits = pc_feat @ text_features.t()
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += len(labels)
    return float(correct) / max(1, total)
def random_target(labels, num_classes):
    targets = torch.randint_like(labels, 0, num_classes)
    targets = torch.where(targets==labels, (targets+1)%num_classes, targets)
    return targets
class CrossEntropyAdvLoss(nn.Module):
    def forward(self, logits, targets): return F.cross_entropy(logits, targets)
class L2Dist(nn.Module):
    def forward(self, adv, ori, weights=None, batch_avg=True):
        if weights is None: weights = torch.ones(adv.shape[0], device=adv.device)
        dist = torch.sqrt(torch.sum((adv-ori)**2, dim=[1,2])) * weights
        return dist.mean() if batch_avg else dist
def generate_whitebox_if_needed(model, raw_dataset, text_features, save_path, device):
    if os.path.exists(save_path): return
    if CWPerturb is None: raise ImportError("CWPerturb required for whitebox gen")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(raw_dataset, batch_size=BATCH_SIZE_GEN_WHITEBOX, shuffle=False, num_workers=4)
    model.eval()
    attacker = CWPerturb(
        model, CrossEntropyAdvLoss(), L2Dist(),
        attack_lr=1e-2, init_weight=10.0, max_weight=200.0, binary_step=10, num_iter=200,
        text_features=text_features
    )
    all_adv, all_lbl = [], []
    logger.info("Generating Whitebox Data...")
    for i, (points, labels) in enumerate(loader):
        points = points.to(device)
        labels = labels.to(device)
        points = ensure_6ch(points)
        points = fps6d(points, NUM_POINTS)
        target = random_target(labels, NUM_CLASSES)
        _, adv, _ = attacker.attack(points, target)
        all_adv.append(torch.from_numpy(adv).cpu())
        all_lbl.append(labels.cpu())
        if i % 10 == 0: print(f"Batch {i}/{len(loader)}")
    np.savez(save_path, points=torch.cat(all_adv).numpy(), labels=torch.cat(all_lbl).numpy())
    logger.info(f"Saved whitebox: {save_path}")
def build_clean_npz_if_needed(raw_dataset, device):
    if os.path.exists(CLEAN_NPZ_PATH): return
    Path(SAVE_DIR_WHITEBOX).mkdir(parents=True, exist_ok=True)
    loader = DataLoader(raw_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=4)
    all_pts, all_lbl = [], []
    for pts, lbl in loader:
        pts = ensure_6ch(pts.to(device))
        pts = fps6d(pts, NUM_POINTS).cpu()
        all_pts.append(pts)
        all_lbl.append(lbl)
    np.savez(CLEAN_NPZ_PATH, points=torch.cat(all_pts).numpy(), labels=torch.cat(all_lbl).numpy())
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_whitebox_gen", action="store_true")
    args = parser.parse_args()
    set_seed(SEED)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Loading CLIP...")
    from safetensors.torch import load_file
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained=None)
    clip_model.load_state_dict(load_file(CLIP_MODEL_PATH), strict=False)
    clip_model.to(DEVICE).eval()
    tokenizer = SimpleTokenizer()
    temp_dataset = ModelNet40Dataset(DATA_ROOT, split="modelnet40_test")
    classnames = [name.replace("_", " ") for name in temp_dataset.classnames]
    logger.info("Loading ULIP-2 Student...")
    model = build_ulip2_student(clip_model).to(DEVICE).eval()
    logger.info(f"Loading Prompts from: {PROMPT_CKPT}")
    bundle = load_new_best_prompt(PROMPT_CKPT, clip_model, tokenizer, classnames, DEVICE)
    pfgr_module = install_pfgr(model, DEVICE)
    if bundle.pfgr_weight is not None:
        msg = pfgr_module.pfgr_branch.load_state_dict(bundle.pfgr_weight, strict=False)
        logger.info(f"PFGR Weights loaded: {msg}")
    post_fps_prompt = None
    if bundle.point_prompt_xyz is not None:
        post_fps_prompt = PostFPSPointPrompt(bundle.point_prompt_xyz).to(DEVICE)
        logger.info("Post-FPS Prompt object created.")
    torch.cuda.empty_cache()
    gc.collect()
    std_text_features = build_std_text_features(clip_model, tokenizer, classnames, DEVICE, batch_size=2)
    prompted_text_features = build_prompted_text_features(clip_model, tokenizer, classnames, bundle, DEVICE, batch_size=2)
    torch.cuda.empty_cache()
    gc.collect()
    raw_test = ModelNet40Dataset(DATA_ROOT, split="modelnet40_test", transform=transform)
    build_clean_npz_if_needed(raw_test, DEVICE)
    toggle_point_prompt(model, pfgr_module, post_fps_prompt, enable=False)
    if not args.skip_whitebox_gen:
        try:
            generate_whitebox_if_needed(model, raw_test, std_text_features, WHITEBOX_PATH, DEVICE)
        except Exception as e:
            logger.warning(f"Whitebox gen failed: {e}")
    datasets = [("Clean", CLEAN_NPZ_PATH)]
    if os.path.exists(WHITEBOX_PATH): datasets.append(("Whitebox", WHITEBOX_PATH))
    if os.path.exists(BLACKBOX_DATA_PATH): datasets.append(("Blackbox", BLACKBOX_DATA_PATH))
    for name, path in datasets:
        logger.info(f"===== Eval Dataset: {name} =====")
        ds = NpzPointCloudDataset(path)
        loader = DataLoader(ds, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=4)
        toggle_point_prompt(model, pfgr_module, post_fps_prompt, enable=False)
        acc_clean = eval_accuracy(model, loader, std_text_features, DEVICE)
        acc_text = eval_accuracy(model, loader, prompted_text_features, DEVICE)
        toggle_point_prompt(model, pfgr_module, post_fps_prompt, enable=True)
        acc_point = eval_accuracy(model, loader, std_text_features, DEVICE)
        acc_both = eval_accuracy(model, loader, prompted_text_features, DEVICE)
        toggle_point_prompt(model, pfgr_module, post_fps_prompt, enable=False)
        print(f"[{name}] Clean Acc:        {acc_clean:.4f}")
        print(f"[{name}] Text Prompt Acc:  {acc_text:.4f}")
        print(f"[{name}] Point Prompt Acc: {acc_point:.4f}")
        print(f"[{name}] Text + Point Acc: {acc_both:.4f}")
        print("-" * 30)
    logger.info("All tests completed.")
if __name__ == "__main__":
    main()
