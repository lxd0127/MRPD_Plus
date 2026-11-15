import os
import warnings
import json
import numpy as np
from pathlib import Path
import copy
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import utils.open_clip as open_clip
from safetensors.torch import load_file
from models.ULIP_models import ULIP2_WITH_OPENCLIP
from models.pointbert.point_encoder import PointTransformer_Colored
from utils.utils import cfg_from_yaml_file
from utils.tokenizer import SimpleTokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")
MRPD_ROOT = "XXX"
DATA_ROOT = os.path.join(MRPD_ROOT, 'XXX')
MODELNET_DIR = os.path.join(DATA_ROOT, 'XXX')
WHITEBOX_DIR = os.path.join(MRPD_ROOT, 'XXX')
CLEAN_NPZ_PATH = os.path.join(DATA_ROOT, 'XXX')
WHITEBOX_NPZ_PATH = os.path.join(WHITEBOX_DIR, 'XXX')
BLACKBOX_NPZ_PATH = os.path.join(MRPD_ROOT, "XXX")
CLIP_SAFETENSORS = os.path.join(
    MRPD_ROOT,
    'XXX'
)
PROMPT_DIR = "XXX"
PROMPT_FILE_NAME = 'XXX'
PROMPT_CKPT_BEST = os.path.join(PROMPT_DIR, PROMPT_FILE_NAME)
DEVICE = "cuda:0"
BATCH_SIZE = 16
NUM_WORKERS = 4
EMBED_DIM = 1280
class PFGR_Module(nn.Module):
    def __init__(self, original_layer, lambda_scale=1.0):
        super().__init__()
        self.original_layer = original_layer
        self.lambda_scale = lambda_scale
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        self.pfgr_branch = nn.Linear(in_features, out_features, bias=False)
    def forward(self, x):
        return self.original_layer(x) + self.lambda_scale * self.pfgr_branch(x)
class PostFPSPointPrompt(nn.Module):
    def __init__(self, prompt_xyz):
        super().__init__()
        self.register_buffer("prompt_xyz", prompt_xyz)
    def forward_hook(self, module, inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            return output
        neighborhood = output[0]
        center = output[1]
        if not (torch.is_tensor(neighborhood) and torch.is_tensor(center)):
            return output
        B = center.shape[0]
        prompt_xyz_expand = self.prompt_xyz.to(center.device, center.dtype).expand(B, -1, -1)
        P = prompt_xyz_expand.shape[1]
        S = neighborhood.shape[2]
        C = neighborhood.shape[3]
        prompt_neighborhood = torch.zeros((B, P, S, C), device=neighborhood.device, dtype=neighborhood.dtype)
        neighborhood_new = torch.cat([neighborhood, prompt_neighborhood], dim=1)
        center_new = torch.cat([center, prompt_xyz_expand], dim=1)
        new_out = list(output)
        new_out[0] = neighborhood_new
        new_out[1] = center_new
        return tuple(new_out)
def _set_submodule(root: nn.Module, path: str, new_module: nn.Module):
    parts = path.split('.')
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)
class PointPromptApplicator:
    def __init__(self, model, pfgr_state_dict, prompt_xyz_tensor):
        self.model = model
        self.pfgr_state_dict = pfgr_state_dict
        self.prompt_xyz_tensor = prompt_xyz_tensor
        self.hook_handle = None
        self.pfgr_target_name = None
        self.original_layer_ref = None
        self.pfgr_module = None
        self.hook_module = None
    def apply(self):
        point_encoder = self.model.point_encoder
        target_layer = None
        target_name = None
        target_layer_names = ['reduce_dim', 'fc', 'proj', 'head']
        for name in target_layer_names:
            if hasattr(point_encoder, name):
                layer = getattr(point_encoder, name)
                if isinstance(layer, nn.Linear):
                    target_layer = layer
                    target_name = name
                    break
        if target_layer is None:
            for name, module in point_encoder.named_modules():
                if isinstance(module, nn.Linear) and 'reduce' in name.lower():
                    target_layer = module
                    target_name = name
                    break
        if target_layer and self.pfgr_state_dict is not None:
            self.pfgr_target_name = target_name
            self.original_layer_ref = target_layer
            pfgr = PFGR_Module(target_layer, lambda_scale=1.0).to(DEVICE)
            pfgr.pfgr_branch.load_state_dict(self.pfgr_state_dict)
            self.pfgr_module = pfgr
            _set_submodule(point_encoder, target_name, pfgr)
        else:
            print("[Applicator] Warning: Could not find linear layer for PFGR or no state dict provided.")
        if self.prompt_xyz_tensor is not None:
            group_module = None
            if hasattr(point_encoder, 'group_divider'):
                group_module = getattr(point_encoder, 'group_divider')
            else:
                for name, m in point_encoder.named_modules():
                    if ('group_divider' in name.lower()) or (m.__class__.__name__.lower() == 'group'):
                        group_module = m
                        break
            if group_module:
                self.hook_module = PostFPSPointPrompt(self.prompt_xyz_tensor.to(DEVICE))
                self.hook_handle = group_module.register_forward_hook(self.hook_module.forward_hook)
            else:
                print("[Applicator] Warning: Could not find group_divider for Post-FPS prompt.")
    def remove(self):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
        if self.pfgr_target_name and self.original_layer_ref:
            point_encoder = self.model.point_encoder
            _set_submodule(point_encoder, self.pfgr_target_name, self.original_layer_ref)
            self.pfgr_module = None
class NpzPointCloudDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        if 'points' in data: self.points = torch.tensor(data['points'], dtype=torch.float32)
        elif 'test_pc' in data: self.points = torch.tensor(data['test_pc'], dtype=torch.float32)
        else: self.points = torch.tensor(data[list(data.keys())[0]], dtype=torch.float32)
        if 'labels' in data: self.labels = torch.tensor(data['labels'], dtype=torch.long)
        elif 'test_label' in data: self.labels = torch.tensor(data['test_label'], dtype=torch.long)
        else: self.labels = torch.tensor(data[list(data.keys())[1]], dtype=torch.long)
        if self.points.shape[-1] == 3:
            colors = torch.full(self.points.shape, 0.5)
            self.points = torch.cat([self.points, colors], dim=-1)
    def __len__(self): return int(self.points.shape[0])
    def __getitem__(self, idx): return self.points[idx], self.labels[idx]
def fps_downsample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    new_xyz = torch.gather(xyz, 1, centroids.unsqueeze(-1).expand(-1, -1, 3))
    return new_xyz
class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split="test", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data, self.labels, self.classnames = [], [], []
        classnames_path = os.path.join(root_dir, 'XXX')
        with open(classnames_path, "r") as f: self.classnames = [line.strip() for line in f.readlines()]
        split_file = os.path.join(root_dir, f"modelnet40_{split}.txt")
        if not os.path.exists(split_file): split_file = os.path.join(root_dir, f"{split}.txt")
        with open(split_file, "r") as f: file_list = [line.strip() for line in f.readlines()]
        for item in file_list:
            category = item.rsplit("_", 1)[0]
            file_path = os.path.join(root_dir, category, f"{item}.txt")
            if os.path.exists(file_path):
                self.data.append(file_path)
                self.labels.append(self.classnames.index(category))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        points = np.loadtxt(self.data[idx], delimiter=",").astype(np.float32)[:, 0:3]
        if self.transform: points = self.transform(points)
        return points, self.labels[idx]
def ensure_clean_1024_npz(clean_npz_path: str):
    alt_path = os.path.join(WHITEBOX_DIR, 'XXX')
    if os.path.exists(alt_path): return alt_path
    if os.path.exists(clean_npz_path): return clean_npz_path
    print(f"Generating {clean_npz_path}...")
    transform = transforms.Compose([lambda pc: torch.tensor(pc, dtype=torch.float32)])
    ds = ModelNet40Dataset(root_dir=MODELNET_DIR, split="test", transform=transform)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    all_pts, all_lbls = [], []
    for pts, lbl in tqdm(dl, desc="Generating clean 1024"):
        pts = pts.cuda(non_blocking=True)
        pts_1024 = fps_downsample(pts, 1024).cpu().numpy()
        all_pts.append(pts_1024)
        all_lbls.append(np.asarray(lbl))
    np.savez(clean_npz_path, points=np.concatenate(all_pts), labels=np.concatenate(all_lbls))
    return clean_npz_path
class DummyArgs:
    def __init__(self):
        self.pc_model = "eva02_base_patch14_448"
        self.pc_feat_dim = 768
        self.pretrained_pc = None
        self.drop_path_rate = 0.1
        self.pc_encoder_dim = 512
        self.embed_dim = EMBED_DIM
        self.num_group = 512
        self.group_size = 64
        self.evaluate_3d = True
def get_ulip2_pointbert_student(open_clip_model):
    config_addr = os.path.join(MRPD_ROOT, 'XXX')
    config = cfg_from_yaml_file(config_addr)
    args = DummyArgs()
    point_encoder = PointTransformer_Colored(config.model, args=args)
    model = ULIP2_WITH_OPENCLIP(
        open_clip_model=open_clip_model,
        point_encoder=point_encoder,
        pc_feat_dims=768,
        embed_dim=EMBED_DIM
    )
    ckpt_path = os.path.join(MRPD_ROOT, 'XXX')
    if not os.path.exists(ckpt_path): return model
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model_state = model.state_dict()
    filtered_sd = {}
    for k, v in sd.items():
        if k in model_state:
            if v.shape == model_state[k].shape: filtered_sd[k] = v
            elif v.shape[::-1] == model_state[k].shape: filtered_sd[k] = v.t()
    model.load_state_dict(filtered_sd, strict=False)
    return model
tokenizer = SimpleTokenizer()
def _tokenize_single(text: str) -> torch.Tensor:
    ids = tokenizer(text)
    if isinstance(ids, torch.Tensor) and ids.ndim == 2: ids = ids[0]
    return ids
def _get_single_token_id(word: str, add_space: bool = True):
    ids = _tokenize_single((" " + word) if add_space else word)
    eot_pos = int(ids.argmax().item())
    content = ids[1:eot_pos]
    content = content[content != 0]
    return int(content.item()) if int(content.numel()) == 1 else None
def encode_text_with_prompt(clip_model, prompt_ctx, tokenized_prompts, ctx_positions):
    transformer = clip_model.transformer
    cast_dtype = transformer.get_cast_dtype() if hasattr(transformer, "get_cast_dtype") else prompt_ctx.dtype
    if prompt_ctx.dim() == 2: prompt_ctx = prompt_ctx.unsqueeze(0).expand(tokenized_prompts.shape[0], -1, -1)
    prompt_ctx = prompt_ctx.to(device=tokenized_prompts.device, dtype=cast_dtype)
    with torch.no_grad():
        x = clip_model.token_embedding(tokenized_prompts).to(device=tokenized_prompts.device, dtype=cast_dtype)
    x[:, ctx_positions, :] = prompt_ctx
    seq_len = x.shape[1]
    x = x + clip_model.positional_embedding[:seq_len].to(dtype=cast_dtype)
    x = transformer(x, attn_mask=clip_model.attn_mask[:seq_len, :seq_len].to(dtype=cast_dtype))
    x = clip_model.ln_final(x)
    pooled = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
    if hasattr(clip_model, "text_projection"): pooled = pooled @ clip_model.text_projection
    return pooled
def load_prompt_bundle(prompt_ckpt: str, clip_model_text, raw_classnames, device: str):
    prompt_ckpt = str(prompt_ckpt)
    prefix = prompt_ckpt[:-3] if prompt_ckpt.endswith(".pt") else prompt_ckpt
    print(f"[Loader] Loading bundle from: {prompt_ckpt}")
    try:
        main_ckpt = torch.load(prompt_ckpt, map_location="cpu")
        pfgr_state_dict = main_ckpt if isinstance(main_ckpt, dict) else None
    except Exception as e:
        print(f"[Loader] Error loading PFGR weights: {e}")
        pfgr_state_dict = None
    point_prompt_path = prefix + "_point_prompt.pt"
    if os.path.exists(point_prompt_path):
        prompt_xyz = torch.load(point_prompt_path, map_location="cpu")
        print("[Loader] Loaded Point Prompt XYZ.")
    else:
        prompt_xyz = None
        print(f"[Loader] Warning: Point prompt file not found: {point_prompt_path}")
    ctx_cont = torch.load(prefix + "_ctx_cont.pt", map_location="cpu")
    ctx_soft = torch.load(prefix + "_ctx_soft.pt", map_location="cpu")
    ctx_tokens = torch.load(prefix + "_ctx_tokens.pt", map_location="cpu")
    if isinstance(ctx_tokens, torch.Tensor):
        ctx_tokens = ctx_tokens.view(-1)
    else:
        ctx_tokens = torch.tensor(ctx_tokens, dtype=torch.long).view(-1)
    n_ctx = int(ctx_cont.shape[0])
    n_soft = int(ctx_soft.shape[0])
    clean_classnames = [c.replace("_", " ") for c in raw_classnames]
    soft_prefix = " ".join(["X"] * n_soft)
    hard_prefix = " ".join(["X"] * n_ctx)
    text_prompts = [f"{soft_prefix} A depth picture of a {hard_prefix} {name}." for name in clean_classnames]
    tokenized_prompts = torch.stack([tokenizer(p) for p in text_prompts]).to(device)
    first = tokenized_prompts[0]
    x_id = _get_single_token_id("X", True)
    if x_id is None: x_id = _get_single_token_id("X", False)
    positions = (first == x_id).nonzero(as_tuple=False).squeeze(-1)
    positions = positions[:(n_soft + n_ctx)]
    token_emb = clip_model_text.token_embedding.weight.to(device)
    ctx_discrete = token_emb.index_select(0, ctx_tokens.to(device)).float()
    ctx_all = torch.cat([ctx_soft.to(device).float(), ctx_discrete.float()], dim=0)
    prompted_text_features = encode_text_with_prompt(clip_model_text, ctx_all, tokenized_prompts, positions)
    prompted_text_features = prompted_text_features / prompted_text_features.norm(dim=-1, keepdim=True)
    std_prompts = [f"X X X A depth picture of {c}." for c in clean_classnames]
    std_tokens = torch.stack([tokenizer(p) for p in std_prompts]).to(device)
    with torch.no_grad():
        std_text_features = clip_model_text.encode_text(std_tokens)
        std_text_features = std_text_features / std_text_features.norm(dim=-1, keepdim=True)
    return pfgr_state_dict, prompt_xyz, std_text_features, prompted_text_features
@torch.no_grad()
def eval_accuracy(model, loader, text_features, device="cuda:0"):
    model.eval()
    correct = 0
    total = 0
    text_features = text_features.to(device)
    for points, labels in loader:
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if points.shape[-1] == 3:
            colors = torch.full(points.shape, 0.5, device=device)
            points = torch.cat([points, colors], dim=-1)
        pc_feat = model.encode_pc(points)
        pc_feat = pc_feat / pc_feat.norm(dim=-1, keepdim=True)
        sim = torch.matmul(pc_feat, text_features.t())
        pred = sim.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)
def run_ablation(split_name, loader, model, std_text, prompted_text, pfgr_state, prompt_xyz):
    print(f"\n=== {split_name} ===")
    acc_clean = eval_accuracy(model, loader, std_text, DEVICE)
    print(f"[{split_name}] clean (Baseline) : {acc_clean:.4f}")
    acc_text = eval_accuracy(model, loader, prompted_text, DEVICE)
    print(f"[{split_name}] text prompt      : {acc_text:.4f}")
    applicator = PointPromptApplicator(model, pfgr_state, prompt_xyz)
    applicator.apply()
    acc_point = eval_accuracy(model, loader, std_text, DEVICE)
    print(f"[{split_name}] point prompt     : {acc_point:.4f}")
    acc_both = eval_accuracy(model, loader, prompted_text, DEVICE)
    print(f"[{split_name}] text+point       : {acc_both:.4f}")
    applicator.remove()
    return {"clean": acc_clean, "text_prompt": acc_text, "point_prompt": acc_point, "text+point": acc_both}
def main():
    if not os.path.exists(PROMPT_CKPT_BEST):
        print(f"Error: Prompt file not found at {PROMPT_CKPT_BEST}")
        return
    print("[Init] Loading CLIP to generate text features...")
    try:
        clip_model_text, _, _ = open_clip.create_model_and_transforms_text("ViT-bigG-14", pretrained=None)
    except:
        clip_model_text, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained=None)
    clip_model_text.load_state_dict(load_file(CLIP_SAFETENSORS), strict=False)
    clip_model_text = clip_model_text.to(DEVICE).eval()
    temp_ds = ModelNet40Dataset(root_dir=MODELNET_DIR, split="test")
    pfgr_state, prompt_xyz, std_text, prompted_text = load_prompt_bundle(
        PROMPT_CKPT_BEST, clip_model_text, temp_ds.classnames, DEVICE
    )
    print("[Init] Deleting CLIP model to free VRAM...")
    del clip_model_text
    torch.cuda.empty_cache()
    gc.collect()
    print("[Init] Loading Student Model...")
    class DummyCLIP:
        def __init__(self):
            self.visual = nn.Identity()
    model_student = get_ulip2_pointbert_student(DummyCLIP()).to(DEVICE).eval()
    clean_path = ensure_clean_1024_npz(CLEAN_NPZ_PATH)
    clean_loader = DataLoader(NpzPointCloudDataset(clean_path), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    whitebox_loader = DataLoader(NpzPointCloudDataset(WHITEBOX_NPZ_PATH), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    if os.path.exists(BLACKBOX_NPZ_PATH):
        blackbox_loader = DataLoader(NpzPointCloudDataset(BLACKBOX_NPZ_PATH), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    else:
        blackbox_loader = None
        print("[Warning] Blackbox data not found.")
    results = {}
    results["clean_1024"] = run_ablation("Clean 1024", clean_loader, model_student, std_text, prompted_text, pfgr_state, prompt_xyz)
    results["pgd_whitebox"] = run_ablation("PGD Whitebox", whitebox_loader, model_student, std_text, prompted_text, pfgr_state, prompt_xyz)
    if blackbox_loader:
        results["pgd_blackbox"] = run_ablation("PGD Blackbox", blackbox_loader, model_student, std_text, prompted_text, pfgr_state, prompt_xyz)
if __name__ == "__main__":
    main()
