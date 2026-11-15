import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import gc
import copy
import utils.open_clip as open_clip
from models.ULIP_models import ULIP2_WITH_OPENCLIP
from models.pointbert.point_encoder import PointTransformer_Colored
from utils.utils import cfg_from_yaml_file
from utils.tokenizer import SimpleTokenizer
from drop import SaliencyDrop
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MRPD_ROOT = "XXX"
DATA_ROOT = os.path.join(MRPD_ROOT, 'XXX')
ROOT_DIR_MN40 = os.path.join(DATA_ROOT, 'XXX')
CLEAN_NPZ_PATH = os.path.join(DATA_ROOT, 'XXX')
WHITEBOX_DIR = os.path.join(MRPD_ROOT, 'XXX')
WHITEBOX_FILE = 'XXX'
WHITEBOX_PATH = os.path.join(WHITEBOX_DIR, WHITEBOX_FILE)
BLACKBOX_PATH = "XXX"
os.makedirs(WHITEBOX_DIR, exist_ok=True)
PROMPT_DIR = "XXX"
PROMPT_FILE_NAME = 'XXX'
BEST_PROMPT_PATH = os.path.join(PROMPT_DIR, PROMPT_FILE_NAME)
MODEL_CONFIG_PATH = os.path.join(MRPD_ROOT, 'XXX')
ULIP2_PRETRAIN_CKPT = os.path.join(MRPD_ROOT, 'XXX')
CLIP_SAFETENSORS = os.path.join(
    MRPD_ROOT,
    'XXX'
)
EVAL_BATCH_SIZE = 16
GEN_BATCH_SIZE = 16
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
            if self.pfgr_state_dict is None:
                logger.warning("[Applicator] No PFGR state dict provided.")
        if self.prompt_xyz_tensor is not None:
            group_module = None
            if hasattr(point_encoder, 'group_divider'):
                group_module = getattr(point_encoder, 'group_divider')
            else:
                for name, m in point_encoder.named_modules():
                    lname = name.lower()
                    if ('group_divider' in lname) or (m.__class__.__name__.lower() == 'group'):
                        group_module = m
                        break
            if group_module:
                self.hook_module = PostFPSPointPrompt(self.prompt_xyz_tensor.to(DEVICE))
                self.hook_handle = group_module.register_forward_hook(self.hook_module.forward_hook)
    def remove(self):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
        if self.pfgr_target_name and self.original_layer_ref:
            point_encoder = self.model.point_encoder
            _set_submodule(point_encoder, self.pfgr_target_name, self.original_layer_ref)
            self.pfgr_module = None
def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    out = []
    for i in range(B):
        out.append(xyz[i, centroids[i]])
    return torch.stack(out, dim=0)
def random_target(labels, num_classes=40):
    targets = torch.randint_like(labels, low=0, high=num_classes)
    targets = torch.where(targets == labels, (targets + 1) % num_classes, targets)
    return targets
class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split="modelnet40_test", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        shape_names = os.path.join(root_dir, 'XXX')
        with open(shape_names, "r") as f:
            self.classnames = [line.strip() for line in f.readlines()]
        split_file = os.path.join(root_dir, f"{split}.txt")
        with open(split_file, "r") as f:
            self.samples = [line.strip() for line in f.readlines()]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        class_name = "_".join(sample_name.split("_")[:-1])
        label = self.classnames.index(class_name)
        file_path = os.path.join(self.root_dir, class_name, sample_name + ".txt")
        point_set = np.loadtxt(file_path, delimiter=",").astype(np.float32)
        points = point_set[:, 0:3]
        if self.transform is not None:
            points = self.transform(points)
        return points, label
class NpzPointCloudDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        if "points" in data: pts = data["points"]
        elif "test_pc" in data: pts = data["test_pc"]
        else: pts = data[list(data.keys())[0]]
        if "labels" in data: lbs = data["labels"]
        elif "test_label" in data: lbs = data["test_label"]
        else: lbs = data[list(data.keys())[1]]
        pts = pts.astype(np.float32)
        lbs = lbs.astype(np.int64)
        if pts.shape[-1] == 3:
            rgb = np.ones((pts.shape[0], pts.shape[1], 3), dtype=np.float32) * 0.5
            pts = np.concatenate([pts, rgb], axis=-1)
        self.points = torch.from_numpy(pts).float()
        self.labels = torch.from_numpy(lbs).long()
    def __len__(self): return int(self.points.shape[0])
    def __getitem__(self, idx): return self.points[idx], self.labels[idx]
transform = transforms.Compose([
    lambda pc: np.hstack((pc[:, :3], np.full((pc.shape[0], 3), 0.5, dtype=np.float32))),
    lambda pc: torch.tensor(pc, dtype=torch.float32)
])
def get_student_model(args, open_clip_model=None):
    if open_clip_model is None:
        class DummyCLIP:
            def __init__(self): self.visual = nn.Identity()
        open_clip_model = DummyCLIP()
    config = cfg_from_yaml_file(MODEL_CONFIG_PATH)
    point_encoder = PointTransformer_Colored(config.model, args=args)
    model = ULIP2_WITH_OPENCLIP(
        open_clip_model=open_clip_model,
        point_encoder=point_encoder,
        pc_feat_dims=768,
        embed_dim=args.embed_dim
    )
    if os.path.exists(ULIP2_PRETRAIN_CKPT):
        checkpoint = torch.load(ULIP2_PRETRAIN_CKPT, map_location="cpu")
        sd = checkpoint.get("state_dict", checkpoint)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model_state = model.state_dict()
        filtered_sd = {}
        for k, v in sd.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_sd[k] = v
                elif v.ndim == 2 and v.shape[::-1] == model_state[k].shape:
                    filtered_sd[k] = v.t()
        model.load_state_dict(filtered_sd, strict=False)
        logger.info("Student Model loaded.")
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
def load_prompt_bundle(prompt_path, clip_model_text, classnames):
    logger.info(f"Loading prompt bundle from: {prompt_path}")
    prefix = prompt_path[:-3] if prompt_path.endswith(".pt") else prompt_path
    try:
        main_ckpt = torch.load(prompt_path, map_location="cpu")
        pfgr_state = main_ckpt if isinstance(main_ckpt, dict) else None
    except Exception as e:
        logger.warning(f"Error loading PFGR: {e}")
        pfgr_state = None
    prompt_xyz = None
    xyz_path = prefix + "_point_prompt.pt"
    if os.path.exists(xyz_path):
        prompt_xyz = torch.load(xyz_path, map_location="cpu")
    ctx_cont = torch.load(prefix + "_ctx_cont.pt", map_location="cpu")
    ctx_soft = torch.load(prefix + "_ctx_soft.pt", map_location="cpu")
    try:
        ctx_tokens = torch.load(prefix + "_ctx_tokens.pt", map_location="cpu")
        use_tokens = True
        if isinstance(ctx_tokens, torch.Tensor): ctx_tokens = ctx_tokens.view(-1)
        else: ctx_tokens = torch.tensor(ctx_tokens, dtype=torch.long).view(-1)
    except:
        use_tokens = False
        logger.warning("Ctx tokens not found, using continuous hard prompt.")
    n_soft = int(ctx_soft.shape[0])
    n_hard = int(ctx_cont.shape[0])
    clean_classnames = [name.replace("_", " ") for name in classnames]
    soft_prefix = " ".join(["X"] * n_soft)
    hard_prefix = " ".join(["X"] * n_hard)
    prompts = [f"{soft_prefix} A depth picture of a {hard_prefix} {name}." for name in clean_classnames]
    tokenized = torch.stack([tokenizer(p) for p in prompts]).to(DEVICE)
    first = tokenized[0]
    x_id = _get_single_token_id("X", True)
    if x_id is None: x_id = _get_single_token_id("X", False)
    positions = (first == x_id).nonzero(as_tuple=False).squeeze(-1)[:(n_soft + n_hard)]
    if use_tokens:
        token_emb = clip_model_text.token_embedding.weight.to(DEVICE)
        hard_emb = token_emb.index_select(0, ctx_tokens.to(DEVICE)).float()
    else:
        hard_emb = ctx_cont.to(DEVICE).float()
    ctx_all = torch.cat([ctx_soft.to(DEVICE).float(), hard_emb], dim=0)
    with torch.no_grad():
        prompted_text = encode_text_with_prompt(clip_model_text, ctx_all, tokenized, positions)
        prompted_text = F.normalize(prompted_text, dim=-1)
    std_prompts = [f"X X X A depth picture of {name}." for name in clean_classnames]
    std_tokenized = torch.stack([tokenizer(p) for p in std_prompts]).to(DEVICE)
    with torch.no_grad():
        std_text = clip_model_text.encode_text(std_tokenized)
        std_text = F.normalize(std_text, dim=-1)
    return pfgr_state, prompt_xyz, std_text, prompted_text
def eval_accuracy(model, loader, text_features, device=DEVICE):
    model.eval()
    correct, total = 0, 0
    text_features = text_features.to(device)
    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)
            pc_features = model.encode_pc(points)
            pc_features = F.normalize(pc_features, dim=-1)
            logits = torch.matmul(pc_features, text_features.t())
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += int(labels.numel())
    return correct / max(total, 1)
def run_ablation(title, loader, model, std_text, prompted_text, pfgr_state, prompt_xyz):
    print(f"\n==================== {title} ====================")
    acc_clean = eval_accuracy(model, loader, std_text)
    print(f"acc_clean      = {acc_clean:.4f}")
    acc_text = eval_accuracy(model, loader, prompted_text)
    print(f"acc_text       = {acc_text:.4f}")
    applicator = PointPromptApplicator(model, pfgr_state, prompt_xyz)
    applicator.apply()
    acc_point = eval_accuracy(model, loader, std_text)
    print(f"acc_point      = {acc_point:.4f}")
    acc_both = eval_accuracy(model, loader, prompted_text)
    print(f"acc_text+point = {acc_both:.4f}")
    applicator.remove()
    return acc_clean, acc_text, acc_point, acc_both
def main():
    parser = argparse.ArgumentParser(description="Drop-200 Test")
    parser.add_argument("--pc_model", type=str, default="eva02_base_patch14_448")
    parser.add_argument("--pc_feat_dim", type=int, default=768)
    parser.add_argument("--embed_dim", type=int, default=1280)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--pc_encoder_dim", type=int, default=512)
    args = parser.parse_args()
    logger.info("Initializing CLIP to generate text features...")
    try:
        from safetensors.torch import load_file
        clip_model_text, _, _ = open_clip.create_model_and_transforms_text("ViT-bigG-14", pretrained=None)
        clip_model_text.load_state_dict(load_file(CLIP_SAFETENSORS), strict=False)
    except:
        clip_model_text, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained=None)
    clip_model_text = clip_model_text.to(DEVICE).eval()
    dummy_ds = ModelNet40Dataset(root_dir=ROOT_DIR_MN40, split="modelnet40_test", transform=None)
    logger.info("Loading Prompt Bundle...")
    pfgr_state, prompt_xyz, std_text, prompted_text = load_prompt_bundle(
        BEST_PROMPT_PATH, clip_model_text, dummy_ds.classnames
    )
    logger.info("Deleting CLIP to free VRAM...")
    del clip_model_text
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Loading Student Model...")
    model_student = get_student_model(args).to(DEVICE).eval()
    if os.path.exists(CLEAN_NPZ_PATH):
        ds_clean = NpzPointCloudDataset(CLEAN_NPZ_PATH)
        loader_clean = DataLoader(ds_clean, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        run_ablation("clean", loader_clean, model_student, std_text, prompted_text, pfgr_state, prompt_xyz)
    if not os.path.exists(WHITEBOX_PATH):
        logger.info(f"Generating Whitebox Drop Data: {WHITEBOX_PATH}")
        raw_test_ds = ModelNet40Dataset(root_dir=ROOT_DIR_MN40, split="modelnet40_test", transform=transform)
        gen_loader = DataLoader(raw_test_ds, batch_size=GEN_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        attacker = SaliencyDrop(model_student, num_drop=200, alpha=1, k=5, text_features=std_text)
        all_pts, all_lbs = [], []
        for points, labels in tqdm(gen_loader):
            points, labels = points.to(DEVICE), labels.to(DEVICE)
            xyz = points[:, :, :3]
            xyz_1024 = farthest_point_sample(xyz, 1024)
            colors = torch.full_like(xyz_1024, 0.5)
            pts_1024 = torch.cat([xyz_1024, colors], dim=-1)
            target = random_target(labels, 40)
            adv_np, _ = attacker.attack(pts_1024, target)
            all_pts.append(adv_np)
            all_lbs.append(labels.cpu().numpy())
        np.savez(WHITEBOX_PATH, points=np.concatenate(all_pts), labels=np.concatenate(all_lbs))
        del gen_loader, attacker, all_pts
        gc.collect()
    ds_white = NpzPointCloudDataset(WHITEBOX_PATH)
    loader_white = DataLoader(ds_white, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    run_ablation("Drop-200 Whitebox", loader_white, model_student, std_text, prompted_text, pfgr_state, prompt_xyz)
    if os.path.exists(BLACKBOX_PATH):
        ds_black = NpzPointCloudDataset(BLACKBOX_PATH)
        loader_black = DataLoader(ds_black, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        run_ablation("Drop-200 Blackbox", loader_black, model_student, std_text, prompted_text, pfgr_state, prompt_xyz)
    else:
        logger.warning(f"Blackbox not found: {BLACKBOX_PATH}")
if __name__ == "__main__":
    main()
