import os
import argparse
import sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--cuda_visible_devices", type=str, default=None)
_pre_args, _ = _pre_parser.parse_known_args(sys.argv[1:])
if _pre_args.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.cuda_visible_devices
elif "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from pathlib import Path
import copy
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.uni3d import create_uni3d
import utils.open_clip as open_clip
from loguru import logger
import psutil
import gc
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import h5py

from utils.projection import process_point_cloud_batch
from utils.tokenizer import SimpleTokenizer


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", ""))
task = "MN_s"
script_name = Path(__file__).stem
suffix = f"_{script_name}"

save_dir = Path('XXX')
save_dir.mkdir(parents=True, exist_ok=True)

logger.info(f"Current task: {task}")

tokenizer = SimpleTokenizer()


def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_xyzrgb_torch(pc: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(pc):
        raise TypeError(f"got {type(pc)}")
    if pc.dim() not in (2, 3):
        raise ValueError(f"got shape={tuple(pc.shape)}")
    if pc.shape[-1] < 3:
        raise ValueError(f"got shape={tuple(pc.shape)}")
    xyz = pc[..., :3]
    rgb = torch.full(xyz.shape[:-1] + (3,), 0.5, device=pc.device, dtype=pc.dtype)
    return torch.cat([xyz, rgb], dim=-1)


def ensure_xyzrgb_np(pc: np.ndarray) -> np.ndarray:
    pc = np.asarray(pc)
    if pc.ndim not in (2, 3):
        raise ValueError(f"got shape={pc.shape}")
    if pc.shape[-1] < 3:
        raise ValueError(f"got shape={pc.shape}")
    xyz = pc[..., :3].astype(np.float32, copy=False)
    rgb = np.full(xyz.shape[:-1] + (3,), 0.5, dtype=np.float32)
    return np.concatenate([xyz, rgb], axis=-1)


class NpzPointCloudDataset(Dataset):
    def __init__(self, npz_path, classnames):
        self.data = np.load(npz_path, mmap_mode='r')
        self.points = self.data['points']
        self.labels = self.data['labels']

        if 'orginal_points' in self.data.files:
            self.orginal_points = self.data['orginal_points']
        elif 'original_points' in self.data.files:
            self.orginal_points = self.data['original_points']
        else:
            raise KeyError(
                f"Actual keys={self.data.files}"
            )

        self.classnames = classnames

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        point_adv = torch.tensor(self.points[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        point_clean = torch.tensor(self.orginal_points[idx], dtype=torch.float32)

        point_adv = ensure_xyzrgb_torch(point_adv)
        point_clean = ensure_xyzrgb_torch(point_clean)
        return point_clean, label, point_adv


class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data = []
        self.labels = []
        self.classnames = []

        classnames_path = os.path.join(root_dir, 'XXX')
        if os.path.exists(classnames_path):
            with open(classnames_path, 'r') as f:
                self.classnames = [line.strip() for line in f.readlines()]
        else:
            raise FileNotFoundError(f"Classnames file not found: {classnames_path}")

        split_file = os.path.join(root_dir, f'{split}.txt')
        with open(split_file, 'r') as f:
            file_list = [line.strip() for line in f.readlines()]

        for item in file_list:
            category = item.rsplit('_', 1)[0]
            file_path = os.path.join(root_dir, category, f'{item}.txt')
            if not os.path.exists(file_path):
                logger.warning(f'[Warning] File not found: {file_path}')
                continue
            self.data.append(file_path)
            self.labels.append(self.classnames.index(category))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pc_path = self.data[idx]
        label = self.labels[idx]
        points = np.loadtxt(pc_path, delimiter=',').astype(np.float32)[:, 0:3]
        if self.transform:
            points = self.transform(points)
        return points, label


class ScanObjectNNDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform

        with h5py.File(self.h5_path, 'r') as f:
            self.data = f['data'][:]
            self.labels = f['label'][:]

        self.labels = self.labels.squeeze()
        self.classnames = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table",
                           "bed", "pillow", "sink", "sofa", "toilet"]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        points = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            points = self.transform(points)
        return points, label


def print_memory(tag: str):
    cpu_mem = psutil.virtual_memory()
    print(f"[{tag}] CPU memory usage: {(cpu_mem.total - cpu_mem.available) / (1024**2):.2f} MB / {cpu_mem.total / (1024**2):.2f} MB")


def save_point_prompt_visualization(prompt_xyz, save_path, title="Point Prompt (XYZ)"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"Cannot visualize point cloud prompt: {e}")
        return

    if torch.is_tensor(prompt_xyz):
        prompt_xyz = prompt_xyz.detach().cpu().numpy()
    prompt_xyz = np.asarray(prompt_xyz)

    if prompt_xyz.ndim == 3:
        prompt_xyz = prompt_xyz[0]
    if prompt_xyz.shape[-1] > 3:
        prompt_xyz = prompt_xyz[:, :3]

    if prompt_xyz.size == 0:
        logger.warning("Point cloud prompt is empty, skipping visualization.")
        return

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(prompt_xyz[:, 0], prompt_xyz[:, 1], prompt_xyz[:, 2], s=12)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


transform = transforms.Compose([
    lambda pc: np.hstack((pc[:, :3], np.full((pc.shape[0], 3), 0.5, dtype=np.float32))),
    lambda pc: torch.tensor(pc, dtype=torch.float32)
])


def contrastive_loss(point_features, text_features, tau=0.07):
    B = point_features.size(0)
    logits = torch.matmul(point_features, text_features.t()) / tau
    labels = torch.arange(B, device=point_features.device, dtype=torch.long)
    loss_p2t = F.cross_entropy(logits, labels)
    loss_t2p = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_p2t + loss_t2p)


def selective_contrastive_loss(
    anchor_features,
    reference_features,
    reference_labels,
    ground_truth_labels,
    tau=0.07,
    topk=1,
    return_stats: bool = False,
):
    with torch.no_grad():
        logits_ref = torch.matmul(reference_features, reference_labels.t()) / tau
        topk_preds = logits_ref.topk(k=topk, dim=1).indices
        ground_truth_labels = ground_truth_labels.view(-1, 1)
        correct_mask = (topk_preds == ground_truth_labels).any(dim=1)

        correct_ratio = float(correct_mask.float().mean().item())

        if correct_mask.sum() == 0:
            zero = torch.tensor(0.0, device=anchor_features.device, requires_grad=True)
            return (zero, correct_ratio) if return_stats else zero

    anchor_selected = anchor_features[correct_mask]
    reference_selected = reference_features[correct_mask]

    B_sel = anchor_selected.size(0)
    logits = torch.matmul(anchor_selected, reference_selected.t()) / tau
    labels_sel = torch.arange(B_sel, device=anchor_features.device)

    loss_a2r = F.cross_entropy(logits, labels_sel)
    loss_r2a = F.cross_entropy(logits.t(), labels_sel)

    loss = 0.5 * (loss_a2r + loss_r2a)
    return (loss, correct_ratio) if return_stats else loss


def geometric_alignment_loss(student_features, teacher_features, mode='mse'):
    student_norm = F.normalize(student_features, dim=-1)
    teacher_norm = F.normalize(teacher_features, dim=-1)

    if mode == 'mse':
        return F.mse_loss(student_norm, teacher_norm)
    elif mode == 'cosine':
        cosine_sim = F.cosine_similarity(student_norm, teacher_norm, dim=-1)
        return (1 - cosine_sim).mean()
    elif mode == 'combined':
        mse = F.mse_loss(student_norm, teacher_norm)
        cosine_sim = F.cosine_similarity(student_norm, teacher_norm, dim=-1)
        cosine_loss = (1 - cosine_sim).mean()
        return mse + 0.5 * cosine_loss
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _tokenize_single(text: str) -> torch.Tensor:
    ids = tokenizer(text)
    if isinstance(ids, torch.Tensor) and ids.ndim == 2:
        ids = ids[0]
    return ids


def _get_single_token_id(word: str, add_space: bool = True):
    text = (" " + word) if add_space else word
    ids = _tokenize_single(text)
    eot_pos = int(ids.argmax().item())
    content = ids[1:eot_pos]
    content = content[content != 0]
    if int(content.numel()) == 1:
        return int(content.item())
    return None


def _build_candidate_token_ids(words):
    ids = []
    seen = set()
    for w in words:
        tid = _get_single_token_id(w, add_space=True)
        if tid is None:
            continue
        if tid not in seen:
            ids.append(tid)
            seen.add(tid)
    return ids


def find_open_clip_text_tower(model: nn.Module) -> nn.Module:
    if all(hasattr(model, k) for k in ("token_embedding", "positional_embedding", "transformer", "ln_final")):
        return model

    for attr in ("text", "text_model", "text_encoder", "text_tower"):
        if hasattr(model, attr):
            m = getattr(model, attr)
            if all(hasattr(m, k) for k in ("token_embedding", "positional_embedding", "transformer", "ln_final")):
                return m

    for _, m in model.named_modules():
        if all(hasattr(m, k) for k in ("token_embedding", "positional_embedding", "transformer", "ln_final")):
            return m

    raise AttributeError(
        f"Cannot find text tower in {type(model).__name__}."
    )


def encode_text_with_prompt(text_tower: nn.Module, prompt_ctx: torch.Tensor, tokenized_prompts: torch.Tensor,
                           ctx_positions: torch.Tensor = None) -> torch.Tensor:
    if prompt_ctx.dim() == 2:
        prompt_ctx = prompt_ctx.unsqueeze(0).expand(tokenized_prompts.shape[0], -1, -1)

    transformer = getattr(text_tower, "transformer")
    if hasattr(transformer, "get_cast_dtype"):
        cast_dtype = transformer.get_cast_dtype()
    elif hasattr(text_tower, "cast_dtype") and text_tower.cast_dtype is not None:
        cast_dtype = text_tower.cast_dtype
    else:
        cast_dtype = prompt_ctx.dtype

    prompt_ctx = prompt_ctx.to(device=tokenized_prompts.device, dtype=cast_dtype)

    with torch.no_grad():
        x = text_tower.token_embedding(tokenized_prompts).to(device=tokenized_prompts.device, dtype=cast_dtype)

    n_ctx = prompt_ctx.shape[1]
    if ctx_positions is None:
        ctx_positions = torch.arange(1, 1 + n_ctx, device=tokenized_prompts.device, dtype=torch.long)
    else:
        ctx_positions = ctx_positions.to(device=tokenized_prompts.device, dtype=torch.long)

    x[:, ctx_positions, :] = prompt_ctx

    seq_len = x.shape[1]
    positional = text_tower.positional_embedding[:seq_len].to(device=tokenized_prompts.device, dtype=cast_dtype)
    x = x + positional

    attn_mask = torch.full((seq_len, seq_len), float('-inf'), device=tokenized_prompts.device)
    attn_mask = torch.triu(attn_mask, diagonal=1).to(dtype=cast_dtype)

    ln_final = getattr(text_tower, "ln_final")

    layout = getattr(text_tower, "_prompt_encode_layout", None)

    def _infer_batch_first(mod: nn.Module) -> bool:
        if hasattr(mod, 'batch_first'):
            try:
                return bool(getattr(mod, 'batch_first'))
            except Exception:
                pass
        for sub in mod.modules():
            if isinstance(sub, nn.MultiheadAttention):
                return bool(getattr(sub, 'batch_first', False))
        return True

    if layout is None:
        layout = 'batch_first' if _infer_batch_first(transformer) else 'seq_first'
        text_tower._prompt_encode_layout = layout

    def _run_seq_first(x_in: torch.Tensor) -> torch.Tensor:
        x2 = x_in.permute(1, 0, 2)
        x2 = transformer(x2, attn_mask=attn_mask)
        if isinstance(x2, (tuple, list)):
            x2 = x2[0]
        x2 = x2.permute(1, 0, 2)
        x2 = ln_final(x2)
        return x2

    def _run_batch_first(x_in: torch.Tensor) -> torch.Tensor:
        x2 = transformer(x_in, attn_mask=attn_mask)
        if isinstance(x2, (tuple, list)):
            x2 = x2[0]
        x2 = ln_final(x2)
        return x2

    if layout == 'seq_first':
        x_out = _run_seq_first(x)
    else:
        x_out = _run_batch_first(x)

    eot_pos = tokenized_prompts.argmax(dim=-1)
    pooled = x_out[torch.arange(x_out.shape[0], device=tokenized_prompts.device), eot_pos]

    text_projection = getattr(text_tower, "text_projection", None)
    if text_projection is not None:
        if isinstance(text_projection, nn.Linear):
            pooled = text_projection(pooled)
        else:
            pooled = pooled @ text_projection

    return pooled


class PFGR_Module(nn.Module):
    def __init__(self, original_layer: nn.Linear, lambda_scale=1.0):
        super().__init__()
        self.original_layer = original_layer
        self.lambda_scale = float(lambda_scale)
        self._is_reparameterized = False


        for param in self.original_layer.parameters():
            param.requires_grad = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features
        self.pfgr_branch = nn.Linear(in_features, out_features, bias=False)


        nn.init.normal_(self.pfgr_branch.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            weight_norm = self.pfgr_branch.weight.data.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            self.pfgr_branch.weight.data = (self.pfgr_branch.weight.data / weight_norm) * 0.1

        logger.info(f"[PFGR] Initialization completed: in={in_features}, out={out_features}, λ={lambda_scale}")

    def forward(self, x):
        if self._is_reparameterized or self.lambda_scale == 0.0:
            return self.original_layer(x)
        return self.original_layer(x) + self.lambda_scale * self.pfgr_branch(x)

    def get_merged_weight(self):
        with torch.no_grad():
            merged_weight = self.original_layer.weight.data + self.lambda_scale * self.pfgr_branch.weight.data
            merged_bias = self.original_layer.bias.data if self.original_layer.bias is not None else None
        return merged_weight, merged_bias

    def reparameterize(self):
        merged_weight, merged_bias = self.get_merged_weight()
        self.original_layer.weight.data.copy_(merged_weight)
        if merged_bias is not None and self.original_layer.bias is not None:
            self.original_layer.bias.data.copy_(merged_bias)

        self._is_reparameterized = True
        self.lambda_scale = 0.0
        with torch.no_grad():
            self.pfgr_branch.weight.zero_()
        logger.info("[PFGR] Reparameterization completed: merged weights and closed PFGR branch")


class PostFPSPointPrompt(nn.Module):
    def __init__(self, prompt_len=10, init_scale=0.05, device='cuda:0'):
        super().__init__()
        self.prompt_len = int(prompt_len)
        self.prompt_xyz = nn.Parameter(torch.randn(1, self.prompt_len, 3, device=device) * float(init_scale))
        self.hook_attached = False
        self.hook_handle = None

    def forward_hook(self, module, inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            return output

        neighborhood = output[0]
        center = output[1]

        if not (torch.is_tensor(neighborhood) and torch.is_tensor(center)):
            return output

        if neighborhood.dim() != 4 or center.dim() != 3:
            return output

        B = int(center.shape[0])
        P = int(self.prompt_xyz.shape[1])
        prompt_xyz = self.prompt_xyz.to(device=center.device, dtype=center.dtype).expand(B, -1, -1)

        Cc = int(center.shape[-1])
        if Cc == 3:
            prompt_center = prompt_xyz
        elif Cc < 3:
            prompt_center = prompt_xyz[..., :Cc]
        else:
            pad_dims = Cc - 3
            pad = torch.zeros((B, P, pad_dims), device=center.device, dtype=center.dtype)
            if pad_dims >= 3:
                pad[..., :3] = 0.5
            else:
                pad[...] = 0.5
            prompt_center = torch.cat([prompt_xyz, pad], dim=-1)

        S = int(neighborhood.shape[2])
        C = int(neighborhood.shape[3])
        prompt_neighborhood = torch.zeros((B, P, S, C), device=neighborhood.device, dtype=neighborhood.dtype)
        if C >= 3:
            prompt_neighborhood[..., :3] = 0.0
        if C >= 6:
            prompt_neighborhood[..., 3:6] = 0.5

        neighborhood_new = torch.cat([neighborhood, prompt_neighborhood], dim=1)
        center_new = torch.cat([center, prompt_center], dim=1)

        new_out = list(output)
        new_out[0] = neighborhood_new
        new_out[1] = center_new

        if len(output) >= 3 and torch.is_tensor(output[2]) and output[2].dim() == 4:
            features = output[2]
            Cf = int(features.shape[-1])
            prompt_features = torch.zeros((B, P, S, Cf), device=features.device, dtype=features.dtype)
            if Cf >= 3:
                prompt_features[..., :3] = 0.0
            if Cf >= 6:
                prompt_features[..., 3:6] = 0.5
            features_new = torch.cat([features, prompt_features], dim=1)
            new_out[2] = features_new

        return tuple(new_out)

    def attach(self, root_module: nn.Module):
        group_module = None

        if hasattr(root_module, 'group_divider'):
            group_module = getattr(root_module, 'group_divider')

        if group_module is None:
            for name, m in root_module.named_modules():
                lname = name.lower()
                cname = m.__class__.__name__.lower()
                if ('group_divider' in lname) or (cname == 'group') or ('groupdivider' in cname):
                    group_module = m
                    break

        if group_module is None:
            logger.warning("[Post-FPS Prompt] group_divider / group module not found, Post-FPS insertion will not take effect.")
            self.hook_attached = False
            return None

        self.hook_handle = group_module.register_forward_hook(self.forward_hook)
        self.hook_attached = True
        logger.info(f"[Post-FPS Prompt] Mounted to module: {group_module.__class__.__name__}")
        return self.hook_handle


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


def _guess_point_encoder(student_model: nn.Module) -> nn.Module:

    for attr in ['point_encoder', 'pc_encoder', 'encoder', 'backbone', 'point_backbone']:
        if hasattr(student_model, attr):
            return getattr(student_model, attr)

    return student_model


class Uni3DPrompt(nn.Module):
    def __init__(
        self,
        student_model: nn.Module,
        clip_model_text: nn.Module,
        classnames,
        device='cuda:0',
        text_device='cuda:1',

        text_prompt_size=3,
        soft_prompt_size=3,
        use_pez_text=True,
        attribute_seeds=None,
        attribute_pool=None,

        point_prompt_len=10,
        pfgr_lambda=1.0,

        min_log_var=-0.8,
        max_log_var=2.0,

        enable_legacy_prompt_fallback=False,
        legacy_prompt_feat_dim=None,
    ):
        super().__init__()

        self.pc_device = torch.device(device)

        self.text_device = torch.device(text_device)

        self.device = self.pc_device
        self.log_tau = nn.Parameter(torch.tensor([0.07, 0.07, 0.07], device=self.pc_device).log())


        self.clip_model_text = clip_model_text.to(self.text_device)
        for p in self.clip_model_text.parameters():
            p.requires_grad = False
        self.clip_model_text.eval()

        self.text_tower = find_open_clip_text_tower(self.clip_model_text)


        self.use_pez_text = bool(use_pez_text)
        self.n_ctx = int(text_prompt_size)
        self.n_soft = int(soft_prompt_size)

        assert classnames is not None and len(classnames) > 0, "classnames cannot be empty"
        classnames = [name.replace("_", " ") for name in classnames]


        soft_prefix = " ".join(["X"] * self.n_soft)
        hard_prefix = " ".join(["X"] * self.n_ctx)
        text_prompts = [f"{soft_prefix} A depth picture of a {hard_prefix} {name}." for name in classnames]
        tokenized_prompts = torch.stack([tokenizer(p) for p in text_prompts])
        self.tokenized_prompts = tokenized_prompts.to(self.text_device)


        x_id_space = _get_single_token_id("X", add_space=True)
        x_id_nospace = _get_single_token_id("X", add_space=False)

        first = self.tokenized_prompts[0]
        positions = []
        if x_id_space is not None:
            positions.append((first == x_id_space).nonzero(as_tuple=False).squeeze(-1))
        if x_id_nospace is not None and x_id_nospace != x_id_space:
            positions.append((first == x_id_nospace).nonzero(as_tuple=False).squeeze(-1))

        if len(positions) == 0:
            raise RuntimeError("Cannot locate placeholder token 'X', please check if tokenizer matches prompt template.")

        positions = torch.cat(positions, dim=0)
        positions = torch.unique(positions).sort()[0]
        total_ctx = self.n_soft + self.n_ctx
        positions = positions[:total_ctx]

        if int(positions.numel()) != total_ctx:
            raise RuntimeError(
                f"Insufficient placeholder X: expected {total_ctx}, but found {int(positions.numel())}.\n"
                f"Suggestion: Check if the template string contains enough 'X', and if the tokenizer's tokenization of 'X' is consistent."
            )

        self.register_buffer("ctx_positions", positions.to(self.text_device))


        if attribute_pool is None:
            attribute_pool = {
                "shape": [
                    "round", "square", "rectangular", "cylindrical", "spherical", "flat", "thin", "thick",
                    "long", "short", "curved", "angular", "pointy", "hollow", "solid", "wide", "narrow",
                    "tall", "small", "large", "oval", "hexagonal", "triangular", "boxy", "bulky", "slim",
                    "compact", "elongated", "broad", "tapered", "slanted", "arched", "concave", "convex"
                ],
                "texture": [
                    "smooth", "rough", "shiny", "matte", "glossy", "fuzzy", "striped", "spotted",
                    "checkered", "textured", "polished", "grainy", "wrinkled", "soft", "hard",
                    "coarse", "fine", "bumpy", "ridged", "ribbed", "woven", "slippery", "sticky",
                    "leathery", "rubbery", "sparkly", "dusty", "patterned"
                ],
                "material": [
                    "metal", "wood", "plastic", "glass", "rubber", "fabric", "leather", "ceramic",
                    "paper", "stone", "concrete", "foam", "cardboard", "silicone", "granite",
                    "marble", "clay", "steel", "aluminum", "bronze", "carbon"
                ],
            }


        if self.n_ctx <= 1:
            slot_types = ["shape"]
        elif self.n_ctx == 2:
            slot_types = ["shape", "texture"]
        elif self.n_ctx == 3:
            slot_types = ["shape", "texture", "material"]
        else:

            slot_types = ["shape", "texture", "material"] + ["shape"] * (self.n_ctx - 3)


        if attribute_seeds is None:
            seed_map = {
                "shape": ["round", "flat", "boxy", "hollow", "solid"],
                "texture": ["smooth", "rough", "matte", "polished"],
                "material": ["metal", "wood", "plastic", "glass"],
            }
            attribute_seeds = []
            for t in slot_types:
                attribute_seeds.append(seed_map.get(t, ["round"])[0])


        merged_pool = []
        for k in ("shape", "texture", "material"):
            merged_pool.extend(attribute_pool.get(k, []))

        with torch.no_grad():
            token_emb_weight = self.text_tower.token_embedding.weight
            for i in range(self.n_ctx):
                pool_name = slot_types[i] if i < len(slot_types) else "geometry"
                words_i = attribute_pool.get(pool_name, merged_pool)
                if not words_i:
                    words_i = merged_pool

                cand_ids_i = _build_candidate_token_ids(words_i)

                if len(cand_ids_i) == 0:
                    logger.warning(f"[TextPrompt] slot-{i} pool='{pool_name}' has no available single token, falling back to full vocabulary projection.")
                    vocab_size = int(token_emb_weight.shape[0])
                    cand_ids_i = list(range(vocab_size))


                pad_id = 0
                sot_id = int(self.tokenized_prompts[0, 0].item())
                eot_id = int(self.tokenized_prompts[0].max().item())
                cand_ids_i = [tid for tid in cand_ids_i if tid not in (pad_id, sot_id, eot_id)]

                cand_ids_i = torch.tensor(cand_ids_i, dtype=torch.long, device=token_emb_weight.device)
                emb_i = token_emb_weight.index_select(0, cand_ids_i).float()
                emb_norm_i = F.normalize(emb_i, dim=-1)

                self.register_buffer(f"cand_ids_{i}", cand_ids_i)
                self.register_buffer(f"cand_emb_{i}", emb_i)
                self.register_buffer(f"cand_emb_norm_{i}", emb_norm_i)


        self.slot_types = slot_types


        ctx_dim = int(self.text_tower.token_embedding.weight.shape[1])
        soft_ctx_vectors = torch.empty(self.n_soft, ctx_dim, device=self.text_device)
        nn.init.normal_(soft_ctx_vectors, std=0.05)
        self.ctx_soft = nn.Parameter(soft_ctx_vectors, requires_grad=True)


        init_vecs = []
        with torch.no_grad():
            token_emb_weight = self.text_tower.token_embedding.weight
            for i in range(self.n_ctx):
                seed = attribute_seeds[i] if i < len(attribute_seeds) else attribute_seeds[-1]
                seed_id = _get_single_token_id(seed, add_space=True)
                if seed_id is None:
                    seed_id = int(getattr(self, f"cand_ids_{i}")[0].item())
                init_vecs.append(token_emb_weight[seed_id].float())

        ctx_init = torch.stack(init_vecs, dim=0) + torch.randn_like(torch.stack(init_vecs, dim=0)) * 0.01
        self.ctx_cont = nn.Parameter(ctx_init, requires_grad=True)


        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)
        self.log_vars = nn.Parameter(torch.zeros(3, device=self.pc_device))
        self.log_vars_clamped = None


        self.student_model = student_model.to(device)
        self.student_model.eval()
        for p in self.student_model.parameters():
            p.requires_grad = False


        self.pfgr_module = None
        self._pfgr_target_path = None
        self._install_pfgr(lambda_scale=float(pfgr_lambda))


        self.post_fps_prompt = None
        self.enable_legacy_prompt_fallback = bool(enable_legacy_prompt_fallback)
        self.legacy_prompt = None

        if int(point_prompt_len) > 0:
            self.post_fps_prompt = PostFPSPointPrompt(prompt_len=int(point_prompt_len), device=device)
            try:
                point_encoder_root = _guess_point_encoder(self.student_model)
                self.post_fps_prompt.attach(point_encoder_root)
            except Exception as e:
                logger.warning(f"[Post-FPS Prompt] Mounting failed: {e}")
                self.post_fps_prompt.hook_attached = False


        if self.enable_legacy_prompt_fallback:
            if legacy_prompt_feat_dim is None:
                legacy_prompt_feat_dim = 192
            self.legacy_prompt = nn.Parameter(
                torch.randn(1, int(point_prompt_len), int(legacy_prompt_feat_dim), device=device) * 0.02,
                requires_grad=True
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.student_model is not None:
            self.student_model.eval()
        if self.clip_model_text is not None:
            self.clip_model_text.eval()
        return self


    def _install_pfgr(self, lambda_scale: float):
        point_encoder = _guess_point_encoder(self.student_model)

        prefer_names = ['reduce_dim', 'proj', 'projection', 'head', 'fc', 'mlp', 'out_proj', 'pc_proj']
        target_layer = None
        target_name = None


        for name in prefer_names:
            if hasattr(point_encoder, name):
                layer = getattr(point_encoder, name)
                if isinstance(layer, nn.Linear):
                    target_layer = layer
                    target_name = name
                    break


        if target_layer is None:
            candidates = []
            for name, module in point_encoder.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                score = 0
                lname = name.lower()
                if any(k in lname for k in ['reduce', 'proj', 'head', 'out']):
                    score += 10

                score += int(getattr(module, 'out_features', 0)) // 64
                candidates.append((score, name, module))

            if len(candidates) > 0:
                candidates.sort(key=lambda x: x[0], reverse=True)
                _, target_name, target_layer = candidates[0]

        if target_layer is not None:
            self.pfgr_module = PFGR_Module(target_layer, lambda_scale=lambda_scale).to(self.device)
            try:
                _set_submodule(point_encoder, target_name, self.pfgr_module)
                self._pfgr_target_path = target_name
                logger.info(f"[PFGR] Installed to: {target_name}, λ={lambda_scale}")
            except Exception as e:
                logger.warning(f"[PFGR] Installation failed: {e}")
                self.pfgr_module = None
                self._pfgr_target_path = None
        else:
            logger.warning("[PFGR] Suitable nn.Linear layer not found, PFGR is not installed.")


    @torch.no_grad()
    def get_current_discrete_token_ids(self):
        token_ids = []
        for i in range(self.n_ctx):
            ctx_vec = self.ctx_cont[i].float()
            ctx_vec = F.normalize(ctx_vec, dim=-1)

            cand_ids = getattr(self, f"cand_ids_{i}")
            cand_emb_norm = getattr(self, f"cand_emb_norm_{i}")

            sim = torch.matmul(cand_emb_norm, ctx_vec)
            idx = int(sim.argmax().item())
            token_ids.append(int(cand_ids[idx].item()))
        return token_ids

    def _get_ctx_for_forward(self):
        discrete_embs = []
        token_ids = []

        for i in range(self.n_ctx):
            ctx_vec = self.ctx_cont[i].float()
            ctx_vec_norm = F.normalize(ctx_vec, dim=-1)

            cand_ids = getattr(self, f"cand_ids_{i}")
            cand_emb = getattr(self, f"cand_emb_{i}")
            cand_emb_norm = getattr(self, f"cand_emb_norm_{i}")

            sim = torch.matmul(cand_emb_norm, ctx_vec_norm)
            idx = sim.argmax()

            token_ids.append(cand_ids[idx])
            discrete_embs.append(cand_emb[idx].to(self.ctx_cont.dtype))

        ctx_discrete = torch.stack(discrete_embs, dim=0)
        token_ids = torch.stack(token_ids, dim=0)


        ctx_pez = self.ctx_cont + (ctx_discrete - self.ctx_cont).detach()
        ctx_all = torch.cat([self.ctx_soft, ctx_pez], dim=0)
        return ctx_all, token_ids


    def get_text_feat(self):
        if self.use_pez_text:
            ctx_for_forward, _ = self._get_ctx_for_forward()
        else:
            ctx_for_forward = torch.cat([self.ctx_soft, self.ctx_cont], dim=0)

        text_features = encode_text_with_prompt(self.text_tower,
            ctx_for_forward,
            self.tokenized_prompts,
            ctx_positions=self.ctx_positions
        )
        if text_features.device != self.pc_device:
            text_features = text_features.to(self.pc_device, non_blocking=True)
        return text_features

    def get_point_prompt_xyz(self):
        if self.post_fps_prompt is None:
            return None
        return self.post_fps_prompt.prompt_xyz

    def using_post_fps_prompt(self) -> bool:
        return self.post_fps_prompt is not None and bool(self.post_fps_prompt.hook_attached)

    def get_legacy_prompt(self):
        if self.legacy_prompt is None:
            return None
        return self.legacy_prompt.detach()

    def get_multi_task_loss(self, loss_image, loss_point, loss_text):
        log_vars_clamped = torch.clamp(self.log_vars, min=self.min_log_var, max=self.max_log_var)
        loss = (
            torch.exp(-log_vars_clamped[0]) * loss_image + log_vars_clamped[0] +
            torch.exp(-log_vars_clamped[1]) * loss_point + log_vars_clamped[1] +
            torch.exp(-log_vars_clamped[2]) * loss_text + log_vars_clamped[2]
        )
        self.log_vars_clamped = log_vars_clamped
        return loss

    def save_prompt(self, path: str):

        if self.pfgr_module is not None:
            torch.save(self.pfgr_module.pfgr_branch.state_dict(), path)
            logger.info(f"PFGR weights saved to: {path}")

        prefix = path[:-3] if path.endswith(".pt") else path


        torch.save(self.ctx_cont.detach().cpu(), prefix + "_ctx_cont.pt")
        torch.save(self.ctx_soft.detach().cpu(), prefix + "_ctx_soft.pt")

        try:
            token_ids = self.get_current_discrete_token_ids()
            torch.save(torch.tensor(token_ids, dtype=torch.long), prefix + "_ctx_tokens.pt")
            decoded_words = [tokenizer.decode([tid]) for tid in token_ids]
            with open(prefix + "_words.txt", "w") as f:
                f.write(str(decoded_words))
            logger.info(f"Current hard prompt words: {decoded_words}")
        except Exception as e:
            logger.warning(f"Failed to save/decode ctx_tokens: {e}")


        if self.post_fps_prompt is not None:
            prompt_xyz = self.post_fps_prompt.prompt_xyz.detach().cpu()
            np.save(prefix + "_point_prompt.npy", prompt_xyz.numpy())
            torch.save(prompt_xyz, prefix + "_point_prompt.pt")
            save_point_prompt_visualization(prompt_xyz, prefix + "_point_prompt.png")
            logger.info(f"Point prompt (Post-FPS) saved to: {prefix + '_point_prompt.npy'}")


        if self.legacy_prompt is not None:
            torch.save(self.legacy_prompt.detach().cpu(), prefix + "_legacy_prompt.pt")

    def reparameterize_for_inference(self):
        if self.pfgr_module is None:
            return

        self.pfgr_module.reparameterize()

        if getattr(self, '_pfgr_target_path', None):
            try:
                point_encoder = _guess_point_encoder(self.student_model)
                _set_submodule(point_encoder, self._pfgr_target_path, self.pfgr_module.original_layer)
                logger.info(f"[PFGR] Replaced {self._pfgr_target_path} back to the merged original linear layer")
                self.pfgr_module = None
            except Exception as e:
                logger.warning(f"[PFGR] Failed to replace back to original layer: {e}")

    def encode_pc_only(self, point_clouds: torch.Tensor) -> torch.Tensor:
        pc_input = ensure_xyzrgb_torch(point_clouds).contiguous().float()


        with autocast(enabled=False):
            if self.using_post_fps_prompt():
                pc_embed = self.student_model.encode_pc(pc_input)
            else:
                if self.legacy_prompt is not None:
                    B = pc_input.shape[0]
                    prompt = self.legacy_prompt.expand(B, -1, -1)
                    pc_embed = self.student_model.encode_pc(pc_input, prompt)
                else:
                    pc_embed = self.student_model.encode_pc(pc_input)
        return pc_embed

    def forward(self, point_clouds: torch.Tensor):

        if self.use_pez_text:
            ctx_for_forward, _ = self._get_ctx_for_forward()
        else:
            ctx_for_forward = torch.cat([self.ctx_soft, self.ctx_cont], dim=0)

        text_features = encode_text_with_prompt(
            self.text_tower,
            ctx_for_forward,
            self.tokenized_prompts,
            ctx_positions=self.ctx_positions
        )

        if text_features.device != self.pc_device:
            text_features = text_features.to(self.pc_device, non_blocking=True)


        pc_input = ensure_xyzrgb_torch(point_clouds).contiguous().float()


        with autocast(enabled=False):
            if self.using_post_fps_prompt():
                pc_embed = self.student_model.encode_pc(pc_input)
            else:

                if self.legacy_prompt is not None:
                    B = pc_input.shape[0]
                    prompt = self.legacy_prompt.expand(B, -1, -1)
                    pc_embed = self.student_model.encode_pc(pc_input, prompt)
                else:
                    pc_embed = self.student_model.encode_pc(pc_input)

        return pc_embed, text_features


@torch.no_grad()
def test_attack(
    model,
    epsilon,
    text_features=None,
    batch_text_features=None,
    prompt=None,
):
    if batch_text_features is None:
        batch_text_features = text_features

    correct = 0
    total = 0
    model.eval()
    try:
        _dev = next(model.parameters()).device
    except StopIteration:
        _dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    if not hasattr(test_attack, "_saved_first_perturbed"):
        setattr(test_attack, "_saved_first_perturbed", False)


    for data in test_loader:
        labels = data[1].to(_dev)
        perturbed_data = data[2].to(_dev)
        perturbed_data = ensure_xyzrgb_torch(perturbed_data).contiguous()


        if getattr(test_attack, "_saved_first_perturbed", False) is False:
            try:
                first_pc = perturbed_data[0].detach().cpu().numpy()
                np.save(str(save_dir / 'XXX'), first_pc)
                logger.info("Saved the first point cloud of batch 0 to XXX")
            except Exception as e:
                logger.warning(f"Failed to save XXX: {e}")
            setattr(test_attack, "_saved_first_perturbed", True)

        if prompt is not None:
            if prompt.dim() == 3 and prompt.shape[0] != perturbed_data.shape[0]:
                prompt_batch = prompt.expand(perturbed_data.shape[0], -1, -1)
            else:
                prompt_batch = prompt
            point_features = model.encode_pc(perturbed_data, prompt_batch)
        else:
            point_features = model.encode_pc(perturbed_data)

        point_features = F.normalize(point_features, dim=-1)
        similarity = torch.matmul(point_features, batch_text_features.T)
        predictions = similarity.argmax(dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    acc = correct / max(total, 1)
    print(f"Adversarial Test Accuracy: {acc:.4f}")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uni3D Prompt Training (Text PEZ + Point PFGR/Post-FPS)")


    parser.add_argument("--pretrained_pc", type=str, default=None, help="Path to pretrained point cloud backbone (optional); kept for create_uni3d compatibility")


    parser.add_argument("--cuda_visible_devices", type=str, default=None,
                        help="Override CUDA_VISIBLE_DEVICES early (e.g., '0' or '0,1'). Must be set before torch import.")
    parser.add_argument("--student_device", type=str, default="cuda:0", help="Device for student model and training (default: cuda:0)")
    parser.add_argument("--point_teacher_device", type=str, default="cuda:0", help="Device for point teacher encode_pc (default: cuda:0)")
    parser.add_argument("--visual_device", type=str, default="cuda:1", help="Device for CLIP visual teacher (default: cuda:1)")
    parser.add_argument("--text_device", type=str, default=None, help="Device for CLIP text tower + text prompt (default: same as student_device to avoid OOM on visual GPU)")

    parser.add_argument("--teacher_pc_model", type=str, default="eva02_base_patch14_448")
    parser.add_argument("--teacher_pc_feat_dim", type=int, default=768)
    parser.add_argument("--teacher_ckpt_path", type=str, default="")

    parser.add_argument("--student_pc_model", type=str, default="eva02_tiny_patch14_224")
    parser.add_argument("--student_pc_feat_dim", type=int, default=192)
    parser.add_argument("--student_ckpt_path", type=str, default="")

    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--pc_encoder_dim", type=int, default=512)
    parser.add_argument("--num_group", type=int, default=512)
    parser.add_argument("--group_size", type=int, default=64)


    parser.add_argument("--dataname", type=str, default="Modelnet40", choices=["Modelnet40", "ScanObjectNN"])
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--modelnet_root", type=str, default='XXX')
    parser.add_argument("--train_npz", type=str, default='XXX')
    parser.add_argument("--test_npz", type=str, default='XXX')
    parser.add_argument("--scan_train_h5", type=str, default='XXX')
    parser.add_argument("--scan_train_npz", type=str, default='XXX')
    parser.add_argument("--scan_test_npz", type=str, default='XXX')


    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)


    parser.add_argument("--clean_warmup_epochs", type=int, default=7,
                        help="Use clean points only for the first N epochs.")
    parser.add_argument("--adv_mix_max_prob", type=float, default=1.0,
                        help="Max probability to sample adversarial points per batch sample.")
    parser.add_argument("--adv_mix_ramp_epochs", type=int, default=10,
                        help="Epochs to ramp adv mix probability to max.")

    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])


    parser.add_argument("--pfgr_lambda", type=float, default=1.0)
    parser.add_argument("--pfgr_warmup_epochs", type=int, default=5)
    parser.add_argument("--pfgr_ramp_epochs", type=int, default=10)


    parser.add_argument("--image_loss_warmup_epochs", type=int, default=5)
    parser.add_argument("--selective_topk", type=int, default=5)
    parser.add_argument("--image_loss_ramp_epochs", type=int, default=10)
    parser.add_argument("--gate_update", type=str, default="fixed", choices=["fixed", "ema"])
    parser.add_argument("--gate_ema", type=float, default=0.995)


    parser.add_argument("--point_loss_mode", type=str, default="cosine", choices=["mse", "cosine", "combined"])
    parser.add_argument("--loss_point_scale", type=float, default=200.0)


    parser.add_argument("--point_contrast_weight", type=float, default=1.0,
                        help="point contrastive loss weight (before point_scale_now).")
    parser.add_argument("--point_geom_weight", type=float, default=0.1,
                        help="geometric alignment auxiliary weight (0=disable, small weight recommended for regularization).")
    parser.add_argument("--point_selective_topk", type=int, default=5,
                        help="topk for point selective alignment.")
    parser.add_argument("--point_selective_warmup_epochs", type=int, default=5,
                        help="point selective starts from N-th epoch.")
    parser.add_argument("--point_selective_ramp_epochs", type=int, default=10,
                        help="ramp-up epoch count for point selective mix.")

    parser.add_argument("--point_loss_warmup_epochs", type=int, default=5)
    parser.add_argument("--point_loss_ramp_epochs", type=int, default=10)


    parser.add_argument("--use_clean_for_text", action="store_true",
                        help="Also compute text CE using clean points (in addition to adversarial points).")
    parser.add_argument("--text_clean_weight", type=float, default=0.5,
                        help="Weight of clean-text CE when --use_clean_for_text is enabled. final= (1-w)*adv + w*clean")
    parser.add_argument("--text_anchor_weight", type=float, default=0.2,
                        help="Regularize learned text prototypes to stay close to fixed (no-X) CLIP text prototypes.")
    parser.add_argument("--freeze_point_epochs", type=int, default=0,
                        help="Freeze point-side prompt/PFGR for the first N epochs (text-only warmup).")


    parser.add_argument("--text_prompt_size", type=int, default=3)
    parser.add_argument("--soft_prompt_size", type=int, default=3)
    parser.add_argument("--disable_pez", action="store_true")


    parser.add_argument("--point_prompt_len", type=int, default=10)


    parser.add_argument("--min_log_var", type=float, default=-0.8)
    parser.add_argument("--max_log_var", type=float, default=2.0)


    parser.add_argument("--enable_legacy_prompt_fallback", action="store_true")
    parser.add_argument("--legacy_prompt_feat_dim", type=int, default=None)


    parser.add_argument("--eval_epsilon", type=float, default=0.05)


    parser.add_argument("--open_clip_pretrained_bin", type=str, default='XXX')
    parser.add_argument("--open_clip_model", type=str, default="EVA02-E-14-plus")

    args = parser.parse_args()


    set_seed(int(args.seed))


    DEVICE_STUDENT = torch.device(args.student_device)
    DEVICE_VISUAL = torch.device(args.visual_device) if torch.cuda.device_count() >= 2 else DEVICE_STUDENT
    DEVICE_POINT_TEACHER = torch.device(args.point_teacher_device) if args.point_teacher_device is not None else DEVICE_STUDENT
    DEVICE_TEXT = torch.device(args.text_device) if (getattr(args, "text_device", None) not in (None, "", "none", "None")) else DEVICE_STUDENT
    if torch.cuda.is_available():
        logger.info(f"CUDA device_count={torch.cuda.device_count()} | visible={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
        if torch.cuda.device_count() < 2:
            logger.warning("Only 1 GPU is visible, replacing visual/text back to cuda:0. If dual GPUs are needed, ensure CUDA_VISIBLE_DEVICES exposes two cards.")
    logger.info(f"Devices | student={DEVICE_STUDENT} | point_teacher={DEVICE_POINT_TEACHER} | visual={DEVICE_VISUAL} | text={DEVICE_TEXT}")


    AMP_ENABLED = False
    AMP_DTYPE = torch.float16 if args.amp_dtype == 'fp16' else torch.bfloat16
    scaler = GradScaler(enabled=AMP_ENABLED)
    logger.info(f"AMP enabled: {AMP_ENABLED} | dtype={args.amp_dtype}")


    logger.info("Loading dataset...")

    global test_loader
    if args.dataname == "Modelnet40":
        orgin_train_dataset = ModelNet40Dataset(
            root_dir=args.modelnet_root,
            split='modelnet40_train',
            transform=transform
        )
        train_dataset = NpzPointCloudDataset(args.train_npz, classnames=orgin_train_dataset.classnames)
        test_dataset = NpzPointCloudDataset(args.test_npz, classnames=orgin_train_dataset.classnames)

    else:
        orgin_train_dataset = ScanObjectNNDataset(h5_path=args.scan_train_h5, transform=transform)
        train_dataset = NpzPointCloudDataset(args.scan_train_npz, classnames=orgin_train_dataset.classnames)
        test_dataset = NpzPointCloudDataset(args.scan_test_npz, classnames=orgin_train_dataset.classnames)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    logger.info("Dataset loaded.")


    logger.info("Loading CLIP Text/Visual models...")
    clip_model_text, _, _ = open_clip.create_model_and_transforms_text(
        args.open_clip_model,
        pretrained=args.open_clip_pretrained_bin,
        device="cpu"
    )
    clip_model_text = clip_model_text.to(DEVICE_TEXT)
    clip_model_text.eval()

    clip_model_visual, _, _ = open_clip.create_model_and_transforms_visual(
        args.open_clip_model,
        pretrained=args.open_clip_pretrained_bin,
        device="cpu"
    )
    clip_model_visual = clip_model_visual.to(DEVICE_VISUAL)
    clip_model_visual.eval()
    gc.collect()
    logger.info("CLIP Text/Visual teacher initialization completed.")


    with torch.no_grad():


        classnames_text = [
        f"A depth picture of a {name.replace('_', ' ')}."
        for name in train_dataset.classnames
        ]
        text_inputs = tokenizer(classnames_text).to(DEVICE_TEXT)
        base_text_features = clip_model_text.encode_text(text_inputs)
        base_text_features = F.normalize(base_text_features, dim=-1)
        if base_text_features.device != DEVICE_STUDENT:
            base_text_features = base_text_features.to(DEVICE_STUDENT, non_blocking=True)


        text_features = base_text_features
        gate_text_features = base_text_features.detach().clone()


    logger.info("Loading point cloud teacher model (Uni3D Teacher)...")
    teacher_args = copy.deepcopy(args)
    teacher_args.pc_model = args.teacher_pc_model
    teacher_args.pc_feat_dim = args.teacher_pc_feat_dim
    teacher_args.ckpt_path = args.teacher_ckpt_path

    if not hasattr(teacher_args, 'pretrained_pc'):
        teacher_args.pretrained_pc = ''

    point_teacher = create_uni3d(teacher_args)
    if os.path.exists(teacher_args.ckpt_path):
        checkpoint = torch.load(teacher_args.ckpt_path, map_location='cpu')
        sd = checkpoint.get('module', checkpoint)
        if isinstance(sd, dict) and len(sd) > 0 and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        point_teacher.load_state_dict(sd, strict=False)
        logger.info(f"Teacher ckpt loaded from {teacher_args.ckpt_path}")
    else:
        logger.warning(f"[Warning] Teacher checkpoint not found at {teacher_args.ckpt_path}")

    point_teacher = point_teacher.to(DEVICE_POINT_TEACHER)
    point_teacher.eval()


    logger.info("Loading point cloud student model (Uni3D Student)...")
    student_args = copy.deepcopy(args)
    student_args.pc_model = args.student_pc_model
    student_args.pc_feat_dim = args.student_pc_feat_dim
    student_args.ckpt_path = args.student_ckpt_path

    if not hasattr(student_args, 'pretrained_pc'):
        student_args.pretrained_pc = ''

    model_student = create_uni3d(student_args)
    if os.path.exists(student_args.ckpt_path):
        checkpoint = torch.load(student_args.ckpt_path, map_location='cpu')
        sd = checkpoint.get('module', checkpoint)
        if isinstance(sd, dict) and len(sd) > 0 and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        model_student.load_state_dict(sd, strict=False)
        logger.info(f"Student ckpt loaded from {student_args.ckpt_path}")
    else:
        logger.warning(f"[Warning] Student checkpoint not found at {student_args.ckpt_path}")

    model_student = model_student.to(DEVICE_STUDENT)
    model_student.eval()


    logger.info("Evaluating student model adversarial accuracy before prompt learning...")
    base_acc = test_attack(model_student, args.eval_epsilon, text_features=text_features, batch_text_features=text_features, prompt=None)
    logger.info(f"Before prompt | Adv acc = {base_acc:.4f}")


    logger.info("Building new Uni3D Prompt...")
    student_with_prompt = Uni3DPrompt(
        student_model=model_student,
        clip_model_text=clip_model_text,
        classnames=train_dataset.classnames,
        device=str(DEVICE_STUDENT),
        text_device=str(DEVICE_TEXT),
        text_prompt_size=args.text_prompt_size,
        soft_prompt_size=args.soft_prompt_size,
        use_pez_text=(not args.disable_pez),
        point_prompt_len=args.point_prompt_len,
        pfgr_lambda=args.pfgr_lambda,
        min_log_var=args.min_log_var,
        max_log_var=args.max_log_var,
        enable_legacy_prompt_fallback=args.enable_legacy_prompt_fallback,
        legacy_prompt_feat_dim=args.legacy_prompt_feat_dim,
    )

    trainable_params = sum(p.numel() for p in student_with_prompt.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student_with_prompt.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / Total parameters: {total_params:,}")
    logger.info(f"Post-FPS Prompt attached: {student_with_prompt.using_post_fps_prompt()}")


    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student_with_prompt.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=1e-5)

    best_accuracy = 0.0

    for epoch in range(int(args.epochs)):


        def _set_point_trainable(trainable: bool):
            if getattr(student_with_prompt, "post_fps_prompt", None) is not None:
                student_with_prompt.post_fps_prompt.prompt_xyz.requires_grad_(trainable)
            if getattr(student_with_prompt, "legacy_prompt", None) is not None:
                student_with_prompt.legacy_prompt.requires_grad_(trainable)
            if getattr(student_with_prompt, "pfgr_module", None) is not None:
                for p in student_with_prompt.pfgr_module.pfgr_branch.parameters():
                    p.requires_grad_(trainable)

        if int(args.freeze_point_epochs) > 0 and epoch < int(args.freeze_point_epochs):
            _set_point_trainable(False)
        else:
            _set_point_trainable(True)


        warmup_p = max(int(args.point_loss_warmup_epochs), 0)
        ramp_p = max(int(args.point_loss_ramp_epochs), 0)
        target_p = float(args.loss_point_scale)

        if epoch < warmup_p:
            point_scale_now = 0.0
        else:
            if ramp_p <= 0:
                point_scale_now = target_p
            else:
                prog = (epoch - warmup_p + 1) / float(ramp_p)
                prog = float(min(max(prog, 0.0), 1.0))
                point_scale_now = target_p * prog

        logger.info(f"[PointLoss] epoch={epoch} | scale={point_scale_now:.3f} (target={target_p:.3f}, warmup={warmup_p}, ramp={ramp_p})")


        if student_with_prompt.pfgr_module is not None:
            warmup = max(int(args.pfgr_warmup_epochs), 0)
            ramp = max(int(args.pfgr_ramp_epochs), 0)
            target = float(args.pfgr_lambda)

            if epoch < warmup:
                lambda_now = 0.0
            else:
                if ramp <= 0:
                    lambda_now = target
                else:
                    progress = (epoch - warmup + 1) / float(ramp)
                    progress = float(min(max(progress, 0.0), 1.0))
                    lambda_now = target * progress

            student_with_prompt.pfgr_module.lambda_scale = float(lambda_now)
            logger.info(f"[PFGR] epoch={epoch} | λ={lambda_now:.6f} (target={target:.6f}, warmup={warmup}, ramp={ramp})")

        student_with_prompt.train()

        total_loss = 0.0
        total_loss_image = 0.0
        total_loss_point = 0.0
        total_loss_text = 0.0
        num_batches = 0

        gating_ratios = []
        alpha_imgs = []

        for data in train_loader:
            labels = data[1].to(DEVICE_STUDENT, non_blocking=True)


            points_clean_student = data[0].to(DEVICE_STUDENT, non_blocking=True)
            points_adv_student = data[2].to(DEVICE_STUDENT, non_blocking=True)

            if epoch < int(args.clean_warmup_epochs):
                points_student = points_clean_student
                p_adv = 0.0
            else:
                max_p = float(args.adv_mix_max_prob)
                max_p = float(min(max(max_p, 0.0), 1.0))

                ramp = max(int(args.adv_mix_ramp_epochs), 0)
                if ramp <= 0:
                    p_adv = max_p
                else:
                    prog = (epoch - int(args.clean_warmup_epochs) + 1) / float(ramp)
                    prog = float(min(max(prog, 0.0), 1.0))
                    p_adv = max_p * prog

                if p_adv <= 0.0:
                    points_student = points_clean_student
                elif p_adv >= 1.0:
                    points_student = points_adv_student
                else:

                    mask = (torch.rand(points_clean_student.shape[0], device=points_clean_student.device) < p_adv).view(-1, 1, 1)
                    points_student = torch.where(mask, points_adv_student, points_clean_student)

            points_student = ensure_xyzrgb_torch(points_student).contiguous()


            points_teacher_clean = data[0]
            points_teacher_clean = ensure_xyzrgb_torch(points_teacher_clean)


            imgs = process_point_cloud_batch(points_teacher_clean[:, :, :3])
            with torch.no_grad():

                image_features = clip_model_visual.encode_image(imgs.to(DEVICE_VISUAL)).detach().to(DEVICE_STUDENT)
                image_features = F.normalize(image_features, dim=-1)


                pts_t = points_teacher_clean.to(DEVICE_POINT_TEACHER, non_blocking=True).contiguous().float()
                if (not torch.isfinite(pts_t).all()):
                    logger.warning("Input point cloud contains NaN/Inf, skipping this batch")
                    continue
                with autocast(enabled=False):
                    point_teacher_features = point_teacher.encode_pc(pts_t).detach()

                point_teacher_features = point_teacher_features.to(DEVICE_STUDENT, non_blocking=True)
                point_teacher_features = F.normalize(point_teacher_features, dim=-1)
                del pts_t
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE):
                student_features, all_text_features = student_with_prompt(points_student)
                student_features = F.normalize(student_features, dim=-1)
                all_text_features = F.normalize(all_text_features, dim=-1)


                if epoch >= int(args.image_loss_warmup_epochs) and args.gate_update == "ema":
                    with torch.no_grad():
                        gate_text_features = gate_text_features * float(args.gate_ema) + (1.0 - float(args.gate_ema)) * all_text_features.detach()
                        gate_text_features = F.normalize(gate_text_features, dim=-1)


                tau_img = student_with_prompt.log_tau[0].exp()
                loss_image_full = contrastive_loss(student_features, image_features, tau=tau_img)

                warmup_img = int(args.image_loss_warmup_epochs)
                if epoch < warmup_img:
                    loss_image = loss_image_full
                    alpha_img = 0.0
                    gating_ratios.append(1.0)
                    alpha_imgs.append(alpha_img)
                else:
                    loss_image_sel, correct_ratio = selective_contrastive_loss(
                        student_features,
                        image_features,
                        gate_text_features,
                        labels,
                        tau=tau_img,
                        topk=int(args.selective_topk),
                        return_stats=True,
                    )
                    gating_ratios.append(correct_ratio)

                    ramp = max(int(args.image_loss_ramp_epochs), 1)
                    alpha_img = min(1.0, float(epoch - warmup_img + 1) / float(ramp))
                    alpha_imgs.append(alpha_img)

                    loss_image = (1.0 - alpha_img) * loss_image_full + alpha_img * loss_image_sel


                tau_point = student_with_prompt.log_tau[1].exp()


                loss_point_full = contrastive_loss(student_features, point_teacher_features, tau=tau_point)

                warmup_sel = max(int(args.point_selective_warmup_epochs), 0)
                if epoch < warmup_sel:
                    loss_point_contrast = loss_point_full
                    alpha_p = 0.0
                else:
                    loss_point_sel, sel_ratio = selective_contrastive_loss(
                        student_features,
                        point_teacher_features,
                        gate_text_features,
                        labels,
                        tau=tau_point,
                        topk=int(args.point_selective_topk),
                        return_stats=True,
                    )
                    ramp_sel = max(int(args.point_selective_ramp_epochs), 1)
                    alpha_p = min(1.0, float(epoch - warmup_sel + 1) / float(ramp_sel))
                    loss_point_contrast = (1.0 - alpha_p) * loss_point_full + alpha_p * loss_point_sel

                loss_point_raw = float(args.point_contrast_weight) * loss_point_contrast

                if float(args.point_geom_weight) > 0.0:
                    loss_point_geom = geometric_alignment_loss(
                        student_features,
                        point_teacher_features,
                        mode=args.point_loss_mode
                    )
                    loss_point_raw = loss_point_raw + float(args.point_geom_weight) * loss_point_geom

                loss_point = loss_point_raw * float(point_scale_now)


                tau_text = student_with_prompt.log_tau[2].exp()


                logits_text_adv = torch.matmul(student_features, all_text_features.t()) / tau_text
                loss_text_adv = F.cross_entropy(logits_text_adv, labels)


                if bool(args.use_clean_for_text):
                    points_clean_student = points_teacher_clean.to(DEVICE_STUDENT, non_blocking=True)
                    student_features_clean = student_with_prompt.encode_pc_only(points_clean_student)
                    student_features_clean = F.normalize(student_features_clean, dim=-1)
                    logits_text_clean = torch.matmul(student_features_clean, all_text_features.t()) / tau_text
                    loss_text_clean = F.cross_entropy(logits_text_clean, labels)
                    w_clean = float(args.text_clean_weight)
                    w_clean = float(min(max(w_clean, 0.0), 1.0))
                    loss_text_ce = (1.0 - w_clean) * loss_text_adv + w_clean * loss_text_clean
                else:
                    loss_text_ce = loss_text_adv


                if float(args.text_anchor_weight) > 0.0:
                    loss_text_anchor = (1.0 - (all_text_features * base_text_features).sum(dim=-1)).mean()
                    loss_text = loss_text_ce + float(args.text_anchor_weight) * loss_text_anchor
                else:
                    loss_text = loss_text_ce

                loss = student_with_prompt.get_multi_task_loss(loss_image, loss_point, loss_text)

            total_loss += float(loss.item())
            total_loss_image += float(loss_image.item())
            total_loss_point += float(loss_point.item())
            total_loss_text += float(loss_text.item())
            num_batches += 1

            if AMP_ENABLED:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_with_prompt.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_with_prompt.parameters(), max_norm=5.0)
                optimizer.step()


        avg_loss = total_loss / max(num_batches, 1)
        avg_loss_image = total_loss_image / max(num_batches, 1)
        avg_loss_point = total_loss_point / max(num_batches, 1)
        avg_loss_text = total_loss_text / max(num_batches, 1)

        if len(gating_ratios) > 0:
            logger.info(f"selective gating ratio (avg): {float(np.mean(gating_ratios)):.4f}")
            logger.info(f"image selective mix alpha (avg): {float(np.mean(alpha_imgs)):.4f}")

        if student_with_prompt.log_vars_clamped is not None:
            sigmas = torch.exp(student_with_prompt.log_vars_clamped / 2).detach().cpu().numpy()
            logger.info(f"Learned sigmas: image={sigmas[0]:.4f}, point={sigmas[1]:.4f}, text={sigmas[2]:.4f}")
            taus = student_with_prompt.log_tau.detach().exp().cpu().numpy()
            logger.info(f"taus: image={taus[0]:.4f}, point={taus[1]:.4f}, text={taus[2]:.4f}")

        scheduler.step()
        logger.info(f"Epoch {epoch}: total={avg_loss:.4f}, image={avg_loss_image:.4f}, point={avg_loss_point:.4f}, text={avg_loss_text:.4f}")


        student_with_prompt.eval()
        with torch.no_grad():
            eval_text_features = student_with_prompt.get_text_feat()
            eval_text_features = F.normalize(eval_text_features, dim=-1)


        eval_prompt = None
        if (not student_with_prompt.using_post_fps_prompt()) and (student_with_prompt.legacy_prompt is not None):
            eval_prompt = student_with_prompt.get_legacy_prompt()

        accuracy = test_attack(
            model_student,
            args.eval_epsilon,
            text_features=text_features,
            batch_text_features=eval_text_features,
            prompt=eval_prompt,
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            student_with_prompt.save_prompt(str(save_dir / f"{task}_best_prompt{suffix}.pt"))
            logger.info(f"Best prompt saved | acc={best_accuracy:.4f}")

        student_with_prompt.save_prompt(str(save_dir / f"{task}_latest_prompt{suffix}.pt"))


        try:
            del image_features, point_teacher_features, student_features, eval_text_features, points_student, labels
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"[Epoch {epoch}] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB | Max: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")


    logger.info("Training completed, executing PFGR reparameterization to achieve Zero Inference Overhead...")
    student_with_prompt.reparameterize_for_inference()

    final_model_path = str(save_dir / f"{task}_final_reparameterized{suffix}.pt")
    torch.save(model_student.state_dict(), final_model_path)
    logger.info(f"Reparameterized student model saved to: {final_model_path}")
