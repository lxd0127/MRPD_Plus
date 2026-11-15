import os
from pathlib import Path
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.ULIP_models import ULIP2_WITH_OPENCLIP
from models.pointbert.point_encoder import PointTransformer, PointTransformer_Colored
from utils.utils import cfg_from_yaml_file
import utils.open_clip as open_clip
from loguru import logger
import psutil
import gc
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from utils.projection import process_point_cloud_batch
from utils.tokenizer import SimpleTokenizer
from models.point_encoder import fps6d
import h5py
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

task = "MN_s"

script_name = Path(__file__).stem
suffix = f"_{script_name}"
save_dir = Path('XXX')
save_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Current task: {task}")
tokenizer = SimpleTokenizer()

def ensure_xyzrgb_np(pc: np.ndarray) -> np.ndarray:
    pc = np.asarray(pc)
    if pc.ndim not in (2, 3):
        raise ValueError(f"got shape={pc.shape}")
    if pc.shape[-1] < 3:
        raise ValueError(f"got shape={pc.shape}")
    xyz = pc[..., :3].astype(np.float32, copy=False)
    rgb = np.full(xyz.shape[:-1] + (3,), 0.5, dtype=np.float32)
    return np.concatenate([xyz, rgb], axis=-1)

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
                f"actual keys={self.data.files}"
            )
        self.classnames = classnames

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        point = torch.tensor(self.points[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        original = torch.tensor(self.orginal_points[idx], dtype=torch.float32)
        point = ensure_xyzrgb_torch(point)
        original = ensure_xyzrgb_torch(original)
        return original, label, point


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
            raise FileNotFoundError(f"Category names file not found: {classnames_path}")

        split_file = os.path.join(root_dir, f'{split}.txt')
        with open(split_file, 'r') as f:
            file_list = [line.strip() for line in f.readlines()]

        for item in file_list:
            category = item.rsplit('_', 1)[0]
            file_path = os.path.join(root_dir, category, f'{item}.txt')
            if not os.path.exists(file_path):
                print(f'File not found: {file_path}')
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


def print_memory(tag):
    cpu_mem = psutil.virtual_memory()
    print(f"[{tag}] CPU Memory Usage: {(cpu_mem.total - cpu_mem.available) / (1024**2):.2f} MB / {cpu_mem.total / (1024**2):.2f} MB")


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
    if prompt_xyz.ndim == 3:
        prompt_xyz = prompt_xyz[0]

    if prompt_xyz.shape[-1] > 3:
        prompt_xyz = prompt_xyz[:, :3]

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
    lambda pc: np.hstack((pc[:, :3], np.full((pc.shape[0], 3), 0.5))),
    lambda pc: torch.tensor(pc, dtype=torch.float32)
])


def cos_loss(point_features, text_features):
    point_features = F.normalize(point_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    cosine_sim = F.cosine_similarity(point_features, text_features, dim=-1)
    return -cosine_sim.mean()


def contrastive_loss(point_features, text_features, tau=0.07):
    B = point_features.size(0)
    logits = torch.matmul(point_features, text_features.t()) / tau
    labels = torch.arange(B, device=point_features.device, dtype=torch.long)
    loss_p2t = F.cross_entropy(logits, labels)
    loss_t2p = F.cross_entropy(logits.t(), labels)
    return (loss_p2t + loss_t2p) * 0.5


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


def test_attack(model, epsilon, text_features=None, batch_text_features=None, prompt=None, pfgr_module=None):

    if batch_text_features is not None:
        logger.info(f"batch_text_features shape: {batch_text_features.shape}")
        if text_features is not None:
            logger.info(f"text_features shape: {text_features.shape}")
    else:
        batch_text_features = text_features

    amp_enabled = globals().get("AMP_ENABLED", False)
    amp_dtype = globals().get("AMP_DTYPE", torch.float16)

    correct = 0
    total = 0
    model.eval()
    save = True

    for data in test_loader:
        logger.info("Generating perturbed data...")
        perturbed_data = data[2].cuda(0)

        if save:
            first_point_cloud = perturbed_data[0].cpu().detach().numpy()
            np.save(str(save_dir / 'XXX'), first_point_cloud)
            print("Saved the first point cloud of batch 0 to XXX!")
            save = False

        logger.info("Start evaluation")
        with torch.no_grad():
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                point_clouds = ensure_xyzrgb_torch(perturbed_data).contiguous()
                point_features = model.encode_pc(point_clouds)
                point_features = point_features / point_features.norm(dim=-1, keepdim=True)

                similarity_point = torch.matmul(point_features, batch_text_features.T)
                predictions = similarity_point.argmax(dim=-1)

            logger.info(f"Prediction results: {predictions}")

            correct += (predictions == data[1].to('cuda:0')).sum().item()
            total += data[1].size(0)

    accuracy = correct / total
    print(f"Adversarial test accuracy: {accuracy:.4f}")

    del perturbed_data, point_features, batch_text_features, predictions
    torch.cuda.empty_cache()
    gc.collect()
    return accuracy


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


def encode_text_with_prompt(clip_model, prompt_ctx, tokenized_prompts, ctx_positions=None):
    transformer = clip_model.transformer
    cast_dtype = transformer.get_cast_dtype() if hasattr(transformer, "get_cast_dtype") else prompt_ctx.dtype

    if prompt_ctx.dim() == 2:
        prompt_ctx = prompt_ctx.unsqueeze(0).expand(tokenized_prompts.shape[0], -1, -1)
    prompt_ctx = prompt_ctx.to(device=tokenized_prompts.device, dtype=cast_dtype)

    with torch.no_grad():
        x = clip_model.token_embedding(tokenized_prompts).to(device=tokenized_prompts.device, dtype=cast_dtype)

    n_ctx = prompt_ctx.shape[1]
    if ctx_positions is None:
        ctx_positions = torch.arange(1, 1 + n_ctx, device=tokenized_prompts.device, dtype=torch.long)
    else:
        ctx_positions = ctx_positions.to(device=tokenized_prompts.device, dtype=torch.long)

    x[:, ctx_positions, :] = prompt_ctx

    seq_len = x.shape[1]
    positional = clip_model.positional_embedding[:seq_len].to(device=tokenized_prompts.device, dtype=cast_dtype)
    x = x + positional

    attn_mask = getattr(clip_model, "attn_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask[:seq_len, :seq_len].to(device=tokenized_prompts.device, dtype=cast_dtype)

    x = transformer(x, attn_mask=attn_mask)
    x = clip_model.ln_final(x)

    pooled = x[torch.arange(x.shape[0], device=tokenized_prompts.device), tokenized_prompts.argmax(dim=-1)]

    text_projection = getattr(clip_model, "text_projection", None)
    if text_projection is not None:
        if isinstance(text_projection, nn.Linear):
            pooled = text_projection(pooled)
        else:
            pooled = pooled @ text_projection
    return pooled


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


        nn.init.normal_(self.pfgr_branch.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            weight_norm = self.pfgr_branch.weight.data.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            self.pfgr_branch.weight.data = (self.pfgr_branch.weight.data / weight_norm) * 0.1

        logger.info(f"[PFGR] Initialization complete: in={in_features}, out={out_features}, λ={lambda_scale}")

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

class PostFPSPointPrompt(nn.Module):

    def __init__(self, prompt_len=4, init_scale=0.03, device='cuda:0'):
        super().__init__()
        self.prompt_len = int(prompt_len)
        self.prompt_xyz = nn.Parameter(torch.randn(1, self.prompt_len, 3, device=device) * float(init_scale))
        self.runtime_scale = 0.0
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

        if self.prompt_len <= 0 or float(self.runtime_scale) <= 0.0:
            return output

        B = int(center.shape[0])
        P = int(self.prompt_xyz.shape[1])
        prompt_xyz = (
            self.prompt_xyz.to(device=center.device, dtype=center.dtype) * float(self.runtime_scale)
        ).expand(B, -1, -1)

        Cc = int(center.shape[-1])
        if Cc == 3:
            prompt_center = prompt_xyz
        elif Cc < 3:
            prompt_center = prompt_xyz[..., :Cc]
        else:
            pad_dims = Cc - 3
            pad = torch.zeros((B, P, pad_dims), device=center.device, dtype=center.dtype)
            prompt_center = torch.cat([prompt_xyz, pad], dim=-1)

        S = int(neighborhood.shape[2])
        C = int(neighborhood.shape[3])
        prompt_neighborhood = torch.zeros((B, P, S, C), device=neighborhood.device, dtype=neighborhood.dtype)

        neighborhood_new = torch.cat([neighborhood, prompt_neighborhood], dim=1)
        center_new = torch.cat([center, prompt_center], dim=1)

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
                lname = name.lower()
                cname = m.__class__.__name__.lower()
                if ('group_divider' in lname) or (cname == 'group'):
                    group_module = m
                    break

        if group_module is None:
            logger.warning("Cannot guarantee Post-FPS insertion point.")
            self.hook_attached = False
            return None

        self.hook_handle = group_module.register_forward_hook(self.forward_hook)
        self.hook_attached = True
        logger.info(f"[Post-FPS Prompt] Attached to module: {group_module.__class__.__name__}")
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


class PointPrompt(nn.Module):
    def __init__(self,
                 prompt_len=10,
                 feat_dim=192,
                 student_model=None,
                 device='cuda:0',
                 ckpt_path=None,
                 text_prompt_size=7,
                 soft_prompt_size=3,
                 classnames=None,
                 clip_model_text=None,

                 use_pez_text=True,
                 attribute_seeds=None,
                 attribute_pool=None,

                 pfgr_lambda=1.0,
                 min_log_var=-5.0,
                 max_log_var=3.0):
        super().__init__()
        self.device = device
        self.log_tau = nn.Parameter(torch.tensor([0.07, 0.07, 0.07]).log())


        self.clip_model_text = clip_model_text
        self.clip_model_text.to(device)

        for param in self.clip_model_text.parameters():
            param.requires_grad = False
        self.clip_model_text.eval()


        self.use_pez_text = bool(use_pez_text)
        self.n_ctx = int(text_prompt_size)
        self.n_soft = int(soft_prompt_size)
        self.enforce_unique_hard_prompts = (self.n_ctx == 7)
        if self.enforce_unique_hard_prompts:
            logger.info("[TextPrompt] Enabled hard prompt global deduplication (discrete tokens across slots are unique)")

        assert classnames is not None and len(classnames) > 0, "classnames cannot be empty"
        classnames = [name.replace("_", " ") for name in classnames]


        soft_prefix = " ".join(["X"] * self.n_soft)
        hard_prefix = " ".join(["X"] * self.n_ctx)
        text_prompts = [f"{soft_prefix} A depth picture of a {hard_prefix} {name}." for name in classnames]
        tokenized_prompts = torch.stack([tokenizer(p) for p in text_prompts])
        self.tokenized_prompts = tokenized_prompts.to(device)


        x_id_space = _get_single_token_id("X", add_space=True)
        x_id_nospace = _get_single_token_id("X", add_space=False)

        first = self.tokenized_prompts[0]
        positions = []
        if x_id_space is not None:
            positions.append((first == x_id_space).nonzero(as_tuple=False).squeeze(-1))
        if x_id_nospace is not None and x_id_nospace != x_id_space:
            positions.append((first == x_id_nospace).nonzero(as_tuple=False).squeeze(-1))

        if len(positions) == 0:
            raise RuntimeError("Cannot locate placeholder token 'X', please check if the tokenizer matches the prompt template.")

        positions = torch.cat(positions, dim=0)
        positions = torch.unique(positions).sort()[0]
        total_ctx = self.n_soft + self.n_ctx
        positions = positions[:total_ctx]

        if int(positions.numel()) != total_ctx:
            raise RuntimeError(
                f"Insufficient placeholder X: expected {total_ctx}, but only found {int(positions.numel())}."
                f"Suggestion: check if the template string contains enough 'X' and if the tokenizer's tokenization of 'X' is consistent."
            )

        self.register_buffer("ctx_positions", positions)


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

        if attribute_seeds is None:
            attribute_seeds = ["round", "smooth", "metal"]

        pools = [
            attribute_pool.get("shape", []),
            attribute_pool.get("texture", []),
            attribute_pool.get("material", []),
        ]
        merged_pool = []
        for p in pools:
            merged_pool.extend(p)

        with torch.no_grad():
            token_emb_weight = self.clip_model_text.token_embedding.weight

            for i in range(self.n_ctx):
                if self.n_ctx == 7:
                    if i < 3:
                        pool_idx = 0
                    elif i < 5:
                        pool_idx = 1
                    else:
                        pool_idx = 2
                else:
                    pool_idx = i % len(pools)

                words_i = pools[pool_idx]
                cand_ids_i = _build_candidate_token_ids(words_i)

                if len(cand_ids_i) == 0:
                    logger.warning(f"[TextPrompt] slot-{i}'s attribute_pool has no available single tokens, degrading to full vocabulary projection.")
                    vocab_size = int(token_emb_weight.shape[0])
                    cand_ids_i = list(range(vocab_size))

                    pad_id = 0
                    sot_id = int(self.tokenized_prompts[0, 0].item())
                    eot_id = int(self.tokenized_prompts[0].max().item())
                    cand_ids_i = [tid for tid in cand_ids_i if tid not in (pad_id, sot_id, eot_id)]

                cand_ids_i = torch.tensor(cand_ids_i, dtype=torch.long, device=device)
                emb_i = token_emb_weight.index_select(0, cand_ids_i).float()
                emb_norm_i = F.normalize(emb_i, dim=-1)

                self.register_buffer(f"cand_ids_{i}", cand_ids_i)
                self.register_buffer(f"cand_emb_{i}", emb_i)
                self.register_buffer(f"cand_emb_norm_{i}", emb_norm_i)


        ctx_dim = int(self.clip_model_text.token_embedding.weight.shape[1])
        soft_ctx_vectors = torch.empty(self.n_soft, ctx_dim, device=device)
        nn.init.normal_(soft_ctx_vectors, std=0.05)
        self.ctx_soft = nn.Parameter(soft_ctx_vectors, requires_grad=True)

        init_vecs = []
        with torch.no_grad():
            token_emb_weight = self.clip_model_text.token_embedding.weight
            for i in range(self.n_ctx):
                if self.n_ctx == 7:
                    seeds_7 = ["round", "square", "flat", "smooth", "rough", "metal", "wood"]
                    seed = seeds_7[i]
                else:
                    seed_idx = i % len(attribute_seeds)
                    seed = attribute_seeds[seed_idx]

                seed_id = _get_single_token_id(seed, add_space=True)
                if seed_id is None:
                    seed_id = int(getattr(self, f"cand_ids_{i}")[0].item())
                init_vecs.append(token_emb_weight[seed_id].float())

        ctx_init = torch.stack(init_vecs, dim=0) + torch.randn_like(torch.stack(init_vecs, dim=0)) * 0.01
        self.ctx_cont = nn.Parameter(ctx_init, requires_grad=True)

        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.log_vars_clamped = None


        self.student_model = student_model
        self.student_model.to(device)
        self.student_model.eval()
        for param in self.student_model.parameters():
            param.requires_grad = False


        self.pfgr_module = None
        self._pfgr_target_path = None
        self._install_pfgr(pfgr_lambda)


        self.post_fps_prompt = None
        if prompt_len > 0:
            self.post_fps_prompt = PostFPSPointPrompt(prompt_len=prompt_len, device=device)
            try:
                self.post_fps_prompt.attach(self.student_model.point_encoder)
            except Exception as e:
                logger.warning(f"[Post-FPS Prompt] Mount failed: {e}")

        if ckpt_path is not None and os.path.exists(ckpt_path):
            self._load_checkpoint(ckpt_path)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, 'student_model', None) is not None:
            self.student_model.eval()
        if getattr(self, 'clip_model_text', None) is not None:
            self.clip_model_text.eval()
        return self

    def _install_pfgr(self, lambda_scale):
        point_encoder = self.student_model.point_encoder

        target_layer_names = ['reduce_dim', 'fc', 'proj', 'head']
        target_layer = None
        target_name = None

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

        if target_layer is not None:
            self.pfgr_module = PFGR_Module(target_layer, lambda_scale=lambda_scale).to(self.device)
            _set_submodule(point_encoder, target_name, self.pfgr_module)
            self._pfgr_target_path = target_name
            logger.info(f"[PFGR] Installed at: {target_name}, λ={lambda_scale}")
        else:
            logger.error("[PFGR] Compatible linear layer not found, PFGR not installed!")
            logger.info("[PFGR] Available linear modules:")
            for name, module in point_encoder.named_modules():
                if isinstance(module, nn.Linear):
                    logger.info(f"  - {name}: {module}")

    def _load_checkpoint(self, ckpt_path):
        logger.info(f"Attempting to load checkpoint: {ckpt_path}")
        try:

            state_dict = torch.load(ckpt_path, map_location='cpu')
            if isinstance(state_dict, dict) and self.pfgr_module is not None:
                try:
                    self.pfgr_module.pfgr_branch.load_state_dict(state_dict, strict=False)
                    logger.info("PFGR weights loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load PFGR weights: {e}")

            prefix = ckpt_path[:-3] if ckpt_path.endswith(".pt") else ckpt_path


            ctx_cont_path = prefix + "_ctx_cont.pt"
            ctx_soft_path = prefix + "_ctx_soft.pt"

            if os.path.exists(ctx_cont_path):
                ctx = torch.load(ctx_cont_path, map_location="cpu")
                if torch.is_tensor(ctx) and ctx.shape == self.ctx_cont.shape:
                    self.ctx_cont.data.copy_(ctx.to(self.ctx_cont.device))
                    logger.info("Text prompt ctx_cont loaded successfully")
                else:
                    logger.warning(f"ctx_cont shape mismatch: {getattr(ctx,'shape',None)} vs {self.ctx_cont.shape}, skipping load")

            if os.path.exists(ctx_soft_path):
                ctxs = torch.load(ctx_soft_path, map_location="cpu")
                if torch.is_tensor(ctxs) and ctxs.shape == self.ctx_soft.shape:
                    self.ctx_soft.data.copy_(ctxs.to(self.ctx_soft.device))
                    logger.info("Text prompt ctx_soft loaded successfully")
                else:
                    logger.warning(f"ctx_soft shape mismatch: {getattr(ctxs,'shape',None)} vs {self.ctx_soft.shape}, skipping load")


            if self.post_fps_prompt is not None:
                point_prompt_pt = prefix + "_point_prompt.pt"
                point_prompt_npy = prefix + "_point_prompt.npy"
                prompt_xyz = None
                if os.path.exists(point_prompt_pt):
                    prompt_xyz = torch.load(point_prompt_pt, map_location="cpu")
                elif os.path.exists(point_prompt_npy):
                    prompt_xyz = torch.from_numpy(np.load(point_prompt_npy))

                if prompt_xyz is not None:
                    if torch.is_tensor(prompt_xyz) and prompt_xyz.shape == self.post_fps_prompt.prompt_xyz.shape:
                        self.post_fps_prompt.prompt_xyz.data.copy_(prompt_xyz.to(self.post_fps_prompt.prompt_xyz.device))
                        logger.info("Point prompt (Post-FPS) loaded successfully")
                    else:
                        logger.warning(f"point prompt shape mismatch: {getattr(prompt_xyz,'shape',None)} vs {self.post_fps_prompt.prompt_xyz.shape}, skipping load")

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")


    def _project_ctx_to_discrete(self, require_unique: bool = False):
        cand_ids_list = []
        cand_emb_list = []
        sim_list = []

        for i in range(self.n_ctx):
            ctx_vec = self.ctx_cont[i].float()
            ctx_vec_norm = F.normalize(ctx_vec, dim=-1)

            cand_ids = getattr(self, f"cand_ids_{i}")
            cand_emb = getattr(self, f"cand_emb_{i}")
            cand_emb_norm = getattr(self, f"cand_emb_norm_{i}")

            sim = torch.matmul(cand_emb_norm, ctx_vec_norm)

            cand_ids_list.append(cand_ids)
            cand_emb_list.append(cand_emb)
            sim_list.append(sim)

        selected_indices = [None] * self.n_ctx
        used_token_ids = set()

        if require_unique:
            slot_order = sorted(
                range(self.n_ctx),
                key=lambda j: int(torch.unique(cand_ids_list[j]).numel())
            )
        else:
            slot_order = list(range(self.n_ctx))

        for slot in slot_order:
            cand_ids = cand_ids_list[slot]
            sim = sim_list[slot]

            if require_unique:
                if len(used_token_ids) > 0:
                    used_ids_tensor = torch.tensor(
                        list(used_token_ids),
                        dtype=cand_ids.dtype,
                        device=cand_ids.device
                    )
                    available_mask = ~torch.isin(cand_ids, used_ids_tensor)
                else:
                    available_mask = torch.ones_like(cand_ids, dtype=torch.bool)

                if not bool(available_mask.any().item()):
                    raise RuntimeError(
                        f"[TextPrompt] Cannot select a unique token for slot-{slot}: candidate pool occupied by preceding slots."
                    )

                score = sim.masked_fill(~available_mask, float('-inf'))
                idx = int(score.argmax().item())
            else:
                idx = int(sim.argmax().item())

            selected_indices[slot] = idx
            used_token_ids.add(int(cand_ids[idx].item()))

        discrete_embs = []
        token_ids = []
        for i in range(self.n_ctx):
            idx = selected_indices[i]
            cand_ids = cand_ids_list[i]
            cand_emb = cand_emb_list[i]
            token_ids.append(cand_ids[idx])
            discrete_embs.append(cand_emb[idx].to(self.ctx_cont.dtype))

        ctx_discrete = torch.stack(discrete_embs, dim=0)
        token_ids = torch.stack(token_ids, dim=0)

        if require_unique:
            unique_count = int(torch.unique(token_ids).numel())
            if unique_count != int(token_ids.numel()):
                raise RuntimeError("[TextPrompt] Internal error: hard prompt token deduplication failed.")

        return ctx_discrete, token_ids

    @torch.no_grad()
    def get_current_discrete_token_ids(self):
        _, token_ids = self._project_ctx_to_discrete(require_unique=self.enforce_unique_hard_prompts)
        return [int(t.item()) for t in token_ids]

    def _get_ctx_for_forward(self):
        ctx_discrete, token_ids = self._project_ctx_to_discrete(
            require_unique=self.enforce_unique_hard_prompts
        )

        ctx_pez = self.ctx_cont + (ctx_discrete - self.ctx_cont).detach()
        ctx_all = torch.cat([self.ctx_soft, ctx_pez], dim=0)
        return ctx_all, token_ids

    def reset_weights(self):
        self.log_vars = nn.Parameter(torch.zeros(3, device=self.device))

    def get_prompt(self):
        return None

    def get_point_prompt_xyz(self):
        if self.post_fps_prompt is None:
            return None
        return self.post_fps_prompt.prompt_xyz

    def get_text_feat(self):
        if self.use_pez_text:
            ctx_for_forward, _ = self._get_ctx_for_forward()
        else:
            ctx_for_forward = torch.cat([self.ctx_soft, self.ctx_cont], dim=0)

        return encode_text_with_prompt(
            self.clip_model_text,
            ctx_for_forward,
            self.tokenized_prompts,
            ctx_positions=self.ctx_positions
        )

    def get_multi_task_loss(self, loss_image, loss_point, loss_text):
        log_vars_clamped = torch.clamp(self.log_vars, min=self.min_log_var, max=self.max_log_var)
        loss = (
            torch.exp(-log_vars_clamped[0]) * loss_image + log_vars_clamped[0] +
            torch.exp(-log_vars_clamped[1]) * loss_point + log_vars_clamped[1] +
            torch.exp(-log_vars_clamped[2]) * loss_text + log_vars_clamped[2]
        )
        self.log_vars_clamped = log_vars_clamped
        return loss

    def save_prompt(self, path):


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
            logger.info(f"Currently learned hard prompt vocabulary: {decoded_words}")
            with open(prefix + "_words.txt", "w") as f:
                f.write(str(decoded_words))
        except Exception as e:
            logger.warning(f"Failed to save/decode ctx_tokens: {e}")


        if self.post_fps_prompt is not None:
            prompt_xyz = self.post_fps_prompt.prompt_xyz.detach().cpu()
            np.save(prefix + "_point_prompt.npy", prompt_xyz.numpy())
            torch.save(prompt_xyz, prefix + "_point_prompt.pt")
            save_point_prompt_visualization(prompt_xyz, prefix + "_point_prompt.png")
            logger.info(f"Point prompt saved to: {prefix + '_point_prompt.npy'}")

    def reparameterize_for_inference(self):
        if self.pfgr_module is None:
            return

        self.pfgr_module.reparameterize()

        if getattr(self, '_pfgr_target_path', None):
            try:
                _set_submodule(self.student_model.point_encoder, self._pfgr_target_path, self.pfgr_module.original_layer)
                logger.info(f"[PFGR] Restored {self._pfgr_target_path} to original linear layer after merge (no extra branches for inference)")
                self.pfgr_module = None
            except Exception as e:
                logger.warning(f"[PFGR] Failed to restore original layer (will continue inference with PFGR_Module wrapper): {e}")

    def forward(self, point_clouds):

        if self.use_pez_text:
            ctx_for_forward, _ = self._get_ctx_for_forward()
        else:
            ctx_for_forward = torch.cat([self.ctx_soft, self.ctx_cont], dim=0)

        text_features = encode_text_with_prompt(
            self.clip_model_text,
            ctx_for_forward,
            self.tokenized_prompts,
            ctx_positions=self.ctx_positions
        )

        pc_input = ensure_xyzrgb_torch(point_clouds).contiguous()

        pc_embed = self.student_model.encode_pc(pc_input)
        return pc_embed, text_features


def get_ulip2_pointbert_teacher(args, open_clip_model):
    config_addr = 'XXX'
    if not os.path.exists(config_addr):
        config_addr = 'XXX'

    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer_Colored(config.model, args=args)
    pc_feat_dims = 768

    model = ULIP2_WITH_OPENCLIP(open_clip_model=open_clip_model, point_encoder=point_encoder, pc_feat_dims=pc_feat_dims, embed_dim=args.embed_dim)

    ckpt_path = 'XXX'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        sd = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        logger.info(f"Teacher Model loaded from: {ckpt_path}")
    else:
        logger.warning(f"[Warning] Teacher model weights not found, path: {ckpt_path}")

    return model


def get_ulip2_pointbert_student(args, open_clip_model):
    config_addr = 'XXX'
    if not os.path.exists(config_addr):
        config_addr = 'XXX'

    config = cfg_from_yaml_file(config_addr)
    point_encoder = PointTransformer_Colored(config.model, args=args)
    pc_feat_dims = 768

    model = ULIP2_WITH_OPENCLIP(
        open_clip_model=open_clip_model,
        point_encoder=point_encoder,
        pc_feat_dims=pc_feat_dims,
        embed_dim=args.embed_dim
    )

    ckpt_path = 'XXX'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        sd = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        sd = {k.replace('module.', ''): v for k, v in sd.items()}

        model_state = model.state_dict()
        filtered_sd = {}

        print("\n" + "=" * 60)
        print("[Loading Mode] Starting to load weights...")

        ckpt_keys = ['pc_projection', 'pc_projection.weight', 'pc_projection.bias']
        model_keys = ['pc_projection', 'pc_projection.weight', 'pc_projection.bias']

        for ck in ckpt_keys:
            if ck not in sd:
                continue

            candidates = [mk for mk in model_keys if mk in model_state]
            if not candidates:
                print("pc_projection related keys not found in model state_dict")
                break

            if ck in candidates:
                tk = ck
            elif ck.endswith('.weight') and 'pc_projection' in candidates:
                tk = 'pc_projection'
            elif ck == 'pc_projection' and 'pc_projection.weight' in candidates:
                tk = 'pc_projection.weight'
            else:
                tk = candidates[0]

            w = sd[ck]
            target_shape = model_state[tk].shape

            if w.shape == target_shape:
                filtered_sd[tk] = w
                print(f"Successfully matched {tk} (from {ck})")
            elif w.ndim == 2 and w.shape[::-1] == target_shape:
                filtered_sd[tk] = w.t()
                print(f"Successfully matched {tk} (executed transpose: {w.shape} -> {target_shape})")
            else:
                print(f"{ck} shape mismatch: Chk {w.shape} vs Mod {target_shape}")


        for k, v in sd.items():
            if 'pc_projection' in k:
                continue

            if k in model_state:
                target_shape = model_state[k].shape

                if v.shape == target_shape:
                    filtered_sd[k] = v

                elif 'first_conv' in k and v.ndim >= 2:
                    if v.shape[1] == 6 and target_shape[1] == 3:
                        print(f"Input layer channel mismatch {k} detected. Cropping 6->3 channels...")
                        filtered_sd[k] = v[:, :3, ...]
                    elif v.shape[1] == 3 and target_shape[1] == 6:
                        print(f"Input layer channel mismatch {k} detected. Padding 3->6 channels (new channels zeroed)...")
                        pad_shape = list(v.shape)
                        pad_shape[1] = 3
                        pad = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
                        filtered_sd[k] = torch.cat([v, pad], dim=1)
                    else:
                        if 'encoder' in k:
                            print(f"Parameter {k} shape mismatch: {v.shape} vs {target_shape}")
                        continue
                elif v.shape[::-1] == target_shape:
                    filtered_sd[k] = v.t()
                else:
                    if 'encoder' in k:
                        print(f"Parameter {k} shape mismatch: {v.shape} vs {target_shape}")
                    continue

        if any(k.startswith('pc_projection') for k in filtered_sd.keys()):
            print("[Success] pc_projection related parameters ready to load!")
        else:
            logger.warning("[Info] pc_projection related parameters not loaded.")

        model.load_state_dict(filtered_sd, strict=False)
        print("=" * 60 + "\n")

        logger.info(f"Student Model loaded from: {ckpt_path}")
    else:
        logger.warning(f"[Warning] Student model weights not found, path: {ckpt_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ULIP-2 Prompt Training (Text PEZ + Point PFGR/Post-FPS)")
    parser.add_argument("--pc_model", type=str, default="eva02_base_patch14_448", help="Point cloud model name")
    parser.add_argument("--pc_feat_dim", type=int, default=768, help="Point cloud feature dimension")
    parser.add_argument("--pretrained_pc", type=str, default=None, help="Path to pretrained point cloud model")
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="Drop path rate")
    parser.add_argument("--pc_encoder_dim", type=int, default=512, help="Point cloud encoder dimension")
    parser.add_argument("--embed_dim", type=int, default=1280, help="Embedding dimension")
    parser.add_argument("--ckpt_path", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--num_group", type=int, default=512, help="Number of groups for point cloud processing")
    parser.add_argument("--group_size", type=int, default=64, help="Group size for point cloud processing")
    parser.add_argument("--evaluate_3d", action="store_true", default=True, help="Skip internal loading of pretrained weights in PointBERT")


    parser.add_argument("--epochs", type=int, default=110, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size")
    parser.add_argument("--lr", type=float, default=7e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=6e-3, help="Weight decay")


    parser.add_argument("--pfgr_lambda", type=float, default=1.0, help="PFGR branch scaling factor λ")
    parser.add_argument("--pfgr_warmup_epochs", type=int, default=4,
                        help="PFGR warmup epochs: set λ=0 for the first N epochs")
    parser.add_argument("--pfgr_ramp_epochs", type=int, default=10,
                        help="PFGR ramp epochs after warmup: linearly increase λ from 0 to target")
    parser.add_argument("--post_fps_warmup_epochs", type=int, default=4,
                        help="Post-FPS warmup epochs: disable prompt injection for the first N epochs")
    parser.add_argument("--post_fps_ramp_epochs", type=int, default=8,
                        help="Post-FPS ramp epochs after warmup: linearly increase prompt scale to 1")
    parser.add_argument("--point_loss_mode", type=str, default="combined", choices=["mse", "cosine", "combined"],
                        help="Geometric alignment loss mode")

    parser.add_argument("--image_loss_warmup_epochs", type=int, default=3,
                        help="Warmup epochs for image loss: disable selective gating for the first N epochs")
    parser.add_argument("--selective_topk", type=int, default=2,
                        help="Top-k for selective gating (teacher correct if GT in top-k)")

    parser.add_argument("--image_loss_ramp_epochs", type=int, default=6,
                        help="After image_loss_warmup_epochs, linearly ramp the selective mix coefficient to 1 over N epochs")
    parser.add_argument("--loss_point_scale", type=float, default=500,
                        help="Scale factor applied to loss_point before multi-task weighting")
    parser.add_argument("--min_log_var", type=float, default=-1.8,
                        help="Lower bound for log_vars (uncertainty weighting)")
    parser.add_argument("--max_log_var", type=float, default=2.5,
                        help="Upper bound for log_vars (uncertainty weighting)")
    parser.add_argument("--gate_update", type=str, default="ema", choices=["fixed", "ema"],
                        help="Update rule for gate_text_features used by selective gating")
    parser.add_argument("--gate_ema", type=float, default=0.995,
                        help="EMA momentum for gate_text_features if gate_update=ema")


    parser.add_argument("--text_prompt_size", type=int, default=7, help="Hard text prompt slots (n_ctx)")
    parser.add_argument("--soft_prompt_size", type=int, default=3, help="Soft text prompt slots (n_soft)")
    parser.add_argument("--disable_pez", action="store_true", help="Disable PEZ projection (hard prompt stays continuous)")


    parser.add_argument("--point_prompt_len", type=int, default=10, help="Post-FPS point prompt length (P)")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP (enabled by default)")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["fp16", "bf16"],
                        help="AMP dtype: fp16 (default) or bf16 (if supported)")

    args = parser.parse_args()

    AMP_ENABLED = (not args.no_amp)
    AMP_DTYPE = torch.float16 if args.amp_dtype == 'fp16' else torch.bfloat16
    scaler = GradScaler(enabled=AMP_ENABLED)
    logger.info(f"AMP enabled: {AMP_ENABLED} | dtype={args.amp_dtype}")

    logger.info("Loading dataset...")

    dataname = "Modelnet40"
    if dataname == "Modelnet40":
        orgin_train_dataset = ModelNet40Dataset(
            root_dir='XXX',
            split='modelnet40_train',
            transform=transform
        )
        train_dataset = NpzPointCloudDataset(
            'XXX',
            classnames=orgin_train_dataset.classnames
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

        test_dataset = NpzPointCloudDataset(
            'XXX',
            classnames=orgin_train_dataset.classnames
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    else:
        orgin_train_dataset = ScanObjectNNDataset(
            h5_path='XXX',
            transform=transform
        )
        train_dataset = NpzPointCloudDataset(
            'XXX',
            classnames=orgin_train_dataset.classnames
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

        test_dataset = NpzPointCloudDataset(
            'XXX',
            classnames=orgin_train_dataset.classnames
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    logger.info("Dataset loaded successfully.")

    logger.info("Loading CLIP model...")
    clip_model_path = 'XXX'
    logger.info(f"CLIP model path: {clip_model_path}")

    try:
        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained=None)
        from safetensors.torch import load_file
        state_dict = load_file(clip_model_path)
        clip_model.load_state_dict(state_dict, strict=False)
        print("CLIP model loaded from safetensors file.")
    except Exception as e:
        print(f"Cannot directly load safetensors: {e}")
        print("Attempting standard loading mode...")
        clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained=clip_model_path)

    clip_model_text = clip_model
    clip_model_visual = clip_model
    import copy
    clip_model_visual = copy.deepcopy(clip_model)

    clip_model_text.to('cuda:0')
    clip_model_text.eval()

    logger.info(f"Text Teacher model loaded from: {clip_model_path}")
    logger.info(f"Visual Teacher model loaded from: {clip_model_path}")

    with torch.no_grad():
        soft_prefix = " ".join(["X"] * int(args.soft_prompt_size))
        hard_prefix = " ".join(["X"] * int(args.text_prompt_size))
        classnames_text = [
            f"{soft_prefix} A depth picture of a {hard_prefix} {name.replace('_', ' ')}."
            for name in train_dataset.classnames
        ]
        text_inputs = tokenizer(classnames_text).to("cuda:0")
        text_features = clip_model_text.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        gate_text_features = text_features.detach().clone()

    logger.info("CLIP text feature space initialization complete.")

    gate_text_features = text_features.detach().clone()

    logger.info("Loading CLIP visual teacher model...")
    clip_model_visual.to('cuda:1')
    clip_model_visual.eval()
    gc.collect()
    logger.info("CLIP visual teacher model initialization complete.")

    logger.info("Loading point cloud teacher model (ULIP-2 PointBERT)...")
    point_teacher = get_ulip2_pointbert_teacher(args, clip_model)
    point_teacher.to('cuda:1')
    point_teacher.eval()
    logger.info("Point cloud teacher model loaded successfully.")

    logger.info("Loading point cloud student model (ULIP-2 PointBERT Student)...")
    model_student = get_ulip2_pointbert_student(args, clip_model)
    model_student.to('cuda:0')

    test_accuracy = test_attack(model_student, 0.05, text_features, prompt=None)
    logger.info(f"Pre-prompt learning, Student model Adversarial test accuracy: {test_accuracy:.4f}")
    model_student.eval()
    logger.info("Point cloud student model loaded successfully.")

    student_with_prompt = PointPrompt(
        prompt_len=args.point_prompt_len,
        feat_dim=384,
        student_model=model_student,
        device='cuda:0',
        classnames=train_dataset.classnames,
        text_prompt_size=args.text_prompt_size,
        soft_prompt_size=args.soft_prompt_size,
        clip_model_text=clip_model_text,
        use_pez_text=(not args.disable_pez),
        pfgr_lambda=args.pfgr_lambda,
        min_log_var=args.min_log_var,
        max_log_var=args.max_log_var,
    )
    student_with_prompt.to('cuda:0')
    logger.info("Fused Prompt loaded successfully.")

    trainable_params = sum(p.numel() for p in student_with_prompt.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student_with_prompt.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / Total parameters: {total_params:,}")


    logger.info("Start training")
    MAX_EPOCHS = int(args.epochs)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student_with_prompt.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

    best_score = -1.0

    for epoch in range(MAX_EPOCHS):
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

        if student_with_prompt.post_fps_prompt is not None:
            warmup = max(int(args.post_fps_warmup_epochs), 0)
            ramp = max(int(args.post_fps_ramp_epochs), 0)

            if epoch < warmup:
                scale_now = 0.0
            else:
                if ramp <= 0:
                    scale_now = 1.0
                else:
                    progress = (epoch - warmup + 1) / float(ramp)
                    progress = float(min(max(progress, 0.0), 1.0))
                    scale_now = progress

            student_with_prompt.post_fps_prompt.runtime_scale = float(scale_now)
            logger.info(f"[Post-FPS] epoch={epoch} | scale={scale_now:.6f}")

        total_loss = 0.0
        total_loss_image = 0.0
        total_loss_point = 0.0
        total_loss_text = 0.0
        total_loss_postfps_reg = 0.0
        num_batches = 0

        gating_ratios = []
        alpha_imgs = []

        student_with_prompt.train()

        for data in train_loader:
            labels = data[1].to('cuda:0')

            points_student = data[2].to('cuda:0')

            points_student = ensure_xyzrgb_torch(points_student).contiguous()
            points_teacher_clean = data[0]

            points_teacher_clean = ensure_xyzrgb_torch(points_teacher_clean)

            imgs = process_point_cloud_batch(points_teacher_clean[:, :, :3])
            with torch.no_grad():

                image_features = clip_model_visual.encode_image(imgs.to('cuda:1')).detach().to('cuda:0')
                image_features = F.normalize(image_features, dim=-1)


                points_teacher_input = points_teacher_clean.to('cuda:1', non_blocking=True).contiguous()
                if torch.isnan(points_teacher_input).any():
                    logger.warning("Error: input point cloud contains NaN, skipping this batch!")
                    continue

                with torch.cuda.device('cuda:1'):
                    point_teacher_features = point_teacher.encode_pc(points_teacher_input)

                point_teacher_features = point_teacher_features.detach().to('cuda:0', non_blocking=True)
                point_teacher_features = F.normalize(point_teacher_features, dim=-1)

                del points_teacher_input

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
                loss_image_full = contrastive_loss(
                    student_features,
                    image_features,
                    tau=tau_img
                )

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

                loss_point_raw = geometric_alignment_loss(
                    student_features,
                    point_teacher_features,
                    mode=args.point_loss_mode
                )
                loss_point = loss_point_raw * float(args.loss_point_scale)

                logits_text = torch.matmul(student_features, all_text_features.t()) / student_with_prompt.log_tau[2].exp()
                loss_text = F.cross_entropy(logits_text, labels)

                loss_postfps_reg = loss_text.new_zeros(())
                if student_with_prompt.post_fps_prompt is not None:
                    prompt_xyz = student_with_prompt.post_fps_prompt.prompt_xyz
                    loss_postfps_reg = 1e-3 * (prompt_xyz.float() ** 2).mean()

                loss = student_with_prompt.get_multi_task_loss(loss_image, loss_point, loss_text) + loss_postfps_reg

            total_loss += float(loss.item())
            total_loss_image += float(loss_image.item())
            total_loss_point += float(loss_point.item())
            total_loss_text += float(loss_text.item())
            total_loss_postfps_reg += float(loss_postfps_reg.item())
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
        avg_loss_postfps_reg = total_loss_postfps_reg / max(num_batches, 1)

        if len(gating_ratios) > 0:
            logger.info(f"selective gating ratio (avg over batches): {float(np.mean(gating_ratios)):.4f}")
            logger.info(f"image selective mix alpha (avg over batches): {float(np.mean(alpha_imgs)):.4f}")

        if student_with_prompt.log_vars_clamped is not None:
            sigmas = torch.exp(student_with_prompt.log_vars_clamped / 2).detach().cpu().numpy()
            logger.info(f"Learned Sigma values: Image={sigmas[0]:.4f}, Point={sigmas[1]:.4f}, Text={sigmas[2]:.4f}")
            taus = student_with_prompt.log_tau.detach().exp().cpu().numpy()
            logger.info(f"Temperature tau: Image={taus[0]:.4f}, Point={taus[1]:.4f}, Text={taus[2]:.4f}")

        scheduler.step()

        logger.info(
            f"Epoch {epoch}: Total={avg_loss:.4f}, Image={avg_loss_image:.4f}, Point={avg_loss_point:.4f}, Text={avg_loss_text:.4f}, Post-FPS_Reg={avg_loss_postfps_reg:.6f}"
        )
        for param_group in optimizer.param_groups:
            logger.info(f"Current learning rate: {param_group['lr']:.6f}")


        student_with_prompt.eval()
        with torch.no_grad():
            eval_text_features = student_with_prompt.get_text_feat()
            eval_text_features = F.normalize(eval_text_features, dim=-1)

        old_lambda = None
        old_scale = None
        pfgr_eval_lambda = float(args.pfgr_lambda)

        try:
            if student_with_prompt.pfgr_module is not None:
                old_lambda = float(student_with_prompt.pfgr_module.lambda_scale)
                student_with_prompt.pfgr_module.lambda_scale = 0.0

            if student_with_prompt.post_fps_prompt is not None:
                old_scale = float(student_with_prompt.post_fps_prompt.runtime_scale)
                student_with_prompt.post_fps_prompt.runtime_scale = 0.0

            acc_text = test_attack(
                model_student,
                0.05,
                text_features,
                eval_text_features,
                prompt=None,
                pfgr_module=None
            )

            if student_with_prompt.pfgr_module is not None:
                student_with_prompt.pfgr_module.lambda_scale = pfgr_eval_lambda
            if student_with_prompt.post_fps_prompt is not None:
                student_with_prompt.post_fps_prompt.runtime_scale = 1.0

            acc_both = test_attack(
                model_student,
                0.05,
                text_features,
                eval_text_features,
                prompt=None,
                pfgr_module=student_with_prompt.pfgr_module
            )
        finally:
            if student_with_prompt.pfgr_module is not None and old_lambda is not None:
                student_with_prompt.pfgr_module.lambda_scale = old_lambda
            if student_with_prompt.post_fps_prompt is not None and old_scale is not None:
                student_with_prompt.post_fps_prompt.runtime_scale = old_scale

        score = acc_both
        logger.info(f"PGD accuracy: text-only={acc_text:.4f}, text+point={acc_both:.4f}, selection_score={score:.4f}")

        if score > best_score:
            best_score = score
            student_with_prompt.save_prompt(str(save_dir / f"{task}_best_prompt{suffix}.pt"))
            logger.info(f"Best prompt saved | PGD text+point accuracy={best_score:.4f}")

        student_with_prompt.save_prompt(str(save_dir / f"{task}_latest_prompt{suffix}.pt"))


        try:
            del image_features, point_teacher_features, student_features, eval_text_features, points_student, labels
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"[Epoch {epoch}] VRAM used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB | Max: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    logger.info("Training complete. Executing PFGR reparameterization for Zero Inference Overhead...")
    student_with_prompt.reparameterize_for_inference()

    final_model_path = str(save_dir / f"{task}_final_reparameterized{suffix}.pt")
    torch.save(model_student.state_dict(), final_model_path)
    logger.info(f"Reparameterized model saved to: {final_model_path}")
