import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from loguru import logger
from models.uni3d import create_uni3d
import utils.open_clip as open_clip
from utils.tokenizer import SimpleTokenizer
def ensure_xyzrgb_torch(points: torch.Tensor, rgb_value: float = 0.5) -> torch.Tensor:
    if not torch.is_tensor(points):
        raise TypeError("points must be a torch.Tensor")
    if points.ndim == 2:
        pts = points.unsqueeze(0)
        squeeze_back = True
    else:
        pts = points
        squeeze_back = False
    if pts.shape[-1] == 3:
        B, N, _ = pts.shape
        rgb = torch.full((B, N, 3), float(rgb_value), device=pts.device, dtype=pts.dtype)
        pts = torch.cat([pts, rgb], dim=-1)
    elif pts.shape[-1] >= 6:
        pts = pts[..., :6]
    else:
        B, N, C = pts.shape
        pad = torch.zeros((B, N, 6 - C), device=pts.device, dtype=pts.dtype)
        pts = torch.cat([pts, pad], dim=-1)
    return pts.squeeze(0) if squeeze_back else pts
def _resolve_device(dev_str: str, fallback: str = "cuda:0") -> torch.device:
    if torch.cuda.is_available() and isinstance(dev_str, str) and dev_str.startswith("cuda"):
        try:
            if ":" in dev_str:
                idx = int(dev_str.split(":")[1])
                if idx < torch.cuda.device_count():
                    return torch.device(dev_str)
            else:
                return torch.device("cuda:0")
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device(fallback)
    return torch.device("cpu")
def _parse_float_list(spec, default_values):
    if spec is None:
        return list(default_values)
    if isinstance(spec, (list, tuple)):
        try:
            return [float(x) for x in spec]
        except Exception:
            return list(default_values)
    values = []
    for item in str(spec).split(','):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except Exception:
            pass
    return values if values else list(default_values)
def _parse_str_list(spec, default_values):
    if spec is None:
        return list(default_values)
    if isinstance(spec, (list, tuple)):
        values = [str(x).strip() for x in spec if str(x).strip()]
        return values if values else list(default_values)
    values = [item.strip() for item in str(spec).split(',') if item.strip()]
    return values if values else list(default_values)
def _farthest_point_sample_idx(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    assert xyz.ndim == 3 and xyz.shape[-1] >= 3
    B, N, _ = xyz.shape
    npoint = int(min(max(int(npoint), 1), N))
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, device=xyz.device, dtype=xyz.dtype)
    farthest = torch.randint(0, N, (B,), device=xyz.device)
    batch_indices = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = ((xyz - centroid) ** 2).sum(-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids
def maybe_fps6d(points: torch.Tensor, num_points: int) -> torch.Tensor:
    if not torch.is_tensor(points):
        raise TypeError('points must be torch.Tensor')
    squeeze_back = False
    if points.ndim == 2:
        points = points.unsqueeze(0)
        squeeze_back = True
    if points.ndim != 3:
        raise ValueError(f'Expected 2D or 3D tensor, got shape={tuple(points.shape)}')
    B, N, C = points.shape
    num_points = int(num_points)
    if num_points <= 0 or N <= num_points:
        return points.squeeze(0) if squeeze_back else points
    xyz = points[..., :3].contiguous()
    idx = _farthest_point_sample_idx(xyz, num_points)
    idx_expand = idx.unsqueeze(-1).expand(-1, -1, C)
    sampled = torch.gather(points, 1, idx_expand)
    return sampled.squeeze(0) if squeeze_back else sampled
transform = transforms.Compose([
    lambda pc: np.hstack((pc[:, :3], np.full((pc.shape[0], 3), 0.5, dtype=np.float32))),
    lambda pc: torch.tensor(pc, dtype=torch.float32),
])
class NpzPointCloudDataset(Dataset):
    POINT_KEYS = ("adv_pc", "perturbed_points", "points", "pointcloud", "pc", "data")
    LABEL_KEYS = ("labels", "test_label", "adv_label", "y", "label", "target")
    def __init__(self, npz_path: str, transform=None):
        self.npz_path = str(npz_path)
        data = np.load(self.npz_path)
        point_key = next((k for k in self.POINT_KEYS if k in data), None)
        label_key = next((k for k in self.LABEL_KEYS if k in data), None)
        if point_key is None or label_key is None:
            raise KeyError(
                f"NPZ keys not supported: {self.npz_path}. "
                f"Found={list(data.keys())}, point_keys={self.POINT_KEYS}, label_keys={self.LABEL_KEYS}"
            )
        self.points = data[point_key]
        self.labels = np.asarray(data[label_key]).astype(np.int64).reshape(-1)
        self.transform = transform
        logger.info(
            f"[NPZ] Loaded {self.npz_path} | point_key={point_key} {tuple(self.points.shape)} | "
            f"label_key={label_key} {tuple(self.labels.shape)}"
        )
    def __len__(self):
        return int(self.points.shape[0])
    def __getitem__(self, idx):
        pts = self.points[idx]
        if self.transform is not None:
            pts = self.transform(pts)
        else:
            pts = torch.tensor(pts, dtype=torch.float32)
            pts = ensure_xyzrgb_torch(pts)
        label = int(self.labels[idx])
        return pts, label
class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split="modelnet40_train", transform=None):
        self.root_dir = str(root_dir)
        self.split = split
        self.transform = transform
        shape_names_file = os.path.join(self.root_dir, 'XXX')
        with open(shape_names_file, "r") as f:
            self.classnames = [line.strip() for line in f.readlines()]
        split_file = os.path.join(self.root_dir, f"{split}.txt")
        with open(split_file, "r") as f:
            self.file_list = [line.strip() for line in f.readlines()]
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        category = file_name.rsplit("_", 1)[0]
        label = self.classnames.index(category)
        points_file = os.path.join(self.root_dir, category, f"{file_name}.txt")
        points = np.loadtxt(points_file, delimiter=",").astype(np.float32)
        if self.transform is not None:
            points = self.transform(points)
        else:
            points = torch.tensor(points, dtype=torch.float32)
            points = ensure_xyzrgb_torch(points)
        return points, label
tokenizer = SimpleTokenizer()
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
        f"Cannot find text tower in {type(model).__name__} (requires token_embedding/positional_embedding/transformer/ln_final)."
    )
@torch.no_grad()
def encode_text_with_prompt(
    text_tower: nn.Module,
    prompt_ctx: torch.Tensor,
    tokenized_prompts: torch.Tensor,
    ctx_positions: torch.Tensor = None
) -> torch.Tensor:
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
    attn_mask = getattr(text_tower, "attn_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask[:seq_len, :seq_len].to(device=tokenized_prompts.device)
    ln_final = getattr(text_tower, "ln_final")
    layout = getattr(text_tower, "_prompt_encode_layout", None)
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
    if layout is None:
        try:
            x_out = _run_seq_first(x)
            text_tower._prompt_encode_layout = "seq_first"
        except Exception:
            x_out = _run_batch_first(x)
            text_tower._prompt_encode_layout = "batch_first"
    else:
        x_out = _run_seq_first(x) if layout == "seq_first" else _run_batch_first(x)
    eot_pos = tokenized_prompts.argmax(dim=-1)
    pooled = x_out[torch.arange(x_out.shape[0], device=tokenized_prompts.device), eot_pos]
    text_projection = getattr(text_tower, "text_projection", None)
    if text_projection is not None:
        if isinstance(text_projection, nn.Linear):
            pooled = text_projection(pooled)
        else:
            pooled = pooled @ text_projection
    return pooled
def locate_x_positions(tokenized_prompt_1d: torch.Tensor, total_ctx: int) -> torch.Tensor:
    x_id_space = _get_single_token_id("X", add_space=True)
    x_id_nospace = _get_single_token_id("X", add_space=False)
    positions = []
    if x_id_space is not None:
        positions.append((tokenized_prompt_1d == x_id_space).nonzero(as_tuple=False).squeeze(-1))
    if x_id_nospace is not None and x_id_nospace != x_id_space:
        positions.append((tokenized_prompt_1d == x_id_nospace).nonzero(as_tuple=False).squeeze(-1))
    if len(positions) == 0:
        raise RuntimeError("Cannot locate placeholder token 'X', please check if tokenizer and prompt template match.")
    positions = torch.cat(positions, dim=0)
    positions = torch.unique(positions).sort()[0]
    positions = positions[:total_ctx]
    if int(positions.numel()) != int(total_ctx):
        raise RuntimeError(
            f"Not enough placeholders X: expected {total_ctx}, but only found {int(positions.numel())}.\n"
            f"Suggestion: check if template string contains enough 'X', and if tokenizer tokenizes 'X' consistently."
        )
    return positions
@torch.no_grad()
def build_text_features(clip_model_text, classnames, device: torch.device):
    classnames_text = [f"X X X A depth picture of {name}." for name in classnames]
    text_inputs = open_clip.tokenize(classnames_text).to(device)
    text_features = clip_model_text.encode_text(text_inputs)
    text_features = F.normalize(text_features, dim=-1)
    return text_features
def _blend_logits(std_logits: torch.Tensor, prompted_logits: torch.Tensor, beta: float) -> torch.Tensor:
    beta = float(beta)
    return (1.0 - beta) * std_logits + beta * prompted_logits
def _accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item()) if labels.numel() > 0 else 0.0
def collect_point_features(
    model: nn.Module,
    test_loader,
    prompt: torch.Tensor = None,
    device: torch.device = None,
):
    assert device is not None, 'device must be provided'
    model.eval()
    use_point_prompt = prompt is not None
    _set_postfps_enabled(model, use_point_prompt, prompt_xyz=prompt, prompt_scale=1.0)
    _set_pfgr_enabled(model, use_point_prompt)
    all_point_features = []
    all_labels = []
    for data in test_loader:
        points = data[0].to(device)
        labels = torch.as_tensor(data[1], device=device, dtype=torch.long)
        points = ensure_xyzrgb_torch(points).contiguous()
        with torch.no_grad():
            point_features = model.encode_pc(points)
            point_features = F.normalize(point_features, dim=-1)
        all_point_features.append(point_features.detach().cpu())
        all_labels.append(labels.detach().cpu())
    _set_postfps_enabled(model, False)
    _set_pfgr_enabled(model, False)
    return torch.cat(all_point_features, dim=0), torch.cat(all_labels, dim=0)
def collect_logits(
    model: nn.Module,
    test_loader,
    text_features: torch.Tensor,
    prompted_text_features: torch.Tensor,
    prompt: torch.Tensor = None,
    device: torch.device = None,
):
    point_features, labels = collect_point_features(
        model,
        test_loader,
        prompt=prompt,
        device=device,
    )
    std_logits = torch.matmul(point_features, text_features.detach().cpu().T)
    prompted_logits = torch.matmul(point_features, prompted_text_features.detach().cpu().T)
    return std_logits, prompted_logits, labels
@torch.no_grad()
def build_prompted_text_features(
    clip_model_text,
    classnames,
    ctx_soft: torch.Tensor,
    ctx_cont: torch.Tensor,
    ctx_tokens: torch.Tensor,
    device: torch.device,
):
    text_tower = find_open_clip_text_tower(clip_model_text)
    n_soft = int(ctx_soft.shape[0])
    n_hard = int(ctx_cont.shape[0])
    total_ctx = n_soft + n_hard
    soft_prefix = " ".join(["X"] * n_soft)
    hard_prefix = " ".join(["X"] * n_hard)
    text_prompts = [
        f"{soft_prefix} A depth picture of a {hard_prefix} {name.replace('_',' ')}."
        for name in classnames
    ]
    tokenized_prompts = torch.stack([tokenizer(p) for p in text_prompts]).to(device)
    ctx_positions = locate_x_positions(tokenized_prompts[0], total_ctx).to(device)
    with torch.no_grad():
        token_emb_weight = text_tower.token_embedding.weight
        ctx_tokens = ctx_tokens.to(device=device, dtype=torch.long).view(-1)
        ctx_discrete = token_emb_weight[ctx_tokens]
    ctx_soft = ctx_soft.to(device=device, dtype=ctx_discrete.dtype)
    ctx_cont = ctx_cont.to(device=device, dtype=ctx_discrete.dtype)
    ctx_hard = ctx_cont + (ctx_discrete - ctx_cont).detach()
    ctx_all = torch.cat([ctx_soft, ctx_hard], dim=0)
    text_features = encode_text_with_prompt(text_tower, ctx_all, tokenized_prompts, ctx_positions=ctx_positions)
    text_features = F.normalize(text_features, dim=-1)
    return text_features
class PFGR_Module(nn.Module):
    def __init__(self, original_layer: nn.Linear, lambda_scale: float = 1.0):
        super().__init__()
        if not isinstance(original_layer, nn.Linear):
            raise TypeError("PFGR_Module expects nn.Linear as original_layer")
        self.original_layer = original_layer
        self.lambda_scale = float(lambda_scale)
        self._is_reparameterized = False
        for p in self.original_layer.parameters():
            p.requires_grad = False
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        self.pfgr_branch = nn.Linear(in_features, out_features, bias=False)
        nn.init.normal_(self.pfgr_branch.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            weight_norm = self.pfgr_branch.weight.data.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            self.pfgr_branch.weight.data = (self.pfgr_branch.weight.data / weight_norm) * 0.1
        logger.info(f"[PFGR] Initialization complete: in={in_features}, out={out_features}, lambda={lambda_scale}")
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
        logger.info("[PFGR] Reparameterization complete: weights merged and PFGR branch closed (inference uses original layer only)")
def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    if path is None or path == "":
        return root
    m = root
    for p in path.split("."):
        m = getattr(m, p)
    return m
def _set_submodule(root: nn.Module, path: str, new_module: nn.Module):
    if path is None or path == "":
        raise ValueError("path must be non-empty to set submodule")
    parts = path.split(".")
    parent_path = ".".join(parts[:-1])
    name = parts[-1]
    parent = _get_submodule(root, parent_path) if parent_path else root
    setattr(parent, name, new_module)
def _guess_point_encoder(student_model: nn.Module) -> nn.Module:
    if student_model is None:
        raise ValueError("student_model is None")
    for attr in ["pc_encoder", "point_encoder", "point_tower", "pc_model", "point_model", "encoder"]:
        if hasattr(student_model, attr):
            m = getattr(student_model, attr)
            if isinstance(m, nn.Module):
                return m
    for name, _ in student_model.named_modules():
        if name.endswith("group_divider"):
            parent_path = name.rsplit(".", 1)[0] if "." in name else ""
            try:
                return _get_submodule(student_model, parent_path)
            except Exception:
                pass
    return student_model
def install_pfgr(student_model: nn.Module, pfgr_state_dict: dict, lambda_scale: float = 1.0, target_path: str = None):
    point_encoder = _guess_point_encoder(student_model)
    target_name = target_path
    if target_name is None:
        prefer_names = ['reduce_dim', 'proj', 'projection', 'head', 'fc', 'mlp', 'out_proj', 'pc_proj']
        for name in prefer_names:
            if hasattr(point_encoder, name):
                layer = getattr(point_encoder, name)
                if isinstance(layer, nn.Linear):
                    target_name = name
                    break
        if target_name is None:
            candidates = []
            for name, m in point_encoder.named_modules():
                if isinstance(m, nn.Linear):
                    score = 0
                    lname = name.lower()
                    if any(k in lname for k in ['reduce', 'proj', 'head', 'out']):
                        score += 10
                    score += int(m.out_features) // 64
                    candidates.append((score, name, m))
            candidates.sort(reverse=True, key=lambda x: x[0])
            if candidates:
                target_name = candidates[0][1]
    if target_name is None:
        raise RuntimeError("[PFGR] Cannot find a suitable Linear layer to install PFGR.")
    target_layer = _get_submodule(point_encoder, target_name)
    if not isinstance(target_layer, nn.Linear):
        raise RuntimeError(f"[PFGR] Target at '{target_name}' is not nn.Linear (got {type(target_layer)}).")
    pfgr_module = PFGR_Module(target_layer, lambda_scale=lambda_scale).to(next(student_model.parameters()).device)
    _set_submodule(point_encoder, target_name, pfgr_module)
    logger.info(f"[PFGR] Installed on '{target_name}'")
    if pfgr_state_dict is not None:
        missing, unexpected = pfgr_module.pfgr_branch.load_state_dict(pfgr_state_dict, strict=False)
        if missing:
            logger.warning(f"[PFGR] Missing keys when loading pfgr_branch: {missing}")
        if unexpected:
            logger.warning(f"[PFGR] Unexpected keys when loading pfgr_branch: {unexpected}")
        logger.info("[PFGR] Loaded pfgr_branch weights.")
    return pfgr_module, target_name
class PostFPSPointPrompt(nn.Module):
    def __init__(self, prompt_len: int = 10, init_scale: float = 0.05, device: str = "cuda:0"):
        super().__init__()
        self.prompt_len = int(prompt_len)
        self.prompt_xyz = nn.Parameter(torch.randn(1, self.prompt_len, 3, device=device) * float(init_scale))
        self.hook_attached = False
        self.hook_handle = None
        self.enabled = True
    def forward_hook(self, module, inputs, output):
        if not self.enabled:
            return output
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
        Bn, G, S, C = neighborhood.shape
        prompt_neighborhood = torch.zeros((Bn, P, S, C), device=neighborhood.device, dtype=neighborhood.dtype)
        if C >= 6:
            prompt_neighborhood[..., 3:6] = 0.5
        if C >= 3:
            prompt_neighborhood[..., 0:3] = 0.0
        neighborhood_new = torch.cat([neighborhood, prompt_neighborhood], dim=1)
        center_new = torch.cat([center, prompt_center], dim=1)
        new_output = [neighborhood_new, center_new]
        if len(output) > 2 and torch.is_tensor(output[2]) and output[2].dim() == 4:
            features = output[2]
            Bf, Gf, Sf, Cf = features.shape
            prompt_features = torch.zeros((Bf, P, Sf, Cf), device=features.device, dtype=features.dtype)
            if Cf >= 6:
                prompt_features[..., 3:6] = 0.5
            if Cf >= 3:
                prompt_features[..., 0:3] = 0.0
            features_new = torch.cat([features, prompt_features], dim=1)
            new_output.append(features_new)
            if len(output) > 3:
                new_output.extend(output[3:])
            return tuple(new_output)
        if len(output) > 2:
            new_output.extend(output[2:])
        return tuple(new_output)
    def attach(self, point_encoder_root: nn.Module) -> bool:
        if self.hook_attached:
            return True
        group_divider = None
        if hasattr(point_encoder_root, "group_divider"):
            group_divider = getattr(point_encoder_root, "group_divider")
        if group_divider is None:
            for name, m in point_encoder_root.named_modules():
                lname = name.lower()
                if "group_divider" in lname or lname.endswith("group") or "groupdivider" in lname:
                    group_divider = m
                    break
        if group_divider is None:
            logger.warning("[PostFPS] group_divider not found, hook not attached.")
            return False
        self.hook_handle = group_divider.register_forward_hook(self.forward_hook)
        self.hook_attached = True
        logger.info("[PostFPS] Hook attached to group_divider successfully.")
        return True
    def remove(self):
        if self.hook_handle is not None:
            try:
                self.hook_handle.remove()
            except Exception:
                pass
        self.hook_handle = None
        self.hook_attached = False
def _set_postfps_enabled(model: nn.Module, enabled: bool, prompt_xyz: torch.Tensor = None, prompt_scale: float = 1.0):
    m = getattr(model, "_post_fps_prompt", None)
    if m is None:
        return
    m.enabled = bool(enabled)
    if prompt_xyz is not None and torch.is_tensor(prompt_xyz):
        if prompt_xyz.ndim == 2:
            prompt_xyz = prompt_xyz.unsqueeze(0)
        if prompt_xyz.ndim == 3 and prompt_xyz.shape[-1] >= 3:
            scaled = prompt_xyz[:, :, :3] * float(prompt_scale)
            m.prompt_xyz.data = scaled.to(device=m.prompt_xyz.device, dtype=m.prompt_xyz.dtype)
def _set_pfgr_enabled(model: nn.Module, enabled: bool):
    m = getattr(model, "_pfgr_module", None)
    if m is None:
        return
    default = float(getattr(model, "_pfgr_lambda_default", 0.0))
    m.lambda_scale = default if enabled else 0.0
def test_clean(
    model,
    test_loader,
    text_features=None,
    batch_text_features=None,
    prompt=None,
    defense=None,
    device=None,
):
    assert device is not None, "device must be provided"
    model.eval()
    if batch_text_features is None:
        batch_text_features = text_features
    correct = 0
    total = 0
    use_point_prompt = prompt is not None
    _set_postfps_enabled(model, use_point_prompt, prompt_xyz=prompt, prompt_scale=1.0)
    _set_pfgr_enabled(model, use_point_prompt)
    for data in test_loader:
        points = data[0].to(device)
        labels = torch.as_tensor(data[1], device=device, dtype=torch.long)
        points = ensure_xyzrgb_torch(points).contiguous()
        if defense is not None:
            pass
        with torch.no_grad():
            point_features = model.encode_pc(points)
            point_features = F.normalize(point_features, dim=-1)
            similarity = torch.matmul(point_features, batch_text_features.T)
            predictions = similarity.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    _set_postfps_enabled(model, False)
    _set_pfgr_enabled(model, False)
    accuracy = correct / max(total, 1)
    print(f"Clean Accuracy: {accuracy:.4f}")
    return accuracy
def group_test(model_student, dataset, text_features, prompted_text_features, point_prompt, type_name,
              batch_size=32, num_workers=4, device=None):
    assert device is not None, "device must be provided"
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"\n================== {type_name} ==================\n")
    print("test clean model")
    test_clean(model_student, loader, text_features, prompt=None, defense=None, device=device)
    print("test Multi-Modal Prompt")
    test_clean(
        model_student, loader, text_features=prompted_text_features, prompt=point_prompt,
        defense=None, device=device
    )
    print("test Multi-Modal Prompt with prompted text features")
    test_clean(model_student, loader, text_features=prompted_text_features, prompt=None, defense=None, device=device)
    print("test Multi-Modal Prompt with point prompt")
    test_clean(model_student, loader, text_features=text_features, prompt=point_prompt, defense=None, device=device)
def load_prompt_bundle(prompt_ckpt: str, device: torch.device):
    prompt_ckpt = str(prompt_ckpt)
    prefix = prompt_ckpt[:-3] if prompt_ckpt.endswith(".pt") else prompt_ckpt
    logger.info(f"[Path] Main PFGR weight: {prompt_ckpt}")
    logger.info(f"[Path] ctx_cont: {prefix}_ctx_cont.pt")
    logger.info(f"[Path] ctx_soft: {prefix}_ctx_soft.pt")
    logger.info(f"[Path] ctx_tokens: {prefix}_ctx_tokens.pt")
    logger.info(f"[Path] point_prompt: {prefix}_point_prompt.pt")
    logger.info(f"[Path] words: {prefix}_words.txt")
    pfgr_state = None
    if os.path.exists(prompt_ckpt):
        try:
            loaded_data = torch.load(prompt_ckpt, map_location="cpu")
            if isinstance(loaded_data, dict):
                pfgr_state = loaded_data
                logger.info("Loaded PFGR weights dict successfully.")
            else:
                logger.warning(
                    f"Main prompt file contains {type(loaded_data)} instead of dict; skip PFGR weight loading."
                )
        except Exception as e:
            logger.warning(f"Failed to load PFGR weights from {prompt_ckpt}: {e}")
    else:
        logger.warning(f"PFGR ckpt not found: {prompt_ckpt}")
    def _try_load(path: str):
        if os.path.exists(path):
            return torch.load(path, map_location="cpu")
        logger.warning(f"Prompt file missing: {path}")
        return None
    ctx_cont = _try_load(prefix + "_ctx_cont.pt")
    ctx_soft = _try_load(prefix + "_ctx_soft.pt")
    ctx_tokens = _try_load(prefix + "_ctx_tokens.pt")
    point_prompt = _try_load(prefix + "_point_prompt.pt")
    words = None
    words_path = prefix + "_words.txt"
    if os.path.exists(words_path):
        try:
            with open(words_path, "r") as f:
                words = f.read()
        except Exception as e:
            logger.warning(f"Failed to read words file: {words_path} ({e})")
    if torch.is_tensor(ctx_cont):
        ctx_cont = ctx_cont.to(device)
    if torch.is_tensor(ctx_soft):
        ctx_soft = ctx_soft.to(device)
    if torch.is_tensor(ctx_tokens):
        ctx_tokens = ctx_tokens.to(device)
    if torch.is_tensor(point_prompt):
        point_prompt = point_prompt.to(device)
    if torch.is_tensor(ctx_cont) and torch.is_tensor(ctx_soft):
        logger.info(f"Loaded ctx_cont: {tuple(ctx_cont.shape)} | ctx_soft: {tuple(ctx_soft.shape)}")
    if torch.is_tensor(ctx_tokens):
        logger.info(f"Loaded ctx_tokens: {tuple(ctx_tokens.shape)}")
    if torch.is_tensor(point_prompt):
        logger.info(f"Loaded point_prompt_xyz: {tuple(point_prompt.shape)}")
    if words is not None:
        logger.info(f"Loaded words: {words[:200]}...")
    return {
        "prefix": prefix,
        "pfgr_state": pfgr_state,
        "ctx_cont": ctx_cont,
        "ctx_soft": ctx_soft,
        "ctx_tokens": ctx_tokens,
        "point_prompt": point_prompt,
        "words": words,
    }
def build_attack_dataset_map(data_dir: str, adv_root: str):
    data_dir = str(data_dir)
    adv_root = str(adv_root)
    return {
        "clean": os.path.join(data_dir, 'XXX'),
        "PGD whitebox": os.path.join(adv_root, 'XXX', 'XXX'),
        "Perturb whitebox": os.path.join(adv_root, 'XXX', 'XXX'),
        "KNN whitebox": os.path.join(adv_root, 'XXX', 'XXX'),
        "ADD CD whitebox": os.path.join(adv_root, 'XXX', 'XXX'),
        "ADD HD whitebox": os.path.join(adv_root, 'XXX', 'XXX'),
        "Drop 200 whitebox": os.path.join(adv_root, 'XXX', 'XXX'),
        "AdvPC whitebox": os.path.join(adv_root, 'XXX', 'XXX'),
        "AOF whitebox": os.path.join(adv_root, 'XXX', 'XXX'),
        "PGD blackbox": os.path.join(adv_root, 'XXX', 'XXX'),
        "Perturb blackbox": os.path.join(adv_root, 'XXX', 'XXX'),
        "KNN blackbox": os.path.join(adv_root, 'XXX', 'XXX'),
        "ADD CD blackbox": os.path.join(adv_root, 'XXX', 'XXX'),
        "ADD HD blackbox": os.path.join(adv_root, 'XXX', 'XXX'),
        "Drop 200 blackbox": os.path.join(adv_root, 'XXX', 'XXX'),
        "AdvPC blackbox": os.path.join(adv_root, 'XXX', 'XXX'),
        "AOF blackbox": os.path.join(adv_root, 'XXX', 'XXX'),
    }
def load_student_uni3d(args, device: torch.device):
    args.pc_model = args.student_pc_model
    args.pc_feat_dim = args.student_pc_feat_dim
    args.ckpt_path = args.student_ckpt_path
    model = create_uni3d(args).to(device)
    ckpt = torch.load(args.student_ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "module" in ckpt:
            sd = ckpt["module"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"Student missing keys: {missing[:20]} ... (total {len(missing)})")
    if unexpected:
        logger.warning(f"Student unexpected keys: {unexpected[:20]} ... (total {len(unexpected)})")
    model.eval()
    return model
def main():
    parser = argparse.ArgumentParser("Uni3D Prompt Test (fair evaluation, no test-set tuning)")
    parser.add_argument("--pretrained_pc", type=str, default=None)
    parser.add_argument("--student_device", type=str, default="cuda:0")
    parser.add_argument("--student_pc_model", type=str, default="eva02_tiny_patch14_224")
    parser.add_argument("--student_pc_feat_dim", type=int, default=192)
    parser.add_argument("--student_ckpt_path", type=str, default="XXX")
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--pc_encoder_dim", type=int, default=512)
    parser.add_argument("--num_group", type=int, default=512)
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument(
        "--prompt_ckpt",
        type=str,
        default="XXX",
        help="Path to PFGR prompt ckpt (*.pt). Other prompt files are derived by prefix.",
    )
    parser.add_argument("--open_clip_model", type=str, default="EVA02-E-14-plus")
    parser.add_argument(
        "--open_clip_pretrained_bin",
        type=str,
        default="XXX",
    )
    parser.add_argument("--modelnet_root", type=str, default="XXX")
    parser.add_argument("--data_dir", type=str, default="XXX")
    parser.add_argument("--adv_root", type=str, default="XXX")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    device = _resolve_device(args.student_device, fallback="cuda:0")
    logger.info(f"Using device: {device}")
    FAIR_PFGR_LAMBDA = 1.0
    bundle = load_prompt_bundle(args.prompt_ckpt, device=device)
    ctx_cont = bundle["ctx_cont"]
    ctx_soft = bundle["ctx_soft"]
    ctx_tokens = bundle["ctx_tokens"]
    point_prompt_xyz = bundle["point_prompt"]
    pfgr_state = bundle["pfgr_state"]
    if ctx_cont is None or ctx_soft is None or ctx_tokens is None or point_prompt_xyz is None:
        raise RuntimeError("Missing prompt files. Please check <prefix>_ctx_cont/_ctx_soft/_ctx_tokens/_point_prompt.pt")
    model_student = load_student_uni3d(args, device=device)
    if isinstance(pfgr_state, dict):
        pfgr_module, pfgr_path = install_pfgr(
            model_student,
            pfgr_state_dict=pfgr_state,
            lambda_scale=FAIR_PFGR_LAMBDA,
            target_path=None,
        )
        model_student._pfgr_module = pfgr_module
        model_student._pfgr_lambda_default = FAIR_PFGR_LAMBDA
        model_student._pfgr_module.lambda_scale = 0.0
    else:
        logger.warning("PFGR state dict not loaded (will skip PFGR).")
    post_prompt = PostFPSPointPrompt(prompt_len=int(point_prompt_xyz.shape[1]), device=str(device)).to(device)
    with torch.no_grad():
        post_prompt.prompt_xyz.data = point_prompt_xyz[:, :, :3].to(device=device, dtype=post_prompt.prompt_xyz.dtype)
    point_encoder_root = _guess_point_encoder(model_student)
    ok = post_prompt.attach(point_encoder_root)
    if not ok:
        raise RuntimeError("Failed to attach Post-FPS prompt hook (group_divider not found).")
    model_student._post_fps_prompt = post_prompt
    model_student._post_fps_prompt.enabled = False
    logger.info("Loading OpenCLIP text encoder...")
    clip_model_text, _, _ = open_clip.create_model_and_transforms_text(
        args.open_clip_model,
        pretrained=args.open_clip_pretrained_bin,
        device="cpu",
    )
    clip_model_text = clip_model_text.to(device)
    clip_model_text.eval()
    for p in clip_model_text.parameters():
        p.requires_grad = False
    origin_train_dataset = ModelNet40Dataset(args.modelnet_root, split="modelnet40_train", transform=transform)
    classnames = origin_train_dataset.classnames
    logger.info(f"Num classes: {len(classnames)}")
    logger.info("Building baseline text features...")
    text_features = build_text_features(clip_model_text, classnames, device=device)
    logger.info("Building prompted text features (soft+hard+PEZ tokens)...")
    prompted_text_features = build_prompted_text_features(
        clip_model_text,
        classnames,
        ctx_soft=ctx_soft,
        ctx_cont=ctx_cont,
        ctx_tokens=ctx_tokens,
        device=device,
    )
    attack_datasets = build_attack_dataset_map(args.data_dir, args.adv_root)
    for attack_name, npz_path in attack_datasets.items():
        if not os.path.exists(npz_path):
            logger.warning(f"Skip '{attack_name}' (file not found): {npz_path}")
            continue
        dataset = NpzPointCloudDataset(npz_path, transform=None)
        group_test(
            model_student,
            dataset,
            text_features=text_features,
            prompted_text_features=prompted_text_features,
            point_prompt=point_prompt_xyz,
            type_name=attack_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
if __name__ == "__main__":
    main()
