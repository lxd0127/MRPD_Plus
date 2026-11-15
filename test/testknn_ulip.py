import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from loguru import logger
import argparse
from tqdm import tqdm
import gc
import sys
import utils.open_clip as open_clip
from models.ULIP_models import ULIP2_WITH_OPENCLIP
from models.pointbert.point_encoder import PointTransformer_Colored
from utils.utils import cfg_from_yaml_file
from utils.tokenizer import SimpleTokenizer
from models.point_encoder import fps6d
ATTACK_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 16
NUM_WORKERS = 4
ROOT_DIR_MN40 = "XXX"
MODEL_CONFIG_PATH = "XXX"
CLIP_MODEL_PATH = "XXX"
WHITEBOX_DIR = "XXX"
WHITEBOX_FILE_NAME = 'XXX'
WHITEBOX_FULL_PATH = os.path.join(WHITEBOX_DIR, WHITEBOX_FILE_NAME)
BLACKBOX_PATH = "XXX"
CLEAN_NPZ_PATH = "XXX"
BEST_PROMPT_PATH = "XXX"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
os.makedirs(WHITEBOX_DIR, exist_ok=True)
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
class CrossEntropyAdvLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets)
class ClipPointsLinf(nn.Module):
    def __init__(self, budget):
        super().__init__()
        self.budget = budget
    def forward(self, pc, ori_pc):
        with torch.no_grad():
            diff = pc - ori_pc
            norm = torch.sum(diff ** 2, dim=1) ** 0.5
            scale_factor = self.budget / (norm + 1e-9)
            scale_factor = torch.clamp(scale_factor, max=1.)
            diff = diff * scale_factor[:, None, :]
            pc = ori_pc + diff
        return pc
class ProjectInnerPoints(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pc, ori_pc, normal=None):
        with torch.no_grad():
            if normal is None:
                return pc
            diff = pc - ori_pc
            inner_diff_normal = torch.sum(diff * normal, dim=1)
            inner_mask = (inner_diff_normal < 0.)
            vng = torch.cross(normal, diff, dim=1)
            vng_norm = torch.sum(vng ** 2, dim=1) ** 0.5
            vref = torch.cross(vng, normal)
            vref_norm = torch.sum(vref ** 2, dim=1) ** 0.5
            diff_proj = diff * vref / (vref_norm[:, None, :] + 1e-9)
            opposite_mask = inner_mask & (vng_norm < 1e-6)
            opposite_mask = opposite_mask.unsqueeze(1).expand_as(diff_proj)
            diff_proj[opposite_mask] = 0.
            inner_mask = inner_mask.unsqueeze(1).expand_as(diff)
            diff[inner_mask] = diff_proj[inner_mask]
            pc = ori_pc + diff
        return pc
class ProjectInnerClipLinf(nn.Module):
    def __init__(self, budget):
        super().__init__()
        self.project_inner = ProjectInnerPoints()
        self.clip_linf = ClipPointsLinf(budget=budget)
    def forward(self, pc, ori_pc, normal=None):
        with torch.no_grad():
            pc = self.project_inner(pc, ori_pc, normal)
            pc = self.clip_linf(pc, ori_pc)
        return pc
class _Distance(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_cuda = torch.cuda.is_available()
    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x).to(x.device)
        diag_ind_y = torch.arange(0, num_points_y).to(x.device)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P
class ChamferkNNDist(nn.Module):
    def __init__(self, chamfer_method='adv2ori', knn_k=5, knn_alpha=1.05,
                 chamfer_weight=1., knn_weight=1.):
        super().__init__()
        self.chamfer_method = chamfer_method
        self.knn_k = knn_k
        self.knn_alpha = knn_alpha
        self.chamfer_weight = chamfer_weight
        self.knn_weight = knn_weight
        self.dist_func = _Distance()
    def chamfer_dist(self, adv_pc, ori_pc):
        P = self.dist_func.batch_pairwise_dist(adv_pc, ori_pc)
        min_dist, _ = torch.min(P, 2)
        chamfer_dist = torch.mean(min_dist)
        return chamfer_dist
    def knn_dist(self, adv_pc, ori_pc):
        P = self.dist_func.batch_pairwise_dist(adv_pc, ori_pc)
        min_dist, _ = torch.min(P, 2)
        sorted_min_dist, _ = torch.sort(min_dist, dim=1)
        idx = int(self.knn_k)
        knn_d = sorted_min_dist[:, idx]
        knn_d = torch.mean(knn_d)
        return knn_d
    def forward(self, adv_pc, ori_pc, weights=None):
        chamfer_loss = self.chamfer_dist(adv_pc, ori_pc)
        knn_loss = self.knn_dist(adv_pc, ori_pc)
        loss = self.chamfer_weight * chamfer_loss + self.knn_weight * knn_loss
        return loss
class CWKNN(nn.Module):
    def __init__(self, model, adv_func, dist_func, clip_func,
                 attack_lr=1e-3, num_iter=2500,
                 text_features=None):
        super().__init__()
        self.model = model
        self.adv_func = adv_func
        self.dist_func = dist_func
        self.clip_func = clip_func
        self.attack_lr = attack_lr
        self.num_iter = num_iter
        self.text_features = text_features
    def _logits(self, pc):
        feat = self.model.encode_pc(pc)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        sim = torch.matmul(feat, self.text_features.t())
        return sim
    def attack(self, point_clouds, targets):
        ori_pc = point_clouds.detach().clone()
        xyz = point_clouds[:, :, :3].detach().clone()
        other = point_clouds[:, :, 3:].detach().clone() if point_clouds.shape[-1] > 3 else None
        xyz_adv = xyz.clone().requires_grad_(True)
        opt = torch.optim.Adam([xyz_adv], lr=self.attack_lr)
        for _ in range(self.num_iter):
            opt.zero_grad()
            if other is not None:
                pc_adv = torch.cat([xyz_adv, other], dim=-1)
            else:
                pc_adv = xyz_adv
            logits = self._logits(pc_adv)
            adv_loss = self.adv_func(logits, targets)
            dist_loss = self.dist_func(pc_adv[:, :, :3], ori_pc[:, :, :3])
            loss = adv_loss + dist_loss
            loss.backward()
            opt.step()
            with torch.no_grad():
                xyz_adv.data = self.clip_func(xyz_adv.data, xyz, normal=None)
        if other is not None:
            pc_final = torch.cat([xyz_adv.detach(), other], dim=-1)
        else:
            pc_final = xyz_adv.detach()
        return pc_final.detach().cpu().numpy(), None
class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split='modelnet40_test', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.classnames = []
        shape_names_file = os.path.join(root_dir, 'XXX')
        with open(shape_names_file, 'r') as f:
            for line in f:
                self.classnames.append(line.strip())
        split_file = os.path.join(root_dir, f"{split}.txt")
        self.samples = []
        with open(split_file, 'r') as f:
            for line in f:
                self.samples.append(line.strip())
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        class_name = "_".join(sample_name.split('_')[:-1])
        class_idx = self.classnames.index(class_name)
        file_path = os.path.join(self.root_dir, class_name, sample_name + ".txt")
        point_set = np.loadtxt(file_path, delimiter=',').astype(np.float32)
        points = point_set[:, 0:3]
        if self.transform:
            points = self.transform(points)
        return points, class_idx
class NpzPointCloudDataset(Dataset):
    def __init__(self, npz_path):
        super().__init__()
        data = np.load(npz_path, allow_pickle=True)
        for k in ["points", "test_pc", "adv_pc", "point_clouds", "pc"]:
            if k in data:
                self.points = data[k]
                break
        else:
            raise KeyError(f"Missing points key in {npz_path}")
        for k in ["labels", "test_label", "adv_label", "y"]:
            if k in data:
                self.labels = data[k]
                break
        else:
            raise KeyError(f"Missing labels key in {npz_path}")
        self.points = self.points.astype(np.float32)
        self.labels = self.labels.astype(np.int64)
    def __len__(self):
        return len(self.points)
    def __getitem__(self, idx):
        pts = self.points[idx]
        lbl = int(self.labels[idx])
        if pts.shape[-1] == 3:
            rgb = np.ones((pts.shape[0], 3), dtype=np.float32) * 0.5
            pts = np.concatenate([pts, rgb], axis=-1)
        return torch.from_numpy(pts).float(), torch.tensor(lbl).long()
transform = transforms.Compose([
    lambda points: np.concatenate([points, np.ones((points.shape[0], 3), dtype=np.float32) * 0.5], axis=1),
    lambda points: torch.from_numpy(points).float()
])
def get_ulip2_pointbert_student(args, open_clip_model):
    config = cfg_from_yaml_file(MODEL_CONFIG_PATH)
    point_encoder = PointTransformer_Colored(config.model, args=args)
    model = ULIP2_WITH_OPENCLIP(
        open_clip_model=open_clip_model,
        point_encoder=point_encoder,
        pc_feat_dims=768,
        embed_dim=args.embed_dim
    )
    ckpt_path = "XXX"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        sd = checkpoint.get('state_dict', checkpoint)
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model_state = model.state_dict()
        filtered_sd = {}
        for k, v in sd.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_sd[k] = v
                elif v.ndim == 2 and v.shape[::-1] == model_state[k].shape:
                    filtered_sd[k] = v.t()
        model.load_state_dict(filtered_sd, strict=False)
        logger.info("Student model loaded.")
    return model
tokenizer = SimpleTokenizer()
class PromptBundle:
    def __init__(self):
        self.ctx_soft = None
        self.ctx_tokens = None
        self.ctx_positions = None
        self.n_soft = 0
        self.n_hard = 0
        self.pfgr_weight = None
        self.point_prompt_xyz = None
def load_prompt_bundle(prompt_ckpt: str, clip_model, tokenizer, classnames, device):
    if not os.path.exists(prompt_ckpt):
        raise FileNotFoundError(f"Missing best prompt file: {prompt_ckpt}")
    prefix = prompt_ckpt.replace(".pt", "")
    ctx_soft_path = prefix + "_ctx_soft.pt"
    ctx_tokens_path = prefix + "_ctx_tokens.pt"
    ctx_cont_path = prefix + "_ctx_cont.pt"
    point_prompt_path = prefix + "_point_prompt.pt"
    b = PromptBundle()
    b.ctx_soft = torch.load(ctx_soft_path, map_location="cpu").to(device).float()
    if os.path.exists(ctx_tokens_path):
        b.ctx_tokens = torch.load(ctx_tokens_path, map_location="cpu").long().to(device)
    elif os.path.exists(ctx_cont_path):
         logger.warning("Using continuous hard prompt as fallback.")
         b.ctx_tokens = None
         b.ctx_hard_cont = torch.load(ctx_cont_path, map_location="cpu").to(device).float()
    b.n_soft = int(b.ctx_soft.shape[0])
    b.n_hard = int(b.ctx_tokens.shape[0]) if b.ctx_tokens is not None else 3
    soft_prefix = " ".join(["X"] * b.n_soft)
    hard_prefix = " ".join(["X"] * b.n_hard)
    prompts = [f"{soft_prefix} A depth picture of a {hard_prefix} {name}." for name in classnames]
    def _get_single_token_id(word, add_space):
        text = (" " + word) if add_space else word
        ids = tokenizer(text)
        if isinstance(ids, torch.Tensor) and ids.ndim==2: ids=ids[0]
        eot_pos = int(ids.argmax().item())
        content = ids[1:eot_pos]
        content = content[content!=0]
        return int(content.item()) if content.numel() == 1 else None
    tokenized = torch.stack([tokenizer(p) for p in prompts]).to(device) if isinstance(prompts[0], str) else tokenizer(prompts).to(device)
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
    logger.info(f"Loaded Prompt Bundle from {prompt_ckpt}")
    return b
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
    logger.info("Building std text features...")
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
    else:
        raise NotImplementedError("Continuous hard ctx fallback not fully implemented.")
    all_feats = []
    logger.info("Building prompted text features...")
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
    if target_layer is None: raise RuntimeError("PFGR target layer not found")
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
def random_target(labels, num_classes=40):
    targets = torch.randint_like(labels, low=0, high=num_classes)
    targets = torch.where(targets == labels, (targets + 1) % num_classes, targets)
    return targets
def run_knn_attack_batch(model, data, text_features):
    point_clouds, labels = data
    point_clouds = point_clouds.to(DEVICE)
    labels = labels.to(DEVICE)
    if point_clouds.shape[1] > 1024:
        point_clouds = fps6d(point_clouds, 1024)
    target = random_target(labels, num_classes=40)
    adv_func = CrossEntropyAdvLoss()
    dist_func = ChamferkNNDist(knn_k=5, chamfer_weight=5., knn_weight=3.)
    clip_func = ProjectInnerClipLinf(budget=0.1)
    attacker = CWKNN(model, adv_func, dist_func, clip_func, attack_lr=1e-3, num_iter=2500, text_features=text_features)
    perturbed, _ = attacker.attack(point_clouds, target)
    return torch.from_numpy(perturbed).to(DEVICE).float()
def eval_accuracy(model, loader, text_features, device=DEVICE, need_fps=False):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)
            if need_fps and points.shape[1] > 1024:
                points = fps6d(points, 1024)
            feats = model.encode_pc(points)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            pred = (feats @ text_features.t()).argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc_model", type=str, default="eva02_base_patch14_448")
    parser.add_argument("--pc_feat_dim", type=int, default=768)
    parser.add_argument("--embed_dim", type=int, default=1280)
    args = parser.parse_args()
    logger.info("Loading CLIP...")
    from safetensors.torch import load_file
    clip_model_text, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained=None)
    clip_model_text.load_state_dict(load_file(CLIP_MODEL_PATH), strict=False)
    clip_model_text.to(DEVICE).eval()
    temp_dataset = ModelNet40Dataset(root_dir=ROOT_DIR_MN40, split='modelnet40_test')
    classnames = [name.replace("_", " ") for name in temp_dataset.classnames]
    logger.info("Loading Student...")
    model_student = get_ulip2_pointbert_student(args, clip_model_text).to(DEVICE).eval()
    logger.info(f"Loading prompts from {BEST_PROMPT_PATH}")
    bundle = load_prompt_bundle(BEST_PROMPT_PATH, clip_model_text, tokenizer, classnames, DEVICE)
    pfgr_module = install_pfgr(model_student, DEVICE)
    if bundle.pfgr_weight is not None:
        pfgr_module.pfgr_branch.load_state_dict(bundle.pfgr_weight, strict=False)
    post_fps_prompt = None
    if bundle.point_prompt_xyz is not None:
        post_fps_prompt = PostFPSPointPrompt(bundle.point_prompt_xyz).to(DEVICE)
    torch.cuda.empty_cache()
    gc.collect()
    std_text_features = build_std_text_features(clip_model_text, tokenizer, classnames, DEVICE, batch_size=2)
    prompted_text_features = build_prompted_text_features(clip_model_text, tokenizer, classnames, bundle, DEVICE, batch_size=2)
    torch.cuda.empty_cache()
    toggle_point_prompt(model_student, pfgr_module, post_fps_prompt, enable=False)
    if os.path.exists(CLEAN_NPZ_PATH):
        clean_ds = NpzPointCloudDataset(CLEAN_NPZ_PATH)
        loader_c = DataLoader(clean_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        acc = eval_accuracy(model_student, loader_c, std_text_features, DEVICE)
    else:
        clean_ds = ModelNet40Dataset(root_dir=ROOT_DIR_MN40, split='modelnet40_test', transform=transform)
        loader_c = DataLoader(clean_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        acc = eval_accuracy(model_student, loader_c, std_text_features, DEVICE, need_fps=True)
    if not os.path.exists(WHITEBOX_FULL_PATH):
        logger.info("Generating Whitebox Data...")
        gen_ds = ModelNet40Dataset(root_dir=ROOT_DIR_MN40, split='modelnet40_test', transform=transform)
        gen_loader = DataLoader(gen_ds, batch_size=ATTACK_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        all_pts, all_lbl = [], []
        for i, (pts, lbl) in enumerate(tqdm(gen_loader)):
            perturbed = run_knn_attack_batch(model_student, (pts, lbl), std_text_features)
            all_pts.append(perturbed.cpu())
            all_lbl.append(lbl.cpu())
            if i % 5 == 0: torch.cuda.empty_cache()
        np.savez(WHITEBOX_FULL_PATH, points=torch.cat(all_pts).numpy(), labels=torch.cat(all_lbl).numpy())
    datasets = [("Clean", CLEAN_NPZ_PATH)]
    if os.path.exists(WHITEBOX_FULL_PATH): datasets.append(("KNN Whitebox", WHITEBOX_FULL_PATH))
    if os.path.exists(BLACKBOX_PATH): datasets.append(("KNN Blackbox", BLACKBOX_PATH))
    for title, path in datasets:
        ds = NpzPointCloudDataset(path)
        loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        print(f"\n===== {title} =====")
        toggle_point_prompt(model_student, pfgr_module, post_fps_prompt, False)
        acc_clean = eval_accuracy(model_student, loader, std_text_features, DEVICE)
        acc_text = eval_accuracy(model_student, loader, prompted_text_features, DEVICE)
        toggle_point_prompt(model_student, pfgr_module, post_fps_prompt, True)
        acc_point = eval_accuracy(model_student, loader, std_text_features, DEVICE)
        acc_both = eval_accuracy(model_student, loader, prompted_text_features, DEVICE)
        toggle_point_prompt(model_student, pfgr_module, post_fps_prompt, False)
        print(f"Clean:        {acc_clean:.4f}")
        print(f"Text Prompt:  {acc_text:.4f}")
        print(f"Point Prompt: {acc_point:.4f}")
        print(f"Text + Point: {acc_both:.4f}")
