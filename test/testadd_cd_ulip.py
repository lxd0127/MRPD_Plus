import os
import sys
import gc
import argparse
import random
import warnings
from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from loguru import logger
from tqdm import tqdm
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "XXX"
DATA_ROOT = os.path.join(ROOT_DIR, 'XXX')
SAVE_DIR_WHITEBOX = os.path.join(ROOT_DIR, 'XXX')
WHITEBOX_PATH = os.path.join(SAVE_DIR_WHITEBOX, 'XXX')
CLEAN_NPZ_PATH = "XXX"
BLACKBOX_DATA_PATH = "XXX"
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
try:
    from utils.tokenizer import SimpleTokenizer
    import utils.open_clip as open_clip
    from utils.utils import cfg_from_yaml_file
    from models.ULIP_models import ULIP2_WITH_OPENCLIP
    from models.pointbert.point_encoder import PointTransformer_Colored
    from models.point_encoder import fps6d
except ImportError as e:
    logger.error(f"Module import failed, please check the path: {e}")
    sys.exit(1)
NUM_POINTS = 1024
NUM_CLASSES = 40
BATCH_SIZE_EVAL = 32
BATCH_SIZE_GEN = 16
SEED = 0
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
    try:
        loaded = torch.load(prompt_ckpt, map_location="cpu")
        if isinstance(loaded, dict):
            b.pfgr_weight = loaded
            logger.info("PFGR Weights dict loaded.")
        else:
            logger.warning(f"Main file is {type(loaded)}, skipping PFGR weight load.")
            b.pfgr_weight = None
    except Exception as e:
        logger.warning(f"Error loading PFGR weights: {e}")
        b.pfgr_weight = None
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
class _Distance(nn.Module):
    def __init__(self):
        super(_Distance, self).__init__()
        self.use_cuda = torch.cuda.is_available()
    def batch_pairwise_dist(self, x, y):
        if x.shape[-1] > 3:
            x = x[:, :, :3]
        if y.shape[-1] > 3:
            y = y[:, :, :3]
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        dtype = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P
class ChamferDistance(_Distance):
    def __init__(self):
        super(ChamferDistance, self).__init__()
    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss1 = torch.mean(mins, dim=1)
        mins, _ = torch.min(P, 2)
        loss2 = torch.mean(mins, dim=1)
        return loss1, loss2
chamfer = ChamferDistance()
class ChamferDist(nn.Module):
    def __init__(self, method="adv2ori"):
        super(ChamferDist, self).__init__()
        self.method = method
    def forward(self, adv_pc, ori_pc, weights=None, batch_avg=True):
        B = adv_pc.shape[0]
        if weights is None:
            weights = torch.ones((B,), device=adv_pc.device)
        loss1, loss2 = chamfer(adv_pc, ori_pc)
        if self.method == "adv2ori":
            loss = loss1
        elif self.method == "ori2adv":
            loss = loss2
        else:
            loss = (loss1 + loss2) / 2.0
        weights = weights.float()
        loss = loss * weights
        if batch_avg:
            return loss.mean()
        return loss
class CrossEntropyAdvLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyAdvLoss, self).__init__()
    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets)
def get_critical_points_flexible(model, pc, label, num, text_features):
    B, K, C = pc.shape
    input_pc = pc.clone().detach().float().to(DEVICE)
    input_pc.requires_grad_()
    label = label.long().to(DEVICE)
    point_features = model.encode_pc(input_pc)
    point_features = F.normalize(point_features, dim=-1)
    logits = torch.matmul(point_features, text_features.T)
    loss = F.cross_entropy(logits, label)
    loss.backward()
    with torch.no_grad():
        grad = input_pc.grad.data
        grad_xyz = grad[:, :, :3]
        grad_norm = torch.sum(grad_xyz ** 2, dim=-1)
        _, idx = grad_norm.topk(k=num, dim=-1)
        critical_points = torch.stack([
            pc[i, idx[i]] for i in range(B)
        ], dim=0).clone().detach()
    return critical_points
class CWAdd_Flexible:
    def __init__(self, model, adv_func, dist_func, text_features,
                 attack_lr=1e-2, init_weight=5e3, max_weight=4e4,
                 binary_step=10, num_iter=500, num_add=512):
        self.model = model.to(DEVICE).eval()
        self.adv_func = adv_func
        self.dist_func = dist_func
        self.text_features = text_features.to(DEVICE)
        self.attack_lr = attack_lr
        self.init_weight = init_weight
        self.max_weight = max_weight
        self.binary_step = binary_step
        self.num_iter = num_iter
        self.num_add = num_add
    def attack(self, data, target):
        B, K, C = data.shape
        data = data.float().to(DEVICE).detach()
        ori_data = data.clone().detach()
        target = target.long().to(DEVICE).detach()
        label_val = target.cpu().numpy()
        lower_bound = np.zeros((B,))
        upper_bound = np.ones((B,)) * self.max_weight
        current_weight = np.ones((B,)) * self.init_weight
        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, self.num_add, C))
        cri_data = get_critical_points_flexible(
            self.model, ori_data, target, self.num_add, self.text_features)
        for binary_step in range(self.binary_step):
            adv_data = cri_data + torch.randn((B, self.num_add, C)).to(DEVICE) * 1e-7
            adv_data.requires_grad_()
            bestdist = np.array([1e10] * B)
            bestscore = np.array([-1] * B)
            opt = optim.Adam([adv_data], lr=self.attack_lr)
            for iteration in range(self.num_iter):
                cat_data = torch.cat([ori_data, adv_data], dim=1)
                point_features = self.model.encode_pc(cat_data)
                point_features = F.normalize(point_features, dim=-1)
                logits = torch.matmul(point_features, self.text_features.T)
                pred = torch.argmax(logits, dim=-1)
                dist_val = self.dist_func(adv_data, ori_data, batch_avg=False).detach().cpu().numpy()
                pred_val = pred.detach().cpu().numpy()
                input_val = adv_data.detach().cpu().numpy()
                for e, (dist, pred_label, label, ii) in enumerate(zip(dist_val, pred_val, label_val, input_val)):
                    if dist < bestdist[e] and pred_label == label:
                        bestdist[e] = dist
                        bestscore[e] = pred_label
                    if dist < o_bestdist[e] and pred_label == label:
                        o_bestdist[e] = dist
                        o_bestscore[e] = pred_label
                        o_bestattack[e] = ii
                adv_loss = self.adv_func(logits, target).mean()
                dist_loss = self.dist_func(adv_data, ori_data, weights=torch.from_numpy(current_weight).float().to(DEVICE)).mean()
                loss = adv_loss + dist_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
            for e, label in enumerate(label_val):
                if bestscore[e] == label and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e]:
                    lower_bound[e] = max(lower_bound[e], current_weight[e])
                else:
                    upper_bound[e] = min(upper_bound[e], current_weight[e])
                current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.
        fail_idx = (lower_bound == 0.)
        o_bestattack[fail_idx] = input_val[fail_idx]
        success_num = (lower_bound > 0.).sum()
        ori_data = ori_data.detach().cpu().numpy()
        final_attack = np.concatenate([ori_data, o_bestattack], axis=1)
        return o_bestdist, final_attack, success_num
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def ensure_6ch(points: torch.Tensor) -> torch.Tensor:
    if points.shape[-1] == 3:
        colors = torch.full((points.shape[0], points.shape[1], 3), 0.5, device=points.device, dtype=points.dtype)
        points = torch.cat([points, colors], dim=-1)
    return points
class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split="modelnet40_test", transform=None):
        self.root_dir = root_dir
        self.transform = transform
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
        if self.transform:
            points = self.transform(points)
        return points, label
class NpzPointCloudDataset(Dataset):
    def __init__(self, npz_path: str):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"npz not found: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        pts = None
        for k in ['adv_pc', 'perturbed_points', 'points']:
            if k in data:
                pts = data[k]
                break
        if pts is None and 'test_pc' in data:
            logger.warning(f"Warning: Loading 'test_pc' (clean) from {npz_path}.")
            pts = data['test_pc']
        if pts is None:
            pts = data[list(data.keys())[0]]
        lbs = None
        for k in ['labels', 'test_label', 'adv_label', 'y']:
            if k in data:
                lbs = data[k]
                break
        if lbs is None:
            lbs = data[list(data.keys())[1]]
        self.points = torch.tensor(pts, dtype=torch.float32)
        self.labels = torch.tensor(lbs, dtype=torch.long)
        if self.points.shape[-1] == 3:
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
        raise FileNotFoundError(f"weights not found: {ULIP2_CKPT_PATH}")
    checkpoint = torch.load(ULIP2_CKPT_PATH, map_location="cpu")
    sd = checkpoint.get("state_dict", checkpoint)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model_state = model.state_dict()
    filtered_sd = {}
    pc_proj_key_ckpt = None
    if "pc_projection.weight" in sd: pc_proj_key_ckpt = "pc_projection.weight"
    elif "pc_projection" in sd: pc_proj_key_ckpt = "pc_projection"
    if pc_proj_key_ckpt:
        w = sd[pc_proj_key_ckpt]
        target_key = "pc_projection"
        target_shape = model_state[target_key].shape
        if w.shape == target_shape: filtered_sd[target_key] = w
        elif tuple(reversed(w.shape)) == target_shape: filtered_sd[target_key] = w.t()
    for k, v in sd.items():
        if "pc_projection" in k: continue
        if k not in model_state: continue
        target_shape = model_state[k].shape
        if v.shape == target_shape: filtered_sd[k] = v
        elif "first_conv" in k and v.ndim >= 2 and v.shape[1] == 6 and target_shape[1] == 3:
            filtered_sd[k] = v[:, :3, ...]
        elif tuple(reversed(v.shape)) == target_shape: filtered_sd[k] = v.t()
    model.load_state_dict(filtered_sd, strict=False)
    logger.info("ULIP-2 Student Loaded.")
    return model
@torch.no_grad()
def eval_accuracy(model, loader, text_features, point_prompt, device):
    model.eval()
    correct, total = 0, 0
    for points, labels in loader:
        points = points.to(device)
        labels = labels.to(device)
        points = ensure_6ch(points)
        B = int(points.shape[0])
        pc_feat = model.encode_pc(points)
        pc_feat = F.normalize(pc_feat, dim=-1)
        logits = pc_feat @ text_features.t()
        pred = logits.argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += B
    return float(correct) / max(1, total)
def generate_whitebox_add_cd(model, raw_dataset, text_features, save_path):
    print(f"\n[Info] Generating whitebox attack data (ADD-CD)...")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(raw_dataset, batch_size=BATCH_SIZE_GEN, shuffle=False, num_workers=4)
    model.eval()
    adv_func = CrossEntropyAdvLoss()
    dist_func = ChamferDist(method='adv2ori')
    attacker = CWAdd_Flexible(model, adv_func, dist_func, text_features,
                     attack_lr=1e-2, init_weight=2e2, max_weight=9e2,
                     binary_step=10, num_iter=500, num_add=512)
    all_points = []
    all_labels = []
    for i, (points, labels) in enumerate(loader):
        points = points.to(DEVICE)
        labels = labels.to(DEVICE)
        points = ensure_6ch(points)
        points = fps6d(points, 1024)
        target = torch.randint_like(labels, low=0, high=NUM_CLASSES)
        target = torch.where(target == labels, (target + 1) % NUM_CLASSES, target)
        _, perturbed_pc, _ = attacker.attack(points, target)
        perturbed_pc = torch.from_numpy(perturbed_pc).float().cpu()
        all_points.append(perturbed_pc)
        all_labels.append(labels.cpu())
        if i % 5 == 0: logger.info(f"Generating batch {i}/{len(loader)}")
        torch.cuda.empty_cache()
    all_points = torch.cat(all_points, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    np.savez(save_path, points=all_points, labels=all_labels)
    logger.info(f"Whitebox data saved to {save_path}")
def build_clean_npz_if_needed(raw_dataset):
    if os.path.exists(CLEAN_NPZ_PATH): return
    logger.info(f"Generating clean npz: {CLEAN_NPZ_PATH}")
    Path(SAVE_DIR_WHITEBOX).mkdir(parents=True, exist_ok=True)
    loader = DataLoader(raw_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=4)
    all_pts, all_lbl = [], []
    for pts, lbls in loader:
        pts = pts.to(DEVICE)
        pts = ensure_6ch(pts)
        pts = fps6d(pts, NUM_POINTS).detach().cpu()
        all_pts.append(pts)
        all_lbl.append(lbls)
    np.savez(CLEAN_NPZ_PATH, points=torch.cat(all_pts, dim=0).numpy(), labels=torch.cat(all_lbl, dim=0).numpy())
def run_ablation(title, dataset, model, std_text_features, prompted_text_features, pfgr_module, post_fps_prompt):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=4)
    print(f"\n===== {title} =====")
    toggle_point_prompt(model, pfgr_module, post_fps_prompt, False)
    acc_clean = eval_accuracy(model, loader, std_text_features, None, DEVICE)
    acc_text = eval_accuracy(model, loader, prompted_text_features, None, DEVICE)
    toggle_point_prompt(model, pfgr_module, post_fps_prompt, True)
    acc_point = eval_accuracy(model, loader, std_text_features, None, DEVICE)
    acc_both = eval_accuracy(model, loader, prompted_text_features, None, DEVICE)
    toggle_point_prompt(model, pfgr_module, post_fps_prompt, False)
    print(f"Clean:        {acc_clean:.4f}")
    print(f"Text Prompt:  {acc_text:.4f}")
    print(f"Point Prompt: {acc_point:.4f}")
    print(f"Text + Point: {acc_both:.4f}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_whitebox_gen", action="store_true", help="Do not automatically generate whitebox")
    args = parser.parse_args()
    set_seed(SEED)
    logger.info("Initializing CLIP ...")
    try:
        clip_model, _, _ = open_clip.create_model_and_transforms_text("ViT-bigG-14", pretrained=None)
    except:
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained=None)
    from safetensors.torch import load_file
    if os.path.exists(CLIP_MODEL_PATH):
        sd = load_file(CLIP_MODEL_PATH)
        clip_model.load_state_dict(sd, strict=False)
    clip_model = clip_model.to(DEVICE).eval()
    tokenizer = SimpleTokenizer()
    temp_dataset = ModelNet40Dataset(DATA_ROOT, split="modelnet40_test")
    classnames = [name.replace("_", " ") for name in temp_dataset.classnames]
    logger.info("Initializing ULIP-2 student ...")
    class DummyCLIP:
        def __init__(self): self.visual = nn.Identity()
    model = build_ulip2_student(DummyCLIP()).to(DEVICE).eval()
    logger.info(f"Loading NEW best prompt: {PROMPT_CKPT}")
    bundle = load_prompt_bundle(PROMPT_CKPT, clip_model, tokenizer, classnames, DEVICE)
    pfgr_module = install_pfgr(model, DEVICE)
    if bundle.pfgr_weight is not None:
        pfgr_module.pfgr_branch.load_state_dict(bundle.pfgr_weight, strict=False)
        logger.info("PFGR Weights loaded.")
    post_fps_prompt = None
    if bundle.point_prompt_xyz is not None:
        post_fps_prompt = PostFPSPointPrompt(bundle.point_prompt_xyz).to(DEVICE)
        logger.info("Post-FPS Prompt created.")
    torch.cuda.empty_cache()
    gc.collect()
    std_text_features = build_std_text_features(clip_model, tokenizer, classnames, DEVICE, batch_size=2)
    prompted_text_features = build_prompted_text_features(clip_model, tokenizer, classnames, bundle, DEVICE, batch_size=2)
    torch.cuda.empty_cache()
    raw_test_dataset = ModelNet40Dataset(DATA_ROOT, split="modelnet40_test", transform=transform)
    build_clean_npz_if_needed(raw_test_dataset)
    toggle_point_prompt(model, pfgr_module, post_fps_prompt, False)
    if os.path.exists(CLEAN_NPZ_PATH):
        ds_clean = NpzPointCloudDataset(CLEAN_NPZ_PATH)
        run_ablation("Clean", ds_clean, model, std_text_features, prompted_text_features, pfgr_module, post_fps_prompt)
    if not os.path.exists(WHITEBOX_PATH) and not args.skip_whitebox_gen:
        generate_whitebox_add_cd(model, raw_test_dataset, std_text_features, WHITEBOX_PATH)
    if os.path.exists(WHITEBOX_PATH):
        ds_wb = NpzPointCloudDataset(WHITEBOX_PATH)
        run_ablation("ADD-CD Whitebox", ds_wb, model, std_text_features, prompted_text_features, pfgr_module, post_fps_prompt)
    else:
        logger.warning("Whitebox data missing.")
    if os.path.exists(BLACKBOX_DATA_PATH):
        ds_bb = NpzPointCloudDataset(BLACKBOX_DATA_PATH)
        run_ablation("ADD-CD Blackbox", ds_bb, model, std_text_features, prompted_text_features, pfgr_module, post_fps_prompt)
    else:
        logger.warning("Blackbox data missing.")
    logger.info("Tests completed.")
if __name__ == "__main__":
    main()
