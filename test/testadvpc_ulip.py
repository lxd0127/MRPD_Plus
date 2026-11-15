import os
import sys
import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from loguru import logger
from tqdm import tqdm
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
try:
    import utils.open_clip as open_clip
    from models.ULIP_models import ULIP2_WITH_OPENCLIP
    from models.pointbert.point_encoder import PointTransformer_Colored
    from utils.utils import cfg_from_yaml_file
    from utils.tokenizer import SimpleTokenizer
    from models.point_encoder import fps6d
    from AdvPC import CWAdvPC
    import encoders_decoders
except ImportError as e:
    logger.error(f"Module import failed, please ensure you are running in the correct project path: {e}")
    sys.exit(1)
class Config:
    AE_MODEL_PATH = "XXX"
    WHITEBOX_SAVE_DIR = "XXX"
    WHITEBOX_FILE_NAME = 'XXX'
    BLACKBOX_DATA_PATH = "XXX"
    CLEAN_DATA_PATH = "XXX"
    MODELNET40_PATH = "XXX"
    CLIP_MODEL_PATH = "XXX"
    POINTBERT_CONFIG = "XXX"
    ULIP2_PRETRAIN_CKPT = "XXX"
    BEST_PROMPT_PATH = "XXX"
    PC_MODEL = "eva02_base_patch14_448"
    EMBED_DIM = 1280
    PC_FEAT_DIM = 768
    NUM_POINTS = 1024
    BATCH_SIZE_GEN = 16
    BATCH_SIZE_EVAL = 32
    NUM_WORKERS = 1
args = Config()
os.makedirs(args.WHITEBOX_SAVE_DIR, exist_ok=True)
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
        loaded_data = torch.load(prompt_ckpt, map_location="cpu")
        if isinstance(loaded_data, dict):
            b.pfgr_weight = loaded_data
            logger.info("Successfully loaded PFGR weights dict.")
        else:
            logger.warning(f"Warning: Main prompt file contains {type(loaded_data)} instead of dict. Skipping PFGR weight loading.")
            b.pfgr_weight = None
    except Exception as e:
        logger.warning(f"Failed to load PFGR weights: {e}")
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
transform = transforms.Compose([
    lambda pc: np.hstack((pc[:, :3], np.full((pc.shape[0], 3), 0.5, dtype=np.float32))),
    lambda pc: torch.tensor(pc, dtype=torch.float32)
])
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
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        keys = list(data.keys())
        pts = None
        for k in ['adv_pc', 'perturbed_points', 'points']:
            if k in data:
                pts = data[k]
                break
        if pts is None and 'test_pc' in data:
            logger.warning(f"Warning: Loading 'test_pc' (clean) from {npz_path}. If this is an attack file, verify keys.")
            pts = data['test_pc']
        if pts is None:
            pts = data[keys[0]]
        lbs = None
        for k in ['labels', 'test_label', 'adv_label', 'y']:
            if k in data:
                lbs = data[k]
                break
        if lbs is None:
            lbs = data[keys[1]]
        pts = pts.astype(np.float32)
        lbs = lbs.astype(np.int64)
        if pts.shape[-1] == 3:
            rgb = np.ones((pts.shape[0], pts.shape[1], 3), dtype=np.float32) * 0.5
            pts = np.concatenate([pts, rgb], axis=-1)
        self.points = torch.tensor(pts, dtype=torch.float32)
        self.labels = torch.tensor(lbs, dtype=torch.long)
    def __len__(self):
        return int(self.points.shape[0])
    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]
def get_ulip2_pointbert_student(open_clip_model):
    config = cfg_from_yaml_file(args.POINTBERT_CONFIG)
    class ArgsMock:
        pc_model = args.PC_MODEL
        pc_feat_dim = args.PC_FEAT_DIM
        drop_path_rate = 0.1
        pc_encoder_dim = 512
        num_group = 512
        group_size = 64
    point_encoder = PointTransformer_Colored(config.model, args=ArgsMock())
    model = ULIP2_WITH_OPENCLIP(
        open_clip_model=open_clip_model,
        point_encoder=point_encoder,
        pc_feat_dims=768,
        embed_dim=args.EMBED_DIM
    )
    ckpt_path = args.ULIP2_PRETRAIN_CKPT
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        sd = checkpoint.get("state_dict", checkpoint)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model_state = model.state_dict()
        filtered_sd = {}
        for k, v in sd.items():
            if "pc_projection" in k:
                if k in model_state and v.shape == model_state[k].shape:
                    filtered_sd[k] = v
                elif k in model_state:
                    filtered_sd[k] = v.t()
            elif k in model_state and v.shape == model_state[k].shape:
                filtered_sd[k] = v
        model.load_state_dict(filtered_sd, strict=False)
        logger.info("Student Model (ULIP-2) loaded.")
    return model
class ChamferDist(nn.Module):
    def __init__(self, method="adv2ori"):
        super().__init__()
        self.method = method
        self.chamfer_layer = encoders_decoders.ChamferDistance()
    def forward(self, adv_pc, ori_pc, weights=None, batch_avg=True):
        B = adv_pc.shape[0]
        if weights is None: weights = torch.ones((B,), device=adv_pc.device)
        loss1, loss2 = self.chamfer_layer(adv_pc, ori_pc)
        loss = loss1 if self.method == "adv2ori" else (loss1 + loss2) / 2.0
        return (loss * weights).mean() if batch_avg else loss * weights
class CrossEntropyAdvLoss(nn.Module):
    def forward(self, logits, targets): return F.cross_entropy(logits, targets)
class ClipPointsLinf(nn.Module):
    def __init__(self, budget):
        super().__init__()
        self.budget = budget
    def forward(self, pc, ori_pc):
        with torch.no_grad():
            diff = pc - ori_pc
            norm = torch.sum(diff ** 2, dim=1) ** 0.5
            scale = self.budget / (norm + 1e-9)
            scale = torch.clamp(scale, max=1.0)
            diff = diff * scale[:, None, :]
            pc = ori_pc + diff
        return pc
def generate_whitebox_advpc(model, text_features, dataset):
    logger.info(">>> Generating whitebox AdvPC attack data...")
    ae_model = encoders_decoders.AutoEncoder(3)
    target_dim = 1024 * 3
    if ae_model.decoder.fc_layers[2].out_features != target_dim:
        ae_model.decoder.fc_layers[2] = nn.Linear(ae_model.decoder.fc_layers[2].in_features, target_dim)
    ae_state = torch.load(args.AE_MODEL_PATH, map_location="cpu")
    if len(ae_state) > 0 and list(ae_state.keys())[0].startswith("module."):
        ae_state = {k[7:]: v for k, v in ae_state.items()}
    ae_model.load_state_dict(ae_state)
    ae_model.to(DEVICE).eval()
    clip_func = ClipPointsLinf(budget=0.18)
    dist_func = ChamferDist()
    adv_func = CrossEntropyAdvLoss()
    attacker = CWAdvPC(
        model, ae_model, adv_func, dist_func,
        attack_lr=1e-2, binary_step=2, num_iter=200, GAMMA=0.25,
        clip_func=clip_func, text_features=text_features
    )
    loader = DataLoader(dataset, batch_size=args.BATCH_SIZE_GEN, shuffle=False, num_workers=args.NUM_WORKERS)
    model.eval()
    all_pts, all_lbl = [], []
    for i, (pts, lbl) in enumerate(tqdm(loader)):
        pts = pts.to(DEVICE)
        lbl = lbl.to(DEVICE)
        if pts.shape[1] > args.NUM_POINTS: pts = fps6d(pts, args.NUM_POINTS)
        pts_xyz = pts[:, :, :3].contiguous()
        target = torch.randint_like(lbl, 0, 40)
        target = torch.where(target==lbl, (target+1)%40, target)
        _, adv_np, _ = attacker.attack(pts_xyz, target, lbl)
        adv_xyz = torch.from_numpy(adv_np).to(DEVICE).float()
        if pts.shape[2] == 6:
            adv_pts = torch.cat([adv_xyz, pts[:, :, 3:]], dim=2)
        else:
            adv_pts = torch.cat([adv_xyz, torch.full_like(adv_xyz, 0.5)], dim=2)
        all_pts.append(adv_pts.cpu())
        all_lbl.append(lbl.cpu())
        torch.cuda.empty_cache()
    all_pts = torch.cat(all_pts).numpy()
    all_lbl = torch.cat(all_lbl).numpy()
    save_path = os.path.join(args.WHITEBOX_SAVE_DIR, args.WHITEBOX_FILE_NAME)
    np.savez(save_path, points=all_pts, labels=all_lbl)
    logger.info(f"Saved: {save_path}")
    del ae_model, attacker
    gc.collect()
    return save_path
def eval_accuracy(model, loader, text_features, device=DEVICE):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for pts, lbl in loader:
            pts = pts.to(device)
            lbl = lbl.to(device)
            if pts.shape[1] > args.NUM_POINTS: pts = fps6d(pts, args.NUM_POINTS)
            feats = model.encode_pc(pts)
            feats = F.normalize(feats, dim=-1)
            pred = (feats @ text_features.t()).argmax(dim=-1)
            correct += (pred == lbl).sum().item()
            total += lbl.numel()
    return correct / max(total, 1)
def run_ablation(title, model, dataset, std_text, prompted_text, pfgr_module, post_fps_prompt):
    loader = DataLoader(dataset, batch_size=args.BATCH_SIZE_EVAL, shuffle=False, num_workers=args.NUM_WORKERS)
    print(f"\n===== {title} =====")
    toggle_point_prompt(model, pfgr_module, post_fps_prompt, False)
    acc_clean = eval_accuracy(model, loader, std_text, DEVICE)
    acc_text = eval_accuracy(model, loader, prompted_text, DEVICE)
    toggle_point_prompt(model, pfgr_module, post_fps_prompt, True)
    acc_point = eval_accuracy(model, loader, std_text, DEVICE)
    acc_both = eval_accuracy(model, loader, prompted_text, DEVICE)
    toggle_point_prompt(model, pfgr_module, post_fps_prompt, False)
    print(f"Clean:        {acc_clean:.4f}")
    print(f"Text Prompt:  {acc_text:.4f}")
    print(f"Point Prompt: {acc_point:.4f}")
    print(f"Text + Point: {acc_both:.4f}")
def main():
    logger.info("Init CLIP...")
    from safetensors.torch import load_file
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained=None)
    clip_model.load_state_dict(load_file(args.CLIP_MODEL_PATH), strict=False)
    clip_model.to(DEVICE).eval()
    ds = ModelNet40Dataset(root_dir=args.MODELNET40_PATH, split="modelnet40_test")
    classnames = [n.replace("_", " ") for n in ds.classnames]
    logger.info(f"Loading Prompt: {args.BEST_PROMPT_PATH}")
    bundle = load_prompt_bundle(args.BEST_PROMPT_PATH, clip_model, tokenizer, classnames, DEVICE)
    torch.cuda.empty_cache()
    std_text_features = build_std_text_features(clip_model, tokenizer, classnames, DEVICE, batch_size=2)
    prompted_text_features = build_prompted_text_features(clip_model, tokenizer, classnames, bundle, DEVICE, batch_size=2)
    model_student = get_ulip2_pointbert_student(clip_model).to(DEVICE).eval()
    pfgr_module = install_pfgr(model_student, DEVICE)
    if bundle.pfgr_weight is not None:
        pfgr_module.pfgr_branch.load_state_dict(bundle.pfgr_weight, strict=False)
        logger.info("PFGR Weights loaded.")
    post_fps_prompt = None
    if bundle.point_prompt_xyz is not None:
        post_fps_prompt = PostFPSPointPrompt(bundle.point_prompt_xyz).to(DEVICE)
        logger.info("Post-FPS Prompt created.")
    toggle_point_prompt(model_student, pfgr_module, post_fps_prompt, False)
    if os.path.exists(args.CLEAN_DATA_PATH):
        clean_ds = NpzPointCloudDataset(args.CLEAN_DATA_PATH)
        run_ablation("Clean", model_student, clean_ds, std_text_features, prompted_text_features, pfgr_module, post_fps_prompt)
    wb_path = os.path.join(args.WHITEBOX_SAVE_DIR, args.WHITEBOX_FILE_NAME)
    if not os.path.exists(wb_path):
        gen_ds = ModelNet40Dataset(root_dir=args.MODELNET40_PATH, split="modelnet40_test", transform=transform)
        wb_path = generate_whitebox_advpc(model_student, std_text_features, gen_ds)
    wb_ds = NpzPointCloudDataset(wb_path)
    run_ablation("AdvPC Whitebox", model_student, wb_ds, std_text_features, prompted_text_features, pfgr_module, post_fps_prompt)
    if os.path.exists(args.BLACKBOX_DATA_PATH):
        bb_ds = NpzPointCloudDataset(args.BLACKBOX_DATA_PATH)
        run_ablation("AdvPC Blackbox", model_student, bb_ds, std_text_features, prompted_text_features, pfgr_module, post_fps_prompt)
    logger.info("All tests completed.")
if __name__ == "__main__":
    main()
