import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import glob
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
# Optional: only import torchvision if we actually need a ResNet


class ImageNormalizer:
    """
    Normalizes CHW images using given per-channel mean/std (float32).
    Expects inputs in [0,1].
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
        self.std  = torch.clamp(torch.tensor(std, dtype=torch.float32).view(3,1,1), min=1e-3)
    def __call__(self, x):  # x: [3,H,W] in [0,1]
        return (x - self.mean) / self.std


# ---------- Legacy small CNN (for old checkpoints) ----------
class ConvBlock(torch.nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = torch.nn.BatchNorm2d(c_out)
        self.act  = torch.nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class PolicyCNN(torch.nn.Module):
    def __init__(self, n_actions=16):
        super().__init__()
        self.stem = torch.nn.Sequential(
            ConvBlock(3, 32, k=7, s=2, p=3),
            ConvBlock(32, 32),
            torch.nn.MaxPool2d(2),
        )
        self.stage2 = torch.nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            torch.nn.MaxPool2d(2),
        )
        self.stage3 = torch.nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            torch.nn.MaxPool2d(2),
        )
        self.stage4 = torch.nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )
        self.head = torch.nn.Linear(256, n_actions)
    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.head(x)


# ---------- New ResNet policy (for new checkpoints) ----------
class ResNetPolicy(torch.nn.Module):
    """
    Thin wrapper around torchvision ResNet-18/34 to set classifier head.
    """
    def __init__(self, n_actions=16, backbone='resnet18'):
        super().__init__()
        from torchvision.models import resnet18, resnet34
        ResNetPolicy._resnet18 = resnet18
        ResNetPolicy._resnet34 = resnet34

        if backbone == 'resnet18':
            self.backbone = ResNetPolicy._resnet18(weights=None)
        elif backbone == 'resnet34':
            self.backbone = ResNetPolicy._resnet34(weights=None)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_features, n_actions)

    def forward(self, x):
        return self.backbone(x)


def _to_chw(img_hwc: np.ndarray) -> torch.Tensor:
    """Convert HWC numpy array to CHW torch tensor."""
    return torch.from_numpy(np.transpose(img_hwc, (2, 0, 1))).float()


def _resize_chw(x_chw: torch.Tensor, target=256) -> torch.Tensor:
    """Resize CHW tensor to target size."""
    x = x_chw.unsqueeze(0)  # [1,C,H,W]
    x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
    return x.squeeze(0)


@torch.no_grad()
def extract_resnet_features(frame_hw3, normalizer, model, device, target=256):
    """
    Deprecated: PU models removed; this function is unused and kept only for backward compatibility.
    """
    raise RuntimeError("extract_resnet_features is no longer supported (PU models removed).")


# ---------- Model builder that understands both checkpoint formats ----------
def _parse_arch(ckpt):
    """
    Returns ('resnet18' or 'resnet34', 'resnet') for new checkpoints,
    or ('legacy', 'cnn') for old ones without arch info.
    """
    arch = ckpt.get('arch', '')
    if arch.startswith('ResNetPolicy_'):
        parts = arch.split('_')
        if len(parts) >= 2 and parts[1] in ('resnet18', 'resnet34'):
            return parts[1], 'resnet'
    return 'legacy', 'cnn'


def _build_model_from_ckpt(ckpt, device):
    n_actions = int(ckpt.get('n_actions', 16))
    backbone, kind = _parse_arch(ckpt)
    if kind == 'resnet':
        model = ResNetPolicy(n_actions=n_actions, backbone=backbone).to(device)
    else:
        model = PolicyCNN(n_actions=n_actions).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    return model, n_actions


# ---------- I/O + preprocessing ----------
def preprocess_frame(frame_hw3, normalizer, target=256):
    """
    frame_hw3: numpy array [H,W,3], float32 or uint8; scaled to [0,1] here.
    returns torch tensor [1,3,target,target]
    """
    assert frame_hw3.ndim == 3 and frame_hw3.shape[2] == 3
    x_np = frame_hw3.astype(np.float32)
    if x_np.max() > 1.0:  # support uint8 legacy inputs
        x_np = x_np / 255.0
    x = torch.from_numpy(np.transpose(x_np, (2,0,1))).float()   # [3,H,W]
    x = F.interpolate(x.unsqueeze(0), size=(target, target), mode='bilinear', align_corners=False).squeeze(0)  # [3,T,T]
    x = normalizer(x)
    return x.unsqueeze(0)  # [1,3,T,T]


# ---------- Public API ----------
def load_policy(ckpt_path, device=None):
    """
    Load model + normalizer from a saved training checkpoint.
    Backward-compatible with legacy CNN checkpoints and new ResNet checkpoints.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else
                                    ('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'))
    ckpt = torch.load(ckpt_path, map_location=device)

    model, n_actions = _build_model_from_ckpt(ckpt, device)
    mean = ckpt.get('mean', (0.485, 0.456, 0.406))
    std  = ckpt.get('std',  (0.229, 0.224, 0.225))
    normalizer = ImageNormalizer(mean, std)

    return model, normalizer, device, n_actions


@torch.no_grad()
def act_greedy(model, normalizer, device, frame_hw3, target=256):
    x = preprocess_frame(frame_hw3, normalizer, target=target).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    action = int(torch.argmax(probs).item())
    return action, probs.detach().cpu().numpy()


@torch.no_grad()
def act_sample(model, normalizer, device, frame_hw3, temperature=1.0, target=256):
    x = preprocess_frame(frame_hw3, normalizer, target=target).to(device)
    logits = model(x).squeeze(0)
    if temperature != 1.0:
        logits = logits / max(1e-6, float(temperature))
    probs = torch.softmax(logits, dim=-1)
    action = int(torch.multinomial(probs, num_samples=1).item())
    return action, probs.detach().cpu().numpy()


def bc_policy(models, state, skill):
    assert state.max() > 1.0  # allow uint8 input
    state = np.asarray(state).astype(np.float32) / 255.0
    model, normalizer, device, n_actions = models["bc_models"][skill]
    action, _ = act_greedy(model, normalizer, device, state)
    return action


# ---------- PU start / end utilities (removed) ----------
def load_pu_start_models(models_dir: str):
    raise RuntimeError("PU models are no longer supported.")

def applicable_pu_start_models(models, state, *, return_details=False, eps=0.0):
    raise RuntimeError("PU models are no longer supported.")

def load_pu_end_model(models_dir: str, skill: str):
    raise RuntimeError("PU models are no longer supported.")

def predict_pu_end_state(model: dict, state) -> dict:
    raise RuntimeError("PU models are no longer supported.")


# ---------- ResNet feature extractor + model bank loaders ----------
def load_all_models(skill_list = ['wood', 'stone', 'wood_pickaxe', 'stone_pickaxe', 'table'],
                    root: str = 'Traces/stone_pickaxe_easy',
                    bc_checkpoint_dir: str = 'bc_checkpoints_resnet',
                    dataset_mean_std_path: str = 'dataset_mean_std.npy',
                    pu_start_models_dir: str = 'pu_start_models',
                    pu_end_models_dir: str = 'pu_end_models',
                    backbone: str = 'resnet34'):
    bc_models = {}
    for skill in skill_list:
        ckpt_path = os.path.join(root, bc_checkpoint_dir, f'{skill}_policy_resnet34_pt.pt')
        bc_models[skill] = load_policy(ckpt_path)

    return {
        "skills": skill_list,  # canonical order
        "bc_models": bc_models,
    }


def available_skills(models, state):
    # PU gating removed: all skills are always available
    order = models["skills"]
    return np.ones(len(order), dtype=bool)


class FixedSeedAlways(gym.Wrapper):
    """
    Forces the same seed on *every* reset call, regardless of what the caller passes.
    This guarantees full determinism of the environment initial state across episodes.
    """
    def __init__(self, env, seed: int = 1000):
        super().__init__(env)
        self._seed = int(seed)

    def reset(self, *, seed=None, options=None, **kwargs):
        # Ignore any provided seed and enforce the fixed one
        kwargs.pop("seed", None)
        return self.env.reset(seed=self._seed, options=options, **kwargs)

    # (Optional) If someone calls env.seed(...), force our fixed seed too
    def seed(self, seed=None):
        # Some envs still expose a seed() method; keep them pinned
        if hasattr(self.env, "seed") and callable(getattr(self.env, "seed")):
            return self.env.seed(self._seed)
        return [self._seed]


def to_gif_frame(obs):
    import numpy as np
    arr = np.asarray(obs)

    # remove vec batch dim if present (N=1)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]                    # (C,H,W)

    # CHW -> HWC if needed
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))   # (H,W,C)

    # if single channel, replicate to RGB
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)      # (H,W,3)

    # scale/clip & cast to uint8
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255)
        if arr.max() <= 1.0:
            arr = (arr * 255.0).round()
        arr = arr.astype(np.uint8)

    # if still grayscale 2D, OK for GIF; otherwise ensure HWC
    return arr

# ============================================================
# ===============  ROBUST HIERARCHY FIX BELOW  ===============
# ============================================================

# Runtime data structures:
#   models["composite_runtime"] = {
#       skill_name: {
#           call_id_string: {"idx": int}     # instance/phase state for each invocation
#       },
#       ...
#   }
#   models["call_id_ctr"] = int (monotonic counter)

def _get_seq_for_skill(models: Dict[str, Any], skill: str) -> Optional[List[str]]:
    entry = models["bc_models"].get(skill, None)
    if isinstance(entry, tuple) and entry and entry[0] == "__COMPOSITE__":
        return list(entry[1])
    return None

def _ensure_runtime_dict(models: Dict[str, Any]) -> None:
    if "composite_runtime" not in models:
        models["composite_runtime"] = {}
    if "call_id_ctr" not in models:
        models["call_id_ctr"] = 0

def new_call_id(models: Dict[str, Any]) -> int:
    _ensure_runtime_dict(models)
    models["call_id_ctr"] += 1
    return models["call_id_ctr"]

def _rt_for(models: Dict[str, Any], skill: str, call_id: Optional[int]) -> Dict[str, Any]:
    """
    Obtain per-(skill,call_id) runtime dictionary. Falls back to a legacy
    per-skill entry if call_id is None.
    """
    _ensure_runtime_dict(models)
    cr = models["composite_runtime"].setdefault(skill, {})
    key = str(call_id) if call_id is not None else "_legacy_"
    return cr.setdefault(key, {"idx": 0})

def _clear_rt(models: Dict[str, Any], skill: str, call_id: Optional[int]) -> None:
    if "composite_runtime" not in models:
        return
    cr = models["composite_runtime"].get(skill, {})
    key = str(call_id) if call_id is not None else "_legacy_"
    cr.pop(key, None)


def should_terminate(models, state, skill, call_id: Optional[int] = None):
    """
    PU termination removed: never terminate via classifier.
    The environment will end options only on episode end or budget exhaustion.
    """
    return False


def _primitive_should_terminate(models, state, skill) -> bool:
    # Legacy shim; PU removed, never terminate based on classifier
    return False


# option_helpers.py
def bc_policy_hierarchy(models, state, skill, call_id: Optional[int] = None, max_leaf_len: Optional[int] = None):
    entry = models["bc_models"][skill]

    if isinstance(entry, tuple) and entry and entry[0] == "__COMPOSITE__":
        seq: List[str] = entry[1]
        rt = _rt_for(models, skill, call_id)
        idx = int(rt.get("idx", 0))
        leaf_steps = int(rt.get("leaf_steps", 0))

        # Advance only by per-leaf cap (PU termination removed)
        if max_leaf_len is not None and idx < len(seq) and leaf_steps >= max_leaf_len:
            idx += 1
            rt["idx"] = idx
            rt["leaf_steps"] = 0
            leaf_steps = 0

        if idx >= len(seq):
            last = seq[-1]
            return bc_policy(models, state, last)

        cur = seq[idx]
        # Count one step on this leaf
        if max_leaf_len is not None:
            rt["leaf_steps"] = leaf_steps + 1

        return bc_policy(models, state, cur)

    # primitive case unchanged
    state = np.asarray(state)
    assert state.ndim == 3 and state.shape[-1] == 3, "state must be HxWx3"
    model, normalizer, device, n_actions = entry
    action, _ = act_greedy(model, normalizer, device, state)
    return action


# ===== Hierarchy building utilities =====

import hashlib

def _canonicalize_tree(node: Dict[str, Any]) -> str:
    if "symbol" in node:
        return f"S:{node['symbol']}"
    pid = node.get("production", None)
    children = node.get("children", [])
    sig_children = ",".join(_canonicalize_tree(c) for c in children)
    return f"P:{pid}[{sig_children}]"

def _iter_production_nodes(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    if "production" in node:
        out.append(node)
    for c in node.get("children", []):
        out.extend(_iter_production_nodes(c))
    return out

def _leaf_symbols_inorder(node: Dict[str, Any]) -> List[str]:
    if "symbol" in node:
        return [str(node["symbol"])]
    seq = []
    for c in node.get("children", []):
        seq.extend(_leaf_symbols_inorder(c))
    return seq

def load_unique_hierarchies(hierarchies_dir: str) -> List[Dict[str, Any]]:
    uniq = {}
    if not os.path.isdir(hierarchies_dir):
        print(f"[WARN] hierarchies_dir not found: {hierarchies_dir}")
        return []
    for fname in os.listdir(hierarchies_dir):
        if not fname.lower().endswith(".json"):
            continue
        fpath = os.path.join(hierarchies_dir, fname)
        try:
            with open(fpath, "r") as f:
                tree = json.load(f)
            sig = _canonicalize_tree(tree)
            h = hashlib.sha1(sig.encode("utf-8")).hexdigest()
            if h not in uniq:
                uniq[h] = tree
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")
    return list(uniq.values())


def compile_productions_into_skill_bank(
    models: Dict[str, Any],
    hierarchies: List[Dict[str, Any]],
    symbol_map: Dict[str, str],
) -> None:
    """
    For each production node P:
      - Create a pseudo-skill 'Production_<P>'
      - bc model := sentinel ('__COMPOSITE__', [child_skill_names...])
      PU models removed: composites are always included and available.
    """
    skills_set     = set(models["skills"])

    if "composite_runtime" not in models:
        models["composite_runtime"] = {}
    if "call_id_ctr" not in models:
        models["call_id_ctr"] = 0

    composite_names = []
    for tree in hierarchies:
        for pnode in _iter_production_nodes(tree):
            pid = int(pnode["production"])
            leaves = _leaf_symbols_inorder(pnode)
            if not leaves:
                continue

            try:
                seq = [symbol_map[str(s)] for s in leaves]
            except KeyError as e:
                print(f"[WARN] Missing symbol in symbol_map: {e}; skipping Production_{pid}")
                continue

            name = f"Production_{pid}"

            if name in models["bc_models"]:
                continue

            # 1) BC model sentinel (sequence of leaves)
            models["bc_models"][name] = ("__COMPOSITE__", seq)

            # 2) Expose in skills list
            if name not in skills_set:
                models["skills"].append(name)
                skills_set.add(name)
                composite_names.append(name)

    if composite_names:
        print(f"[INFO] Added composites: {', '.join(composite_names)}")


def load_all_models_hierarchy(
    skill_list = ['wood', 'stone', 'wood_pickaxe', 'stone_pickaxe', 'table'],
    hierarchies_dir: Optional[str] = None,
    symbol_map:     Optional[Dict[str, str]] = None,
    root:           str = 'Traces/stone_pickaxe_easy',
    backbone_hint:  str = 'resnet34',
    bc_checkpoint_dir: str = 'bc_checkpoints_resnet',
    dataset_mean_std_path: str = 'dataset_mean_std.npy',
    pu_start_models_dir: str = 'pu_start_models',
    pu_end_models_dir: str = 'pu_end_models',
):
    ckpt_dir = os.path.join(root, bc_checkpoint_dir)
    bc_models = {}

    def _find_ckpt(skill: str) -> str:
        cand = os.path.join(ckpt_dir, f'{skill}_policy_{backbone_hint}_pt.pt')
        if os.path.exists(cand):
            return cand
        hits = sorted(glob.glob(os.path.join(ckpt_dir, f'{skill}_policy_resnet*_pt.pt')))
        if len(hits) > 0:
            return hits[0]
        legacy = os.path.join(ckpt_dir, f'{skill}_policy_cnn.pt')
        if os.path.exists(legacy):
            return legacy
        raise FileNotFoundError(
            f"No checkpoint found for skill '{skill}'. "
            f"Tried '{cand}', any resnet*, and legacy '{legacy}'."
        )

    for skill in skill_list:
        ckpt_path = _find_ckpt(skill)
        bc_models[skill] = load_policy(ckpt_path)

    models = {
        "skills": list(skill_list),
        "bc_models": bc_models,                 # {skill: (model,..) or ("__COMPOSITE__", seq)}
        "composite_runtime": {},                # instance state
        "call_id_ctr": 0,                       # instance id counter
    }

    if hierarchies_dir and symbol_map:
        uniq = load_unique_hierarchies(hierarchies_dir)
        compile_productions_into_skill_bank(models, uniq, symbol_map)
    else:
        if hierarchies_dir and not os.path.isdir(hierarchies_dir):
            print(f"[WARN] hierarchies_dir not found: {hierarchies_dir}")
        if hierarchies_dir and not symbol_map:
            print(f"[WARN] hierarchies_dir provided but no symbol_map; skipping composite productions.")

    return models


# (Optional) demo
if __name__ == "__main__":
    models = load_all_models_hierarchy(
        skill_list = ['wood', 'stone', 'wood_pickaxe', 'stone_pickaxe', 'table'],
        hierarchies_dir = 'Traces/stone_pickaxe_easy/hierarchy_data/Simple',
        symbol_map = {
            "0": "stone",
            "1": "stone_pickaxe",
            "2": "table",
            "3": "wood",
            "4": "wood_pickaxe",
        }
    )
    print(f"Final skills: {models['skills']}")
    print(f"BC models: {list(models['bc_models'].keys())}")
    print(f"Termination models: {list(models['termination_models'].keys())}")
    print(f"Start models: {[m['skill'] for m in models['start_models']]}")
