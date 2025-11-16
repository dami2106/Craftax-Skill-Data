import torch
import torch.nn.functional as F
import numpy as np
import os
import joblib
import json
from joblib import load as joblib_load
import glob
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
# Optional: only import torchvision if we actually need a ResNet
_RESNET_IMPORTED = False


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
        global _RESNET_IMPORTED
        if not _RESNET_IMPORTED:
            from torchvision.models import resnet18, resnet34
            ResNetPolicy._resnet18 = resnet18
            ResNetPolicy._resnet34 = resnet34
            _RESNET_IMPORTED = True

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


# ---------- PCA MLP policy (for PCA-based BC) ----------
class PCAPolicy(torch.nn.Module):
    """
    Small MLP for non-linear action prediction from PCA features.
    Allows the model to learn non-linear decision boundaries in PCA space.
    """
    def __init__(self, z_dim, n_actions):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(z_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions),
        )

    def forward(self, z):
        return self.net(z)  # logits over n_actions


class PCAFeatureNormalizer:
    """
    Normalizes PCA features using given mean/std (float32).
    """
    def __init__(self, mean_vec, std_vec):
        self.mean = torch.from_numpy(mean_vec).float()  # [D]
        self.std = torch.from_numpy(std_vec).float()
        self.std = torch.clamp(self.std, min=1e-3)

    def __call__(self, x):
        # x: [D] or [B, D]
        return (x - self.mean) / self.std


# ---------- Model builder that understands both checkpoint formats ----------
def _parse_arch(ckpt):
    """
    Returns ('resnet18' or 'resnet34', 'resnet') for new checkpoints,
    ('pca_mlp', 'pca') for PCA MLP checkpoints,
    or ('legacy', 'cnn') for old ones without arch info.
    """
    arch = ckpt.get('arch', '')
    if arch == 'PCAPolicy_MLP' or arch.startswith('PCAPolicy'):
        return 'pca_mlp', 'pca'
    if arch.startswith('ResNetPolicy_'):
        parts = arch.split('_')
        if len(parts) >= 2 and parts[1] in ('resnet18', 'resnet34'):
            return parts[1], 'resnet'
    return 'legacy', 'cnn'


def _build_model_from_ckpt(ckpt, device):
    n_actions = int(ckpt.get('n_actions', 16))
    arch_name, kind = _parse_arch(ckpt)
    if kind == 'pca':
        feature_dim = int(ckpt.get('feature_dim', 650))
        model = PCAPolicy(z_dim=feature_dim, n_actions=n_actions).to(device)
    elif kind == 'resnet':
        model = ResNetPolicy(n_actions=n_actions, backbone=arch_name).to(device)
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
    Backward-compatible with legacy CNN checkpoints, ResNet checkpoints, and PCA MLP checkpoints.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else
                                    ('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'))
    ckpt = torch.load(ckpt_path, map_location=device)

    model, n_actions = _build_model_from_ckpt(ckpt, device)
    arch_name, kind = _parse_arch(ckpt)
    
    if kind == 'pca':
        # PCA MLP checkpoint: use PCAFeatureNormalizer
        mean = ckpt.get('mean')
        std = ckpt.get('std')
        if mean is None or std is None:
            raise ValueError(f"PCA checkpoint {ckpt_path} missing 'mean' or 'std' keys")
        normalizer = PCAFeatureNormalizer(mean, std)
    else:
        # Image-based checkpoint: use ImageNormalizer
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


@torch.no_grad()
def act_greedy_pca(model, normalizer, device, pca_features):
    """
    Greedy action selection for PCA-based models.
    pca_features: numpy array of shape [D] or [1, D] (PCA feature vector)
    """
    if isinstance(pca_features, np.ndarray):
        x = torch.from_numpy(pca_features).float()
    else:
        x = pca_features.float()
    
    # Ensure shape is [1, D]
    if x.ndim == 1:
        x = x.unsqueeze(0)
    
    # Normalize
    x = normalizer(x)
    x = x.to(device)
    
    # Forward pass
    logits = model(x)
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    action = int(torch.argmax(probs).item())
    return action, probs.detach().cpu().numpy()


def bc_policy(models, state, skill):
    """
    Behavioral cloning policy. Works with both image-based and PCA-based models.
    For PCA models, state should be PCA features (1D array).
    For image models, state should be HxWx3 image.
    """
    entry = models["bc_models"][skill]
    model, normalizer, device, n_actions = entry
    
    # Check if this is a PCA model by checking if normalizer is PCAFeatureNormalizer
    is_pca = isinstance(normalizer, PCAFeatureNormalizer)
    
    if is_pca:
        # State is already PCA features (from environment)
        # Ensure it's a numpy array
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        state = np.asarray(state, dtype=np.float32)
        
        # If state is 2D (HxWx3 image), convert to PCA features
        if state.ndim == 3:
            # Convert image to PCA features
            X = state.reshape(1, -1).astype(np.float32)
            if state.max() > 1.0:
                X = X / 255.0
            X_centered = models["pca_model"]['scaler'].transform(X)
            X_pca = models["pca_model"]['pca'].transform(X_centered)
            state = X_pca[0]
        elif state.ndim == 2 and state.shape[0] == 1:
            state = state[0]
        
        action, _ = act_greedy_pca(model, normalizer, device, state)
    else:
        # Image-based model: expect HxWx3 image
        assert state.max() > 1.0  # allow uint8 input
        state = np.asarray(state).astype(np.float32) / 255.0
        action, _ = act_greedy(model, normalizer, device, state)
    
    return action


# ---------- PU start / end utilities ----------
def load_pu_start_models(models_dir: str):
    models = []
    for fname in os.listdir(models_dir):
        if not fname.endswith("_meta.json"):
            continue
        skill = fname[:-10]  # strip "_meta.json"
        meta_path  = os.path.join(models_dir, f"{skill}_meta.json")
        model_path = os.path.join(models_dir, f"{skill}_clf.joblib")
        if not os.path.exists(model_path):
            continue
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            thr = float(meta["threshold"])
            clf = joblib_load(model_path)
            models.append({"skill": skill, "clf": clf, "thr": thr, "meta": meta})
        except Exception as e:
            print(f"[WARN] Skipping {skill}: {e}")
    return models

def applicable_pu_start_models(models, state, *, return_details=False, eps=0.0):
    state = np.asarray(state)
    if state.ndim == 1:
        X = state.reshape(1, -1)
    elif state.ndim == 2 and state.shape[0] == 1:
        X = state
    else:
        raise ValueError("`state` must be a single feature vector of shape [d] or [1,d].")

    rows = []
    for m in models:
        prob = float(m["clf"].predict_proba(X)[:, 1][0])
        thr  = float(m["thr"])
        margin = prob - thr
        is_applicable = prob >= (thr + eps)
        rows.append({
            "skill": m["skill"],
            "prob": prob,
            "thr": thr,
            "margin": margin,
            "applicable": is_applicable
        })

    rows.sort(key=lambda r: r["margin"], reverse=True)
    applicable = [r for r in rows if r["applicable"]]
    return rows if return_details else [r["skill"] for r in applicable]

def load_pu_end_model(models_dir: str, skill: str):
    model_path = os.path.join(models_dir, f"{skill}_clf.joblib")
    meta_path  = os.path.join(models_dir, f"{skill}_meta.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta file:  {meta_path}")

    clf = joblib_load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    thr = float(meta["threshold"])
    return {"skill": skill, "clf": clf, "thr": thr, "meta": meta}

def predict_pu_end_state(model: dict, state) -> dict:
    state = np.asarray(state)
    if state.ndim == 1:
        X = state.reshape(1, -1)
    elif state.ndim == 2 and state.shape[0] == 1:
        X = state
    else:
        raise ValueError("`state` must be a single feature vector of shape [d] or [1, d].")

    prob = float(model["clf"].predict_proba(X)[:, 1][0])
    thr = float(model["thr"])
    margin = prob - thr
    return {
        "prob": prob,
        "threshold": thr,
        "is_end": bool(prob >= thr),
        "margin": margin,
    }


# ---------- PCA + model bank loaders ----------
def load_all_models(skill_list = ['wood', 'stone', 'wood_pickaxe', 'stone_pickaxe', 'table'],
                    root: str = 'Traces/stone_pickaxe_easy',
                    bc_checkpoint_dir: str = 'bc_checkpoints_pca',
                    pca_model_path: str = 'pca_models/pca_model_750.joblib',
                    pu_start_models_dir: str = 'pu_start_models',
                    pu_end_models_dir: str = 'pu_end_models'):
    bc_models = {}
    for skill in skill_list:
        # Try PCA MLP checkpoint first
        ckpt_path = os.path.join(root, bc_checkpoint_dir, f'{skill}_pca_mlp_policy.pt')
        if not os.path.exists(ckpt_path):
            # Fallback to old naming convention
            ckpt_path = os.path.join(root, bc_checkpoint_dir, f'{skill}_policy_resnet34_pt.pt')
        bc_models[skill] = load_policy(ckpt_path)

    artifacts = joblib.load(os.path.join(root, pca_model_path))
    scaler = artifacts['scaler']
    pca = artifacts['pca']
    n_features_expected = scaler.mean_.shape[0]

    pu_start_models = load_pu_start_models(os.path.join(root, pu_start_models_dir))

    pu_end_models = {}
    for skill in skill_list:
        try:
            pu_end_models[skill] = load_pu_end_model(os.path.join(root, pu_end_models_dir), skill)
        except FileNotFoundError:
            print(f"[WARN] No PU end model for skill '{skill}'")

    return {
        "skills": skill_list,  # canonical order
        "bc_models": bc_models,
        "termination_models": pu_end_models,
        "start_models": pu_start_models,
        "pca_model": {'scaler': scaler, 'pca': pca, 'n_features_expected': n_features_expected}
    }


def available_skills(models, state):
    """
    Determine which skills are available given the current state.
    state: can be PCA features (1D array) or HxWx3 image
    """
    state = np.asarray(state).astype(np.float32)
    
    # If state is already PCA features (1D), use directly
    if state.ndim == 1:
        Xf = state.reshape(1, -1)
    elif state.ndim == 3:
        # Image input: convert to PCA features
        if state.max() > 1.0:
            state = state / 255.0
        X = state.reshape(1, -1)
        Xc = models["pca_model"]['scaler'].transform(X)
        Xf = models["pca_model"]['pca'].transform(Xc)
    else:
        # Already in correct shape (1, D) or (B, D)
        Xf = state if state.ndim == 2 else state.reshape(1, -1)

    rows = applicable_pu_start_models(models["start_models"], Xf, return_details=True, eps=0.0)
    applicable = {r["skill"] for r in rows if r["applicable"]}
    order = models["skills"]
    return np.array([s in applicable for s in order], dtype=bool)


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
    Phase-aware termination.
    - Primitive skill: use its end-model directly.
    - Composite skill: only allow termination when we're on the *final* leaf, and its end fires,
      or when we've already advanced past the last leaf (idx >= len(seq)).
    """
    # Composite skill special-case
    seq = _get_seq_for_skill(models, skill)
    if seq is not None:
        # We are in a production/composite
        rt = _rt_for(models, skill, call_id)
        idx = int(rt.get("idx", 0))
        # Finished all leaves already?
        if idx >= len(seq):
            return True
        # Only last leaf can end the composite
        last_leaf = seq[-1]
        # If we're not yet on the last leaf, composite must not end
        if idx < len(seq) - 1:
            return False
        # We *are* on the last leaf: gate by its end detector
        return _primitive_should_terminate(models, state, last_leaf)

    # Primitive skill
    return _primitive_should_terminate(models, state, skill)


def _primitive_should_terminate(models, state, skill) -> bool:
    """
    Check if a primitive skill should terminate.
    state: can be PCA features (1D array) or HxWx3 image
    """
    state = np.asarray(state).astype(np.float32)
    
    # If state is already PCA features (1D), use directly
    if state.ndim == 1:
        X_feats = state.reshape(1, -1)
    elif state.ndim == 3:
        # Image input: convert to PCA features
        if state.max() > 1.0:
            state = state / 255.0
        X = state.reshape(1, -1)
        X_centered = models["pca_model"]['scaler'].transform(X)
        X_feats = models["pca_model"]['pca'].transform(X_centered)
    else:
        # Already in correct shape
        X_feats = state if state.ndim == 2 else state.reshape(1, -1)
    
    tm = models["termination_models"].get(skill)
    if tm is None:
        # No end model: never terminate based on classifier
        return False
    return predict_pu_end_state(tm, X_feats)["is_end"]


# option_helpers.py
def bc_policy_hierarchy(models, state, skill, call_id: Optional[int] = None, max_leaf_len: Optional[int] = None):
    entry = models["bc_models"][skill]

    if isinstance(entry, tuple) and entry and entry[0] == "__COMPOSITE__":
        seq: List[str] = entry[1]
        rt = _rt_for(models, skill, call_id)
        idx = int(rt.get("idx", 0))
        leaf_steps = int(rt.get("leaf_steps", 0))

        # If detector says current leaf is done, advance and reset per-leaf counter
        if idx < len(seq):
            cur = seq[idx]
            if _primitive_should_terminate(models, state, cur):
                idx += 1
                rt["idx"] = idx
                rt["leaf_steps"] = 0
                leaf_steps = 0

        # If a per-leaf cap is set and we've spent it, force advance
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

    # primitive case: use bc_policy which handles both PCA and image models
    return bc_policy(models, state, skill)


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
      - start model := start PU of first leaf skill
      - end model   := end  PU of last  leaf skill (kept for availability gating;
                       composite termination is phase-aware in should_terminate)
      - bc model    := sentinel ('__COMPOSITE__', [child_skill_names...])
    """
    start_by_skill = {m["skill"]: m for m in models["start_models"]}
    end_by_skill   = models["termination_models"]
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

            first_leaf, last_leaf = seq[0], seq[-1]

            if first_leaf not in start_by_skill:
                print(f"[WARN] No start PU for '{first_leaf}' (needed by Production_{pid}); skipping.")
                continue
            if last_leaf not in end_by_skill:
                print(f"[WARN] No end PU for '{last_leaf}' (needed by Production_{pid}); skipping.")
                continue

            name = f"Production_{pid}"

            if name in models["bc_models"] and name in models["termination_models"]:
                continue

            # 1) BC model sentinel (sequence of leaves)
            models["bc_models"][name] = ("__COMPOSITE__", seq)

            # 2) Keep alias of last leaf's end model (used for availability gating & legacy)
            models["termination_models"][name] = end_by_skill[last_leaf]

            # 3) Start: alias of first leaf's start PU (for availability)
            first_leaf_start = start_by_skill[first_leaf]
            models["start_models"].append({
                "skill": name,
                "clf": first_leaf_start["clf"],
                "thr": first_leaf_start["thr"],
                "meta": dict(first_leaf_start.get("meta", {})),
            })

            # 4) Expose in skills list
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
    bc_checkpoint_dir: str = 'bc_checkpoints_pca',
    pca_model_path: str = 'pca_models/pca_model_750.joblib',
    pu_start_models_dir: str = 'pu_start_models',
    pu_end_models_dir: str = 'pu_end_models',
):
    ckpt_dir = os.path.join(root, bc_checkpoint_dir)
    bc_models = {}

    def _find_ckpt(skill: str) -> str:
        # Try PCA MLP checkpoint first
        cand = os.path.join(ckpt_dir, f'{skill}_pca_mlp_policy.pt')
        if os.path.exists(cand):
            return cand
        # Fallback to old naming conventions
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
            f"Tried '{os.path.join(ckpt_dir, f'{skill}_pca_mlp_policy.pt')}', '{cand}', any resnet*, and legacy '{legacy}'."
        )

    for skill in skill_list:
        ckpt_path = _find_ckpt(skill)
        bc_models[skill] = load_policy(ckpt_path)

    artifacts = joblib.load(os.path.join(root, pca_model_path))
    scaler = artifacts['scaler']
    pca = artifacts['pca']
    n_features_expected = scaler.mean_.shape[0]

    pu_start_models = load_pu_start_models(os.path.join(root, pu_start_models_dir))

    pu_end_models = {}
    for skill in skill_list:
        try:
            pu_end_models[skill] = load_pu_end_model(os.path.join(root, pu_end_models_dir), skill)
        except FileNotFoundError:
            print(f"[WARN] No PU end model for skill '{skill}'")

    models = {
        "skills": list(skill_list),
        "bc_models": bc_models,                 # {skill: (model,..) or ("__COMPOSITE__", seq)}
        "termination_models": pu_end_models,    # end detectors (primitive; composites get alias of last leaf)
        "start_models": pu_start_models,        # start detectors
        "pca_model": {'scaler': scaler, 'pca': pca, 'n_features_expected': n_features_expected},
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
