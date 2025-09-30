import torch
import torch.nn.functional as F
import numpy as np
import os 
import joblib 
import json 
from joblib import load as joblib_load
import gymnasium as gym
# import torchvision
import glob

# Optional: only import torchvision if we actually need a ResNet
_RESNET_IMPORTED = False

# --- must match your training definitions ---
# class ImageNormalizer:
#     def __init__(self, mean, std):
#         self.mean = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
#         self.std  = torch.clamp(torch.tensor(std, dtype=torch.float32).view(3,1,1), min=1e-3)
#     def __call__(self, x):  # x: [3,H,W] in [0,1]
#         return (x - self.mean) / self.std

# class ConvBlock(torch.nn.Module):
#     def __init__(self, c_in, c_out, k=3, s=1, p=1):
#         super().__init__()
#         self.conv = torch.nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)
#         self.bn   = torch.nn.BatchNorm2d(c_out)  # or GroupNorm if you switched
#         self.act  = torch.nn.GELU()
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

# class PolicyCNN(torch.nn.Module):
#     def __init__(self, n_actions=16):
#         super().__init__()
#         self.stem = torch.nn.Sequential(
#             ConvBlock(3, 32, k=7, s=2, p=3),
#             ConvBlock(32, 32),
#             torch.nn.MaxPool2d(2),
#         )
#         self.stage2 = torch.nn.Sequential(
#             ConvBlock(32, 64),
#             ConvBlock(64, 64),
#             torch.nn.MaxPool2d(2),
#         )
#         self.stage3 = torch.nn.Sequential(
#             ConvBlock(64, 128),
#             ConvBlock(128, 128),
#             torch.nn.MaxPool2d(2),
#         )
#         self.stage4 = torch.nn.Sequential(
#             ConvBlock(128, 256),
#             ConvBlock(256, 256),
#         )
#         self.head = torch.nn.Linear(256, n_actions)

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.stage4(x)
#         x = F.adaptive_avg_pool2d(x, 1)
#         x = torch.flatten(x, 1)
#         return self.head(x)

# # ---- inference helpers ----

# def load_policy(ckpt_path, device=None):
#     """Load model + normalizer from a saved training checkpoint."""
#     device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     ckpt = torch.load(ckpt_path, map_location=device)
#     n_actions = int(ckpt['n_actions'])
#     model = PolicyCNN(n_actions=n_actions).to(device)
#     model.load_state_dict(ckpt['state_dict'])
#     model.eval()
#     normalizer = ImageNormalizer(ckpt['mean'], ckpt['std'])
#     return model, normalizer, device, n_actions

# def preprocess_frame(frame_hw3, normalizer, target=256):
#     """
#     frame_hw3: numpy array [H,W,3], float32 in [0,1]
#     returns torch tensor [1,3,target,target]
#     """
#     assert frame_hw3.ndim == 3 and frame_hw3.shape[2] == 3
#     x = torch.from_numpy(np.transpose(frame_hw3, (2,0,1))).float()   # [3,H,W]
#     x = F.interpolate(x.unsqueeze(0), size=(target, target), mode='bilinear', align_corners=False).squeeze(0)  # [3,T,T]
#     x = normalizer(x)
#     return x.unsqueeze(0)  # [1,3,T,T]

# @torch.no_grad()
# def act_greedy(model, normalizer, device, frame_hw3):
#     """
#     Returns (action_id, probs) where probs is a numpy array length n_actions.
#     """
#     x = preprocess_frame(frame_hw3, normalizer)            # [1,3,256,256]
#     x = x.to(device)
#     logits = model(x)                                      # [1,n_actions]
#     probs = torch.softmax(logits, dim=-1).squeeze(0)       # [n_actions]
#     action = int(torch.argmax(probs).item())
#     return action, probs.cpu().numpy()

# @torch.no_grad()
# def act_sample(model, normalizer, device, frame_hw3, temperature=1.0):
#     x = preprocess_frame(frame_hw3, normalizer).to(device)
#     logits = model(x).squeeze(0)
#     if temperature != 1.0:
#         logits = logits / max(1e-6, float(temperature))
#     probs = torch.softmax(logits, dim=-1)
#     action = int(torch.multinomial(probs, num_samples=1).item())
#     return action, probs.cpu().numpy()


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
    (Weights are loaded from checkpoint; no need to pull pretrained weights here.)
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
            raise ValueError("Unknown backbone: {backbone}")

        in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(in_features, n_actions)

    def forward(self, x):
        return self.backbone(x)


# ---------- Model builder that understands both checkpoint formats ----------
def _parse_arch(ckpt):
    """
    Returns ('resnet18' or 'resnet34', 'resnet') for new checkpoints,
    or ('legacy', 'cnn') for old ones without arch info.
    """
    arch = ckpt.get('arch', '')
    if arch.startswith('ResNetPolicy_'):
        # e.g., 'ResNetPolicy_resnet18_pretrained'
        parts = arch.split('_')
        # expected: ['ResNetPolicy', 'resnet18', 'pretrained']
        if len(parts) >= 2 and parts[1] in ('resnet18', 'resnet34'):
            return parts[1], 'resnet'
    # Legacy fallback
    return 'legacy', 'cnn'


def _build_model_from_ckpt(ckpt, device):
    n_actions = int(ckpt.get('n_actions', 16))
    backbone, kind = _parse_arch(ckpt)
    if kind == 'resnet':
        model = ResNetPolicy(n_actions=n_actions, backbone=backbone).to(device)
    else:
        # legacy small CNN
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


# ---------- Public API (unchanged names/signatures) ----------
def load_policy(ckpt_path, device=None):
    """
    Load model + normalizer from a saved training checkpoint.
    Backward-compatible with legacy CNN checkpoints and new ResNet checkpoints.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else
                                    ('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'))
    ckpt = torch.load(ckpt_path, map_location=device)

    # Build the right model and read n_actions
    model, n_actions = _build_model_from_ckpt(ckpt, device)

    # Normalization: always read mean/std from checkpoint (both formats store them)
    mean = ckpt.get('mean', (0.485, 0.456, 0.406))
    std  = ckpt.get('std',  (0.229, 0.224, 0.225))
    normalizer = ImageNormalizer(mean, std)

    return model, normalizer, device, n_actions


@torch.no_grad()
def act_greedy(model, normalizer, device, frame_hw3, target=256):
    """
    Returns (action_id, probs) where probs is a numpy array length n_actions.
    """
    x = preprocess_frame(frame_hw3, normalizer, target=target).to(device)   # [1,3,256,256]
    logits = model(x)                                      # [1,n_actions]
    probs = torch.softmax(logits, dim=-1).squeeze(0)       # [n_actions]
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

    assert state.max() > 1.0 
     # allow uint8 input

    state = np.asarray(state).astype(np.float32) / 255.0

    model, normalizer, device, n_actions = models["bc_models"][skill]
    action, probs = act_greedy(model, normalizer, device, state)
    return action

def load_pu_start_models(models_dir: str):
    """
    Load (skill, clf, threshold, meta) tuples from <models_dir>.
    Expects files: <skill>_clf.joblib and <skill>_meta.json
    Returns: list[dict] with keys: skill, clf, thr, meta
    """
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
    """
    Given a list from load_pu_models(...) and a state feature vector (shape [d] or [1,d]),
    return/print skills whose probability >= threshold (+eps).
    - return_details=True returns a list of dicts with scores/margins
    - eps lets you demand a small margin above threshold (e.g., eps=0.02).
    """
    # Accept 1D or 2D input
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

    # Sort by confidence margin (best first)
    rows.sort(key=lambda r: r["margin"], reverse=True)

    # Print list of applicable models
    applicable = [r for r in rows if r["applicable"]]
    # if applicable:
    #     print("Applicable models (prob ≥ threshold):")
    #     for r in applicable:
    #         print(f"  - {r['skill']}: p={r['prob']:.3f}  thr={r['thr']:.3f}  margin={r['margin']:.3f}")
    # else:
    #     print("No applicable models for this state.")

    return rows if return_details else [r["skill"] for r in applicable]

def load_pu_end_model(models_dir: str, skill: str):
    """
    Load a single PU model (classifier + metadata) for a given skill.

    Looks for:
      - <models_dir>/<skill>_clf.joblib
      - <models_dir>/<skill>_meta.json

    Returns:
      dict with keys:
        - skill: str
        - clf:   fitted classifier (expects .predict_proba)
        - thr:   float threshold from meta["threshold"]
        - meta:  dict (entire meta JSON)
    """
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
    """
    Score a single state with a loaded PU model dict from load_pu_model(...).

    Args:
      - model: dict with keys {"skill","clf","thr","meta"}
      - state: shape [d] or [1, d]

    Returns:
      dict: {prob, threshold, is_end, margin}
    """
    # Accept 1D or 2D single-row input
    state = np.asarray(state)
    if state.ndim == 1:
        X = state.reshape(1, -1)
    elif state.ndim == 2 and state.shape[0] == 1:
        X = state
    else:
        raise ValueError("`state` must be a single feature vector of shape [d] or [1, d].")

    # Compute positive-class probability
    prob = float(model["clf"].predict_proba(X)[:, 1][0])

    thr = float(model["thr"])
    margin = prob - thr
    return {
        "prob": prob,
        "threshold": thr,
        "is_end": bool(prob >= thr),
        "margin": margin,
    }

def load_all_models(skill_list = ['wood', 'stone', 'wood_pickaxe', 'stone_pickaxe', 'table']):
    bc_models = {}
    for skill in skill_list:
        ckpt_path = os.path.join('Traces/stone_pickaxe_easy', 'bc_checkpoints_resnet', f'{skill}_policy_resnet18_pt.pt')
        bc_models[skill] = load_policy(ckpt_path)

    artifacts = joblib.load('Traces/stone_pickaxe_easy/pca_models/pca_model_750.joblib')
    scaler = artifacts['scaler']
    pca = artifacts['pca']
    n_features_expected = scaler.mean_.shape[0]

    pu_start_models = load_pu_start_models('Traces/stone_pickaxe_easy/pu_start_models')

    pu_end_models = {}
    for skill in skill_list:
        try:
            pu_end_models[skill] = load_pu_end_model('Traces/stone_pickaxe_easy/pu_end_models', skill)
        except FileNotFoundError:
            print(f"[WARN] No PU end model for skill '{skill}'")

    return {
        "skills": skill_list,  # <—— canonical order
        "bc_models": bc_models,
        "termination_models": pu_end_models,
        "start_models": pu_start_models,
        "pca_model": {'scaler': scaler, 'pca': pca, 'n_features_expected': n_features_expected}
    }

def available_skills(models, state):
    # state: uint8 or float32 flat vector -> PCA space
    state = np.asarray(state).astype(np.float32)
    if state.max() > 1.0:  # allow uint8 input
        state = state / 255.0

    X = state.reshape(1, -1)
    Xc = models["pca_model"]['scaler'].transform(X)
    Xf = models["pca_model"]['pca'].transform(Xc)

    rows = applicable_pu_start_models(models["start_models"], Xf, return_details=True, eps=0.0)
    applicable = {r["skill"] for r in rows if r["applicable"]}
    order = models["skills"]
    return np.array([s in applicable for s in order], dtype=bool)

def should_terminate(models, state, skill): 
    state = np.asarray(state).astype(np.float32) / 255.0

    X = state.reshape(1, -1)
    X_centered = models["pca_model"]['scaler'].transform(X)
    X_feats = models["pca_model"]['pca'].transform(X_centered)

    return predict_pu_end_state(models["termination_models"][skill], X_feats)["is_end"]






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

"""
Hierarchy utilities
"""


import hashlib
from typing import Any, Dict, List, Optional

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



# ===== NEW: compile productions into existing banks =====

def compile_productions_into_skill_bank(
    models: Dict[str, Any],
    hierarchies: List[Dict[str, Any]],
    symbol_map: Dict[str, str],
) -> None:
    """
    For each production node P:
      - Create a pseudo-skill 'Production_<P>'
      - start model := start PU of first leaf skill
      - end model   := end  PU of last  leaf skill
      - bc model    := sentinel ('__COMPOSITE__', [child_skill_names...])
    Mutates `models` in-place:
      models['skills'] += [composite_names]
      models['bc_models'][name] = ('__COMPOSITE__', sequence)
      models['termination_models'][name] = termination_models[last_leaf]
      models['start_models'] += [{'skill': name, 'clf': ..., 'thr': ..., 'meta': ...}]  # alias of first leaf
    """
    # Reindex helpers
    start_by_skill = {m["skill"]: m for m in models["start_models"]}
    end_by_skill   = models["termination_models"]
    skills_set     = set(models["skills"])

    # Ensure composite runtime dict exists
    if "composite_runtime" not in models:
        models["composite_runtime"] = {}

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

            # Avoid duplicates if multiple hierarchies contain the same production
            if name in models["bc_models"] and name in models["termination_models"]:
                continue

            # 1) BC model sentinel
            models["bc_models"][name] = ("__COMPOSITE__", seq)

            # 2) Termination = alias of last leaf's end model
            models["termination_models"][name] = end_by_skill[last_leaf]

            # 3) Start = add an alias entry using first leaf's start PU
            #    NOTE: available_skills() already uses models["start_models"], so we just append.
            first_leaf_start = start_by_skill[first_leaf]
            models["start_models"].append({
                "skill": name,
                "clf": first_leaf_start["clf"],
                "thr": first_leaf_start["thr"],
                "meta": dict(first_leaf_start.get("meta", {})),
            })

            # 4) Expose in skills list (after primitives)
            if name not in skills_set:
                models["skills"].append(name)
                skills_set.add(name)
                composite_names.append(name)

    if composite_names:
        # deterministic order optional
        # leave as appended to preserve discovery order
        print(f"[INFO] Added composites: {', '.join(composite_names)}")


# ===== EDIT: bc_policy to handle composite sentinels =====

# def bc_policy_hierarchy(models, state, skill):
#     entry = models["bc_models"][skill]

#     # Composite sentinel?
#     if isinstance(entry, tuple) and entry and entry[0] == "__COMPOSITE__":
#         seq = entry[1]  # list of leaf skill names in order
#         name = skill

#         # progress tracker
#         rt = models.setdefault("composite_runtime", {}).setdefault(name, {"idx": 0})
#         idx = int(rt["idx"])

#         # Guard: if already complete, hold last leaf's action (or return a STOP if you have one)
#         if idx >= len(seq):
#             last = seq[-1]
#             return bc_policy(models, state, last)

#         # If current sub-skill has terminated, advance
#         cur = seq[idx]
#         if should_terminate(models, state, cur):
#             idx += 1
#             rt["idx"] = idx
#             if idx >= len(seq):
#                 # Finished; hold last skill action
#                 return bc_policy(models, state, seq[-1])
#             cur = seq[idx]

#         # Act with current sub-skill
#         return bc_policy(models, state, cur)

#     # ---- Primitive case (unchanged) ----
#     assert state.max() > 1.0  # allow uint8 input
#     state = np.asarray(state).astype(np.float32) / 255.0
#     model, normalizer, device, n_actions = entry
#     action, probs = act_greedy(model, normalizer, device, state)
#     return action


# def load_all_models_hierarchy(
#     skill_list = ['wood', 'stone', 'wood_pickaxe', 'stone_pickaxe', 'table'],
#     hierarchies_dir: Optional[str] = None,
#     symbol_map: Optional[Dict[str, str]] = None,
# ):
#     bc_models = {}
#     for skill in skill_list:
#         ckpt_path = os.path.join('Traces/stone_pickaxe_easy', 'bc_checkpoints', f'{skill}_policy_cnn.pt')
#         bc_models[skill] = load_policy(ckpt_path)

#     artifacts = joblib.load('Traces/stone_pickaxe_easy/pca_models/pca_model_750.joblib')
#     scaler = artifacts['scaler']
#     pca = artifacts['pca']
#     n_features_expected = scaler.mean_.shape[0]

#     pu_start_models = load_pu_start_models('Traces/stone_pickaxe_easy/pu_start_models')

#     pu_end_models = {}
#     for skill in skill_list:
#         try:
#             pu_end_models[skill] = load_pu_end_model('Traces/stone_pickaxe_easy/pu_end_models', skill)
#         except FileNotFoundError:
#             print(f"[WARN] No PU end model for skill '{skill}'")


#     models = {
#         "skills": list(skill_list),
#         "bc_models": bc_models,
#         "termination_models": pu_end_models,
#         "start_models": pu_start_models,
#         "pca_model": {'scaler': scaler, 'pca': pca, 'n_features_expected': n_features_expected}
#     }

#     # === NEW: compile unique hierarchies into this bank ===
#     if hierarchies_dir and symbol_map:
#         uniq = load_unique_hierarchies(hierarchies_dir)
#         compile_productions_into_skill_bank(models, uniq, symbol_map)

#     return models


def bc_policy_hierarchy(models, state, skill):
    """
    Hierarchical controller that is compatible with both legacy CNN and new ResNet policies.
    - Accepts uint8 [H,W,3] or float32 in [0,1]; scaling happens inside act_* via preprocess_frame.
    - Composite entries are marked as ("__COMPOSITE__", [leaf_skill_1, leaf_skill_2, ...]).
    """
    entry = models["bc_models"][skill]

    # ---- Composite case ----
    if isinstance(entry, tuple) and entry and isinstance(entry[0], str) and entry[0] == "__COMPOSITE__":
        seq: List[str] = entry[1]  # ordered list of leaf skill names
        name = skill

        # progress tracker
        rt = models.setdefault("composite_runtime", {}).setdefault(name, {"idx": 0})
        idx = int(rt["idx"])

        # If already complete, keep executing the last leaf (or swap for STOP if you have one)
        if idx >= len(seq):
            last = seq[-1]
            return bc_policy(models, state, last)

        # If the current sub-skill has terminated, advance
        cur = seq[idx]
        if should_terminate(models, state, cur):
            idx += 1
            rt["idx"] = idx
            if idx >= len(seq):
                # Finished; hold last skill action
                return bc_policy(models, state, seq[-1])
            cur = seq[idx]

        # Act with the current sub-skill
        return bc_policy(models, state, cur)

    # ---- Primitive case ----
    # Don't rescale to [0,1] here; act_greedy/preprocess_frame already supports uint8 or float.
    state = np.asarray(state)
    assert state.ndim == 3 and state.shape[-1] == 3, "state must be HxWx3"
    model, normalizer, device, n_actions = entry
    action, _ = act_greedy(model, normalizer, device, state)
    return action


def load_all_models_hierarchy(
    skill_list = ['wood', 'stone', 'wood_pickaxe', 'stone_pickaxe', 'table'],
    hierarchies_dir: Optional[str] = None,
    symbol_map:     Optional[Dict[str, str]] = None,
    root:           str = 'Traces/stone_pickaxe_easy',
    backbone_hint:  str = 'resnet18',  # try this first; will auto-fallback
):

    ckpt_dir = os.path.join(root, 'bc_checkpoints_resnet')
    bc_models = {}

    def _find_ckpt(skill: str) -> str:
        # 1) direct hint
        cand = os.path.join(ckpt_dir, f'{skill}_policy_{backbone_hint}_pt.pt')
        if os.path.exists(cand):
            return cand
        # 2) any resnet file for the skill
        hits = sorted(glob.glob(os.path.join(ckpt_dir, f'{skill}_policy_resnet*_pt.pt')))
        if len(hits) > 0:
            return hits[0]
        # 3) legacy cnn filename
        legacy = os.path.join(ckpt_dir, f'{skill}_policy_cnn.pt')
        if os.path.exists(legacy):
            return legacy
        raise FileNotFoundError(
            f"No checkpoint found for skill '{skill}'. "
            f"Tried '{cand}', any resnet*, and legacy '{legacy}'."
        )

    # Load BC policies (model, normalizer, device, n_actions)
    for skill in skill_list:
        ckpt_path = _find_ckpt(skill)
        bc_models[skill] = load_policy(ckpt_path)

    # PCA artifacts (unchanged)
    artifacts = joblib.load(os.path.join(root, 'pca_models', 'pca_model_750.joblib'))
    scaler = artifacts['scaler']
    pca = artifacts['pca']
    n_features_expected = scaler.mean_.shape[0]

    # PU start models (unchanged)
    pu_start_models = load_pu_start_models(os.path.join(root, 'pu_start_models'))

    # PU end models; some may be missing
    pu_end_models = {}
    for skill in skill_list:
        try:
            pu_end_models[skill] = load_pu_end_model(os.path.join(root, 'pu_end_models'), skill)
        except FileNotFoundError:
            print(f"[WARN] No PU end model for skill '{skill}'")

    models = {
        "skills": list(skill_list),
        "bc_models": bc_models,                 # {skill: (model, normalizer, device, n_actions)} or ("__COMPOSITE__", [..])
        "termination_models": pu_end_models,    # end detectors
        "start_models": pu_start_models,        # start detectors
        "pca_model": {'scaler': scaler, 'pca': pca, 'n_features_expected': n_features_expected}
    }

    # Compile hierarchies if provided
    if hierarchies_dir and symbol_map:
        uniq = load_unique_hierarchies(hierarchies_dir)
        compile_productions_into_skill_bank(models, uniq, symbol_map)

    return models

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