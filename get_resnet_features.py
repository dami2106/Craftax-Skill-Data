# save as: Skill_Learning/extract_resnet_features_dataset.py

import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


# ---------- BC-parity helpers ----------
class ImageNormalizer:
    """Normalize CHW in [0,1] using per-channel mean/std (float32)."""
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.clamp(torch.tensor(std, dtype=torch.float32).view(3, 1, 1), min=1e-3)

    def __call__(self, x):
        return (x - self.mean) / self.std


def _to_chw(img_nhwc: np.ndarray) -> torch.Tensor:
    # NHWC -> CHW float32
    return torch.from_numpy(np.transpose(img_nhwc, (2, 0, 1))).float()


def _resize_chw(x_chw: torch.Tensor, target=256) -> torch.Tensor:
    x = x_chw.unsqueeze(0)  # [1,C,H,W]
    x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
    return x.squeeze(0)


class ResNetFeatureExtractor(nn.Module):
    """Return 512-dim features after global avgpool (before FC)."""
    def __init__(self, backbone="resnet34", pretrained=True):
        super().__init__()
        if backbone == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = resnet18(weights=weights)
        elif backbone == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            base = resnet34(weights=weights)
        else:
            raise ValueError("backbone must be 'resnet18' or 'resnet34'")
        self.stem = nn.Sequential(*list(base.children())[:-1])  # conv..avgpool

    def forward(self, x):
        with torch.no_grad():
            f = self.stem(x)           # [B, 512, 1, 1]
            f = f.view(f.size(0), -1)  # [B, 512]
        return f


def load_bc_settings_from_ckpt(ckpt_path: Path):
    """Read backbone + normalization choice from a BC checkpoint you saved."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    imagenet_norm = bool(ckpt.get("imagenet_norm", True))
    mean = tuple(ckpt.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(ckpt.get("std", (0.229, 0.224, 0.225)))
    arch = str(ckpt.get("arch", ""))  # e.g., 'ResNetPolicy_resnet34_pretrained'

    backbone = "resnet34"
    if "resnet18" in arch.lower():
        backbone = "resnet18"
    elif "resnet34" in arch.lower():
        backbone = "resnet34"

    if imagenet_norm:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    print(f"Loaded BC checkpoint from {ckpt_path}: backbone={backbone}, imagenet_norm={imagenet_norm}, mean={mean}, std={std}")

    return backbone, imagenet_norm, mean, std


def compute_dataset_mean_std(npy_files, sample_every=1):
    """
    Streaming per-channel mean/std over NHWC images in [0,1].
    sample_every>1 lets you subsample frames for speed.
    """
    count = 0
    sum_c = np.zeros(3, dtype=np.float64)
    sqsum_c = np.zeros(3, dtype=np.float64)

    for f in tqdm(npy_files, desc="Computing dataset mean/std"):
        imgs = np.asarray(np.load(f), dtype=np.float32)  # [N,H,W,3]
        imgs = imgs[::sample_every]
        n, h, w, _ = imgs.shape
        flat = imgs.reshape(-1, 3).astype(np.float64)
        sum_c += flat.sum(axis=0)
        sqsum_c += np.square(flat).sum(axis=0)
        count += flat.shape[0]

    mean = sum_c / count
    var = sqsum_c / count - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    return tuple(mean.tolist()), tuple(std.tolist())


def extract_features_for_array(images_nhwc: np.ndarray,
                               normalizer: ImageNormalizer,
                               model: nn.Module,
                               device: torch.device,
                               batch_size: int = 256,
                               dtype: torch.dtype = torch.float32) -> np.ndarray:
    """
    images_nhwc: [N,H,W,3] float32 in [0,1]
    returns np.float32 [N,512]
    """
    N = images_nhwc.shape[0]
    out = np.zeros((N, 512), dtype=np.float32)
    i = 0
    while i < N:
        j = min(i + batch_size, N)
        batch = []
        for k in range(i, j):
            x = _to_chw(images_nhwc[k])           # [3,H,W]
            x = _resize_chw(x, 256)               # [3,256,256]
            x = normalizer(x).to(dtype=dtype)     # normalization identical to BC
            batch.append(x)
        xb = torch.stack(batch, 0).to(device, non_blocking=True)
        fb = model(xb)                             # [B,512]
        out[i:j] = fb.detach().cpu().numpy()
        i = j
    return out


def main():
    ap = argparse.ArgumentParser(description="Extract ResNet features for entire dataset using BC settings.")
    ap.add_argument("--data_dir", type=str, default="Traces/stone_pick_static",
                    help="Path to the data directory (same root as BC)")
    ap.add_argument("--image_dir_name", type=str, default="top_down_obs",
                    help="Subdir with per-episode .npy images")
    ap.add_argument("--out_dir", type=str, default="resnet_features_all",
                    help="Output subdir under --data_dir")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--fp16", action="store_true", help="Use float16 on CUDA for extraction (saved as float32)")

    # How to choose backbone + normalization:
    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument("--ckpt_path", type=str, default="",
                       help="Optional BC checkpoint path to read backbone + normalization from")

    # Manual settings (used only if --ckpt_path is not provided)
    ap.add_argument("--backbone", type=str, default="resnet34", choices=["resnet18", "resnet34"],
                    help="Ignored if --ckpt_path is given")
    ap.add_argument("--imagenet_norm", action="store_true",
                    help="If set, use ImageNet mean/std; else use dataset or provided mean/std")
    ap.add_argument("--mean_std_npy", type=str, default="",
                    help="Optional path to .npy with dict {'mean':(3,), 'std':(3,)} when not using ImageNet")
    ap.add_argument("--compute_dataset_mean_std", action="store_true",
                    help="If not using ImageNet and no mean_std_npy, compute mean/std over the dataset")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    img_dir = data_dir / args.image_dir_name
    out_dir = data_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    npy_files = sorted(img_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {img_dir}")

    # Resolve backbone + normalization
    if args.ckpt_path:
        backbone, imagenet_norm, mean, std = load_bc_settings_from_ckpt(Path(args.ckpt_path))
        print(f"[settings] from checkpoint: backbone={backbone}, imagenet_norm={imagenet_norm}, mean={mean}, std={std}")
    else:
        backbone = args.backbone
        if args.imagenet_norm:
            imagenet_norm = True
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            imagenet_norm = False
            if args.mean_std_npy:
                stats = np.load(args.mean_std_npy, allow_pickle=True).item()
                mean = tuple([float(x) for x in stats["mean"]])
                std = tuple([float(x) for x in stats["std"]])
            elif args.compute_dataset_mean_std:
                mean, std = compute_dataset_mean_std(npy_files, sample_every=1)
            else:
                raise ValueError(
                    "Not using ImageNet normalization. Provide --mean_std_npy or --compute_dataset_mean_std."
                )
        print(f"[settings] manual: backbone={backbone}, imagenet_norm={imagenet_norm}, mean={mean}, std={std}")

    # Device + dtype
    use_mps = torch.backends.mps.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if use_mps else "cpu"))
    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else torch.float32
    print(f"[device] {device} dtype={dtype}")

    # Model + normalizer
    model = ResNetFeatureExtractor(backbone=backbone, pretrained=True).to(device).eval()
    normalizer = ImageNormalizer(mean, std)

    # Process each episode array
    for in_path in tqdm(npy_files, desc="Extracting"):
        out_path = out_dir / in_path.name  # mirror PCA naming: same stem, .npy
        if args.skip_existing and out_path.exists():
            continue

        imgs = np.load(in_path)  # [N,H,W,3], float32 in [0,1]
        if imgs.ndim != 4 or imgs.shape[-1] != 3:
            raise ValueError(f"{in_path.name}: expected [N,H,W,3], got {imgs.shape}")
        imgs = np.asarray(imgs, dtype=np.float32)

        feats = extract_features_for_array(
            imgs, normalizer, model, device, batch_size=args.batch_size, dtype=dtype
        )
        np.save(out_path, feats.astype(np.float32))  # save as float32 for downstream
        # tqdm line already shows progress; keep prints minimal

    print(f"Done. Saved features to: {out_dir}")


if __name__ == "__main__":
    main()