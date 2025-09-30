import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from skill_helpers import * 

# --- NEW: load images per-episode (reuses your skill mask & actions logic) ---
def get_bc_images_by_episode(dir_, files, skill, image_dir_name='pixel_obs'):
    """
    Like get_bc_data_by_episode, but loads raw images instead of PCA features.
    Expects per-episode npy shaped (T, H, W, 3), float32 in [0,1].
    """
    episodes = []
    for file in files:
        with open(os.path.join(dir_, 'groundTruth', file), 'r') as f:
            lines = f.read().splitlines()  # len = T

        img_path = os.path.join(dir_, image_dir_name, file + '.npy')
        act_path = os.path.join(dir_, 'actions', file + '.npy')

        images  = np.load(img_path)   # [T, H, W, 3] float32
        actions = np.load(act_path)   # [T]

        if len(lines) != len(images) or len(images) != len(actions):
            raise ValueError(
                f"Length mismatch in {file}: "
                f"labels={len(lines)} images={len(images)} actions={len(actions)}"
            )

        skill_mask = np.array([lab == skill for lab in lines], dtype=bool)
        other_mask = ~skill_mask

        ep = dict(
            episode_id=file,
            # keep the same interface names as before, but these are images now
            skill_states=images[skill_mask],     # shape [Ns, H, W, 3]
            skill_actions=actions[skill_mask],
            other_states=images[other_mask],
            other_actions=actions[other_mask],
            images=images,
            actions=actions,
            skill_mask=skill_mask
        )
        episodes.append(ep)
    return episodes


def compute_channel_mean_std(X):
    """
    X: numpy array [N, H, W, 3], float32 in [0,1]
    Returns per-channel mean/std as tuples of floats.
    """
    # accumulate sums and sums of squares per channel in NHWC
    N, H, W, C = X.shape
    n_pixels = N * H * W
    # reshape to [N*H*W, C] without copying if possible
    flat = X.reshape(-1, C).astype(np.float64)  # higher precision for sums
    chan_sum = flat.sum(axis=0)                             # [3]
    chan_sqsum = np.square(flat).sum(axis=0)                # [3]
    mean = chan_sum / n_pixels
    var = chan_sqsum / n_pixels - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    return tuple(mean.tolist()), tuple(std.tolist())
# --- replaces Standardizer for images: just channel-wise normalize ---
class ImageNormalizer:
    """
    Normalizes CHW images using dataset-specific per-channel mean/std.
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
        self.std  = torch.tensor(std,  dtype=torch.float32).view(3,1,1)
        # avoid tiny std that can explode activations
        self.std = torch.clamp(self.std, min=1e-3)

    def __call__(self, x):
        # x: [3,H,W] in [0,1]
        return (x - self.mean) / self.std

class ImageBCDataset(Dataset):
    """
    Frame-level tuples: returns (img_t, action_t)
    - img_t: torch.float32 [3, H, W], normalized
    - action_t: torch.long
    """
    def __init__(self, X, y, normalizer=None, augment=False):
        assert X.shape[0] == y.shape[0]
        self.X = X      # numpy: [N, H, W, 3]
        self.y = y      # numpy: [N]
        self.norm = normalizer if normalizer is not None else ImageNormalizer()
        self.augment = augment

    def __len__(self):
        return self.X.shape[0]

    def _to_chw(self, img):
        # NHWC -> CHW
        return torch.from_numpy(np.transpose(img, (2,0,1))).float()

        # replace _random_crop_or_resize with resize-only
    def _resize(self, x, target=256):
        x = x.unsqueeze(0)  # [1,C,H,W]
        x = F.interpolate(x, size=(target, target), mode='bilinear', align_corners=False)
        return x.squeeze(0)

    def __getitem__(self, idx):
        img = self._to_chw(self.X[idx])  # [3,H,W] in [0,1]
        img = self._resize(img, target=256)   # keep entire board
        img = self.norm(img)
        y = torch.tensor(self.y[idx]).long()
        return img, y


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class PolicyCNN(nn.Module):
    def __init__(self, n_actions=16):
        super().__init__()
        # 274x274 -> downsample a few times
        self.stem = nn.Sequential(
            ConvBlock(3,   32, k=7, s=2, p=3),   # ~137x137
            ConvBlock(32,  32),
            nn.MaxPool2d(2),                     # ~68x68
        )
        self.stage2 = nn.Sequential(
            ConvBlock(32,  64),
            ConvBlock(64,  64),
            nn.MaxPool2d(2),                     # ~34x34
        )
        self.stage3 = nn.Sequential(
            ConvBlock(64,  128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),                     # ~17x17
        )
        self.stage4 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )
        self.head = nn.Linear(256, n_actions)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1)   # [B, C, 1, 1]
        x = torch.flatten(x, 1)           # [B, C]
        return self.head(x)
        
    
# ...existing code...
# Replace hardcoded skill with argparse
parser = argparse.ArgumentParser(description="Train CNN policy for a specific skill")
parser.add_argument(
    "--skill",
    type=str,
    default="wood",
    help=f"Skill to train. Available"
)
args = parser.parse_args()

dir_ = 'Traces/stone_pickaxe_easy'
files = os.listdir(os.path.join(dir_, 'groundTruth'))

unique_skills = get_unique_skills(dir_, files)
skill = args.skill

print(f"===== TRAINING SKILL {skill} (CNN) =====")
episodes = get_bc_images_by_episode(dir_, files, skill, image_dir_name='pixel_obs')  # <-- NEW

# Add a checkpoint directory/path
ckpt_dir = os.path.join(dir_, 'bc_checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, f'{skill}_policy_cnn.pt')

# (same split over episodes)
rng = np.random.default_rng(0)
idx = np.arange(len(episodes))
rng.shuffle(idx)
n = len(idx)
train_idx = idx[: int(0.8*n)]
val_idx   = idx[int(0.8*n): int(0.9*n)]
test_idx  = idx[int(0.9*n):]

train_eps = [episodes[i] for i in train_idx]
val_eps   = [episodes[i] for i in val_idx]
test_eps  = [episodes[i] for i in test_idx]

# reuse your flatten function name but now it concatenates images
def bc_flatten_split_images(episode_dicts, use_skill=True):
    X, y = [], []
    s_key = 'skill_states' if use_skill else 'other_states'
    a_key = 'skill_actions' if use_skill else 'other_actions'
    for ep in episode_dicts:
        X.append(ep[s_key])  # [Ni, H, W, 3]
        y.append(ep[a_key])  # [Ni]
    if len(X) == 0:
        return np.empty((0,274,274,3), dtype=np.float32), np.empty((0,), dtype=int)
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

X_tr, y_tr = bc_flatten_split_images(train_eps, use_skill=True)
X_va, y_va = bc_flatten_split_images(val_eps,   use_skill=True)
X_te, y_te = bc_flatten_split_images(test_eps,  use_skill=True)

print(X_tr.shape, y_tr.shape)
print(X_va.shape, y_va.shape)
print(X_te.shape, y_te.shape)

mean_tr, std_tr = compute_channel_mean_std(X_tr)
print("train mean:", mean_tr, "train std:", std_tr)

# build datasets/loaders
normalizer = ImageNormalizer(mean_tr, std_tr)
train_ds = ImageBCDataset(X_tr, y_tr, normalizer=normalizer, augment=False)  # was True
val_ds   = ImageBCDataset(X_va, y_va, normalizer=normalizer, augment=False)
test_ds  = ImageBCDataset(X_te, y_te, normalizer=normalizer, augment=False)

n_actions = 16
counts = np.bincount(y_tr, minlength=n_actions).astype(np.float64)
inv = np.zeros_like(counts); obs = counts > 0
inv[obs] = 1.0 / counts[obs]
sample_w = inv[y_tr]
sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

use_mps = torch.backends.mps.is_available()
train_loader = DataLoader(train_ds, batch_size=64,  sampler=sampler, shuffle=False, pin_memory=not use_mps, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, pin_memory=not use_mps, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, pin_memory=not use_mps, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if use_mps else 'cpu'))

print("Using device:", device)

model = PolicyCNN(n_actions=n_actions).to(device)   # <-- NEW MODEL

criterion = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

best_val = float('inf'); patience=10; bad=0
for epoch in range(150):  # CNN converges slower per step; start with ~60-100
    # train
    model.train()
    total = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * xb.size(0)
    train_loss = total / len(train_ds)

    # val
    model.eval()
    with torch.no_grad():
        tot, correct = 0.0, 0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            tot += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
        val_loss = tot / len(val_ds)
        val_acc = correct / len(val_ds)

    print(f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.3f}")

    if val_loss + 1e-6 < best_val:
        best_val = val_loss
        bad = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        # Save checkpoint on improvement
        torch.save({
            'state_dict': best_state,
            'mean': mean_tr,
            'std': std_tr,
            'n_actions': n_actions,
            'skill': skill,
            'arch': 'PolicyCNN',
            'epoch': epoch,
            'val_loss': best_val,
        }, ckpt_path)
    else:
        bad += 1
        if bad >= patience:
            break

# Load best state back to model
model.load_state_dict(best_state)
model.to(device)
print(f"Loaded best model. Checkpoint saved at: {ckpt_path}")

# test
model.eval()
with torch.no_grad():
    tot, correct = 0.0, 0
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        tot += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
    print(f"TEST  NLL {tot/len(test_ds):.4f} | ACC {correct/len(test_ds):.3f}")

# Optional: how to load later
# ckpt = torch.load(ckpt_path, map_location='cpu')
# model = PolicyCNN(n_actions=ckpt['n_actions'])
# model.load_state_dict(ckpt['state_dict'])
# normalizer = ImageNormalizer(ckpt['mean'], ckpt['std'])