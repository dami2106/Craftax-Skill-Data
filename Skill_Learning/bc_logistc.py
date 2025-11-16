import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Data utilities
# -------------------------

def get_bc_pca_by_episode(
    dir_,
    files,
    skill,
    feature_dir_name="pca_features_650",
    skill_dir_name="groundTruth",
):
    """
    Loads PCA features per-episode and actions.
    Splits each episode into skill vs other frames using provided skill labels.

    Expects:
      - groundTruth/<episode_id>            : text file with one label per line
      - pca_features_650/<episode_id>.npy   : [T, D] PCA features
      - actions/<episode_id>.npy            : [T] actions
    """
    episodes = []
    for file in files:
        # load skill labels
        with open(os.path.join(dir_, skill_dir_name, file), "r") as f:
            lines = f.read().splitlines()  # len = T

        feat_path = os.path.join(dir_, feature_dir_name, file + ".npy")
        act_path = os.path.join(dir_, "actions", file + ".npy")

        feats = np.load(feat_path)   # [T, D]
        actions = np.load(act_path)  # [T]

        # small guard (same pattern as your image BC)
        if len(lines) != len(actions):
            lines.append(lines[-1])

        if len(lines) != len(feats) or len(feats) != len(actions):
            raise ValueError(
                f"Length mismatch in {file}: labels={len(lines)} "
                f"features={len(feats)} actions={len(actions)}"
            )

        skill_mask = np.array([lab == skill for lab in lines], dtype=bool)
        other_mask = ~skill_mask

        ep = dict(
            episode_id=file,
            skill_features=feats[skill_mask],   # [Ns, D]
            skill_actions=actions[skill_mask],
            other_features=feats[other_mask],
            other_actions=actions[other_mask],
            features=feats,
            actions=actions,
            skill_mask=skill_mask,
        )
        episodes.append(ep)
    return episodes


def bc_flatten_split_features(episode_dicts, use_skill=True):
    """
    Concatenate features/actions across episodes.
    Mirrors your bc_flatten_split_images, but for PCA features.
    """
    X, y = [], []
    s_key = "skill_features" if use_skill else "other_features"
    a_key = "skill_actions" if use_skill else "other_actions"
    for ep in episode_dicts:
        if ep[s_key].shape[0] == 0:
            continue
        X.append(ep[s_key])
        y.append(ep[a_key])

    if len(X) == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=int)

    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)


# -------------------------
# Normalization for PCA features
# -------------------------

def compute_feature_mean_std(X):
    """
    X: [N, D] numpy array
    Returns (mean, std) as 1D numpy arrays (float32).

    We re-standardize PCA features (0 mean, unit var per dim),
    which generally helps linear models.
    """
    if X.shape[0] == 0:
        raise ValueError("Empty feature array, cannot compute mean/std.")
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


class PCAFeatureNormalizer:
    def __init__(self, mean_vec, std_vec):
        self.mean = torch.from_numpy(mean_vec).float()  # [D]
        self.std = torch.from_numpy(std_vec).float()
        self.std = torch.clamp(self.std, min=1e-3)

    def __call__(self, x):
        # x: [D]
        return (x - self.mean) / self.std


# -------------------------
# Dataset
# -------------------------

class PCABCDataset(Dataset):
    """
    Returns (features, action) where:
      - features: torch.float32 [D] normalized
      - action: torch.long
    """
    def __init__(self, X, y, normalizer):
        assert X.shape[0] == y.shape[0]
        self.X = X.astype(np.float32)
        self.y = y
        self.norm = normalizer

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        feat = torch.from_numpy(self.X[idx])  # [D]
        feat = self.norm(feat)
        y = torch.tensor(self.y[idx]).long()
        return feat, y


# -------------------------
# MLP policy model
# -------------------------

class PCAPolicy(nn.Module):
    """
    Small MLP for non-linear action prediction from PCA features.
    Allows the model to learn non-linear decision boundaries in PCA space.
    """
    def __init__(self, z_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, z):
        return self.net(z)  # logits over n_actions


# -------------------------
# Training utilities
# -------------------------

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    n_samples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            n_samples += xb.size(0)

    if n_samples == 0:
        return float("nan"), float("nan")

    return total_loss / n_samples, correct / n_samples


# -------------------------
# Main training script
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="MLP BC on PCA features")
    parser.add_argument("--skill", type=str, default="wood", help="Skill to train")
    parser.add_argument("--dir", type=str, default="Traces/stone_pick_static",
                        help="Dataset root")
    parser.add_argument("--skills_name", type=str, default="groundTruth",
                        help="Dir name for skill labels")
    parser.add_argument("--feature_dir_name", type=str, default="pca_features_650",
                        help="Dir name that stores per-episode PCA .npy files")
    parser.add_argument("--save_dir", type=str, default="bc_checkpoints_pca",
                        help="Where to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs per hyperparam combo")
    parser.add_argument("--use_sampler", action="store_true",
                        help="Use WeightedRandomSampler (else shuffle)")
    parser.add_argument("--n_actions", type=int, default=16,
                        help="Number of discrete actions")
    # Hyperparameter grid (you can also edit in code below)
    parser.add_argument("--grid", action="store_true",
                        help="Run small hyperparameter sweep (recommended)")
    args = parser.parse_args()

    dir_ = args.dir
    skills_dir = os.path.join(dir_, args.skills_name)
    files = os.listdir(skills_dir)
    skill = args.skill

    print(f"===== MLP BC ON PCA FEATURES | SKILL = {skill} =====")

    # -------------------------
    # Load PCA episodes
    # -------------------------
    episodes = get_bc_pca_by_episode(
        dir_=dir_,
        files=files,
        skill=skill,
        feature_dir_name=args.feature_dir_name,
        skill_dir_name=args.skills_name,
    )

    # Split by episode indices
    rng = np.random.default_rng(0)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)
    n = len(idx)

    train_idx = idx[: int(0.8 * n)]
    val_idx = idx[int(0.8 * n) : int(0.9 * n)]
    test_idx = idx[int(0.9 * n) :]

    train_eps = [episodes[i] for i in train_idx]
    val_eps = [episodes[i] for i in val_idx]
    test_eps = [episodes[i] for i in test_idx]

    X_tr, y_tr = bc_flatten_split_features(train_eps, use_skill=True)
    X_va, y_va = bc_flatten_split_features(val_eps, use_skill=True)
    X_te, y_te = bc_flatten_split_features(test_eps, use_skill=True)

    print("Train:", X_tr.shape, y_tr.shape)
    print("Val:  ", X_va.shape, y_va.shape)
    print("Test: ", X_te.shape, y_te.shape)

    if X_tr.shape[0] == 0:
        raise RuntimeError("No training data for this skill. Check labels/skill name.")

    feature_dim = X_tr.shape[1]
    print(f"Feature dim = {feature_dim}")

    # -------------------------
    # Normalize features
    # -------------------------
    mean_tr, std_tr = compute_feature_mean_std(X_tr)
    print("Feature normalization: mean/std shapes:", mean_tr.shape, std_tr.shape)
    normalizer = PCAFeatureNormalizer(mean_tr, std_tr)

    train_ds = PCABCDataset(X_tr, y_tr, normalizer=normalizer)
    val_ds = PCABCDataset(X_va, y_va, normalizer=normalizer)
    test_ds = PCABCDataset(X_te, y_te, normalizer=normalizer)

    # Class balancing weights (same idea as your ResNet BC)
    n_actions = args.n_actions
    counts = np.bincount(y_tr, minlength=n_actions).astype(np.float64)
    inv = np.zeros_like(counts)
    obs = counts > 0
    inv[obs] = 1.0 / counts[obs]
    sample_w = inv[y_tr]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    use_mps = torch.backends.mps.is_available()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if use_mps else "cpu")
    )
    pin_mem = not use_mps

    if args.use_sampler:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            pin_memory=pin_mem,
            num_workers=4,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=pin_mem,
            num_workers=4,
        )

    val_loader = DataLoader(
        val_ds, batch_size=1024, shuffle=False, pin_memory=pin_mem, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=1024, shuffle=False, pin_memory=pin_mem, num_workers=4
    )

    print("Using device:", device)

    # -------------------------
    # Hyperparameter grid
    # -------------------------
    if args.grid:
        # Edit these lists for a slightly bigger/smaller sweep.
        lr_list = [3e-3, 1e-3, 3e-4]
        wd_list = [0.0, 1e-4, 1e-3]
    else:
        lr_list = [1e-3]
        wd_list = [1e-4]

    hp_grid = [(lr, wd) for lr in lr_list for wd in wd_list]
    print("Hyperparameter grid (lr, weight_decay):")
    for lr, wd in hp_grid:
        print(f"  lr={lr:.1e}, wd={wd:.1e}")

    # -------------------------
    # Training loop over grid
    # -------------------------
    criterion = nn.CrossEntropyLoss()
    global_best_val_loss = float("inf")
    global_best_state = None
    global_best_meta = {}

    es_patience = 25

    ckpt_dir = os.path.join(dir_, args.save_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{skill}_pca_mlp_policy.pt")

    for (lr, wd) in hp_grid:
        print("\n======================================")
        print(f"Starting run with lr={lr:.1e}, weight_decay={wd:.1e}")
        print("======================================")

        model = PCAPolicy(z_dim=feature_dim, n_actions=n_actions).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=3,
            threshold=1e-3,
            threshold_mode="rel",
            cooldown=1,
            min_lr=1e-6,
            verbose=True,
        )

        best_val_loss = float("inf")
        best_state_for_run = None
        bad_epochs = 0

        for epoch in range(args.epochs):
            model.train()
            total_train_loss = 0.0
            n_train = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_train_loss += loss.item() * xb.size(0)
                n_train += xb.size(0)

            train_loss = total_train_loss / max(1, n_train)

            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            sched.step(val_loss)
            cur_lr = opt.param_groups[0]["lr"]

            print(
                f"[lr={lr:.1e}, wd={wd:.1e}] epoch {epoch:03d} | "
                f"lr {cur_lr:.2e} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | acc {val_acc:.3f}"
            )

            # Early stopping on this run
            if val_loss + 1e-6 < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                best_state_for_run = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= es_patience:
                    print("Early stopping on this run.")
                    break

        # Compare to global best across all grid runs
        if best_state_for_run is not None and best_val_loss < global_best_val_loss:
            global_best_val_loss = best_val_loss
            global_best_state = best_state_for_run
            global_best_meta = {
                "lr": lr,
                "weight_decay": wd,
                "val_loss": best_val_loss,
            }

    if global_best_state is None:
        raise RuntimeError("No successful training run found.")

    # -------------------------
    # Save global best checkpoint
    # -------------------------
    print("\n==== Global best hyperparams ====")
    print(global_best_meta)
    final_model = PCAPolicy(z_dim=feature_dim, n_actions=n_actions)
    final_model.load_state_dict(global_best_state)
    torch.save(
        {
            "state_dict": global_best_state,
            "mean": mean_tr,
            "std": std_tr,
            "n_actions": n_actions,
            "skill": skill,
            "arch": "PCAPolicy_MLP",
            "feature_dim": feature_dim,
            "best_hparams": global_best_meta,
        },
        ckpt_path,
    )
    print(f"Saved best MLP PCA policy to: {ckpt_path}")

    # -------------------------
    # Test set evaluation
    # -------------------------
    final_model.to(device)
    test_loss, test_acc = evaluate(final_model, test_loader, criterion, device)
    print(f"\nTEST NLL {test_loss:.4f} | ACC {test_acc:.3f}")


if __name__ == "__main__":
    main()
