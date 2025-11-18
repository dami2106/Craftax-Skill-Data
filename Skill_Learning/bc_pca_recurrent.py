import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
from itertools import product


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
    episodes = []
    for file in files:
        with open(os.path.join(dir_, skill_dir_name, file), "r") as f:
            lines = f.read().splitlines()

        feat_path = os.path.join(dir_, feature_dir_name, file + ".npy")
        act_path = os.path.join(dir_, "actions", file + ".npy")

        feats = np.load(feat_path)
        actions = np.load(act_path)

        if len(lines) != len(actions):
            lines.append(lines[-1])

        if len(lines) != len(feats) or len(feats) != len(actions):
            raise ValueError(
                f"Length mismatch in {file}: labels={len(lines)} "
                f"features={len(feats)} actions={len(actions)}"
            )

        skill_mask = np.array([lab == skill for lab in lines], dtype=bool)
        other_mask = ~skill_mask

        episodes.append(
            dict(
                episode_id=file,
                skill_features=feats[skill_mask],
                skill_actions=actions[skill_mask],
                other_features=feats[other_mask],
                other_actions=actions[other_mask],
                features=feats,
                actions=actions,
                skill_mask=skill_mask,
            )
        )
    return episodes


def bc_flatten_split_features(episode_dicts, use_skill=True):
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


def compute_feature_mean_std(X):
    if X.shape[0] == 0:
        raise ValueError("Empty feature array, cannot compute mean/std.")
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


class PCAFeatureNormalizer:
    def __init__(self, mean_vec, std_vec):
        self.mean = torch.from_numpy(mean_vec).float()
        self.std = torch.from_numpy(std_vec).float()
        self.std = torch.clamp(self.std, min=1e-3)

    def __call__(self, x):
        return (x - self.mean) / self.std


# -------------------------
# Sequence utilities
# -------------------------

def build_sequence_windows(episodes, seq_len, use_skill=True):
    sequences = []
    labels = []

    for ep in episodes:
        feats = ep["features"]
        actions = ep["actions"]
        mask = ep["skill_mask"] if use_skill else np.ones_like(ep["skill_mask"], dtype=bool)

        L = feats.shape[0]
        if L < seq_len:
            continue

        for t in range(seq_len - 1, L):
            if not mask[t]:
                continue
            window = feats[t - seq_len + 1 : t + 1]
            sequences.append(window)
            labels.append(actions[t])

    if len(sequences) == 0:
        return (
            np.empty((0, seq_len, 0), dtype=np.float32),
            np.empty((0,), dtype=int),
        )

    seq_arr = np.stack(sequences).astype(np.float32)
    labels_arr = np.array(labels, dtype=int)
    return seq_arr, labels_arr


def flatten_sequences(seqs):
    if seqs.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)
    N, T, D = seqs.shape
    return seqs.reshape(N * T, D)


class PCABehaviorSequenceDataset(Dataset):
    def __init__(self, sequences, actions, normalizer):
        assert sequences.shape[0] == actions.shape[0]
        self.X = sequences.astype(np.float32)
        self.y = actions.astype(np.int64)
        self.norm = normalizer

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.X[idx])
        seq = self.norm(seq)
        action = torch.tensor(self.y[idx]).long()
        return seq, action


# -------------------------
# Model
# -------------------------

class GRUPCABehaviorPolicy(nn.Module):
    def __init__(
        self,
        z_dim,
        n_actions,
        hidden_size=256,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=z_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Linear(hidden_size, n_actions)

    def forward(self, seqs, hidden=None):
        out, h_n = self.gru(seqs, hidden)
        last_hidden = h_n[-1]
        logits = self.head(last_hidden)
        return logits, h_n


# -------------------------
# Evaluation
# -------------------------

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    n_samples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            n_samples += xb.size(0)

    if n_samples == 0:
        return float("nan"), float("nan")

    return total_loss / n_samples, correct / n_samples


# -------------------------
# Training script
# -------------------------

def train_for_seq_len(
    seq_len,
    train_eps,
    val_eps,
    test_eps,
    args,
    device,
    pin_mem,
):
    print(f"\n===== GRU BC ON PCA SEQS | SKILL = {args.skill} | T = {seq_len} =====")

    X_tr, y_tr = build_sequence_windows(train_eps, seq_len=seq_len, use_skill=True)
    X_va, y_va = build_sequence_windows(val_eps, seq_len=seq_len, use_skill=True)
    X_te, y_te = build_sequence_windows(test_eps, seq_len=seq_len, use_skill=True)

    print("Train seqs:", X_tr.shape, y_tr.shape)
    print("Val seqs:  ", X_va.shape, y_va.shape)
    print("Test seqs: ", X_te.shape, y_te.shape)

    if X_tr.shape[0] == 0:
        raise RuntimeError(f"No training sequences for T={seq_len}.")

    feature_dim = X_tr.shape[-1]
    mean_tr, std_tr = compute_feature_mean_std(flatten_sequences(X_tr))
    normalizer = PCAFeatureNormalizer(mean_tr, std_tr)

    train_ds = PCABehaviorSequenceDataset(X_tr, y_tr, normalizer)
    val_ds = PCABehaviorSequenceDataset(X_va, y_va, normalizer)
    test_ds = PCABehaviorSequenceDataset(X_te, y_te, normalizer)

    n_actions = args.n_actions
    counts = np.bincount(y_tr, minlength=n_actions).astype(np.float64)
    inv = np.zeros_like(counts)
    obs = counts > 0
    inv[obs] = 1.0 / counts[obs]
    sample_w = inv[y_tr]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

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
        val_ds, batch_size=256, shuffle=False, pin_memory=pin_mem, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, pin_memory=pin_mem, num_workers=4
    )

    if args.grid:
        lr_list = args.lr_list
        wd_list = args.wd_list
        hidden_sizes = args.hidden_sizes
        num_layers_list = args.num_layers_list
        dropout_list = args.dropouts
    else:
        lr_list = [args.lr]
        wd_list = [args.weight_decay]
        hidden_sizes = [args.hidden_size]
        num_layers_list = [args.num_layers]
        dropout_list = [args.dropout]

    hp_grid = list(product(lr_list, wd_list, hidden_sizes, num_layers_list, dropout_list))
    print("Hyperparameter grid (lr, weight_decay, hidden, layers, dropout):")
    for lr, wd, h, layers, drop in hp_grid:
        print(f"  lr={lr:.1e}, wd={wd:.1e}, hidden={h}, layers={layers}, drop={drop}")

    criterion = nn.CrossEntropyLoss()
    global_best_val_loss = float("inf")
    global_best_state = None
    global_best_meta = {}

    best_hparams = None

    for (lr, wd, hidden_size, num_layers, dropout) in hp_grid:
        print("\n--------------------------------------")
        print(
            "Starting run with "
            f"lr={lr:.1e}, weight_decay={wd:.1e}, "
            f"hidden={hidden_size}, layers={num_layers}, dropout={dropout}"
        )
        print("--------------------------------------")

        model = GRUPCABehaviorPolicy(
            z_dim=feature_dim,
            n_actions=n_actions,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
        opt = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=25,
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
                logits, _ = model(xb)
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
                f"[T={seq_len}] epoch {epoch:03d} | lr {cur_lr:.2e} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.3f} | "
                f"hp lr={lr:.1e}, wd={wd:.1e}, hidden={hidden_size}, "
                f"layers={num_layers}, drop={dropout}"
            )

            if val_loss + 1e-6 < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                best_state_for_run = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= args.es_patience:
                    print("Early stopping on this run.")
                    break

        if best_state_for_run is not None and best_val_loss < global_best_val_loss:
            global_best_val_loss = best_val_loss
            global_best_state = best_state_for_run
            best_hparams = {
                "lr": lr,
                "weight_decay": wd,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
            }
            global_best_meta = {
                **best_hparams,
                "val_loss": best_val_loss,
            }

    if global_best_state is None:
        raise RuntimeError("No successful training run found.")

    ckpt_dir = os.path.join(args.dir, args.save_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.skill}_gru_seq{seq_len}.pt")

    if best_hparams is None:
        raise RuntimeError("Best hyperparameters not recorded.")

    final_model = GRUPCABehaviorPolicy(
        z_dim=feature_dim,
        n_actions=n_actions,
        hidden_size=best_hparams["hidden_size"],
        num_layers=best_hparams["num_layers"],
        dropout=best_hparams["dropout"],
    )
    final_model.load_state_dict(global_best_state)

    torch.save(
        {
            "state_dict": global_best_state,
            "mean": mean_tr,
            "std": std_tr,
            "n_actions": n_actions,
            "skill": args.skill,
            "arch": "GRUPCABehaviorPolicy",
            "feature_dim": feature_dim,
            "seq_len": seq_len,
            "best_hparams": global_best_meta,
            "hidden_size": best_hparams["hidden_size"],
            "num_layers": best_hparams["num_layers"],
            "dropout": best_hparams["dropout"],
        },
        ckpt_path,
    )
    print(f"Saved best GRU PCA policy to: {ckpt_path}")

    print("\n==== Best validation configuration ====")
    print(
        f"Seq len {seq_len} | "
        f"lr={global_best_meta['lr']:.1e} | "
        f"wd={global_best_meta['weight_decay']:.1e} | "
        f"hidden={global_best_meta['hidden_size']} | "
        f"layers={global_best_meta['num_layers']} | "
        f"dropout={global_best_meta['dropout']} | "
        f"val_loss={global_best_meta['val_loss']:.4f}"
    )

    final_model.to(device)
    test_loss, test_acc = evaluate(final_model, test_loader, criterion, device)
    print(
        f"\n==== FINAL TEST (T={seq_len}) ====\n"
        f"NLL {test_loss:.4f} | ACC {test_acc:.3f}\n"
        f"(best validation config above)"
    )


def main():
    parser = argparse.ArgumentParser(description="GRU BC on PCA feature sequences")
    parser.add_argument("--skill", type=str, default="wood", help="Skill to train")
    parser.add_argument(
        "--dir", type=str, default="Traces/stone_pick_static", help="Dataset root"
    )
    parser.add_argument(
        "--skills_name", type=str, default="groundTruth", help="Dir name for skill labels"
    )
    parser.add_argument(
        "--feature_dir_name",
        type=str,
        default="pca_features_650",
        help="Dir name for PCA .npy files",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="bc_checkpoints_pca_gru_gt",
        help="Where to save checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--use_sampler", action="store_true")
    parser.add_argument("--n_actions", type=int, default=16)
    parser.add_argument(
        "--seq_lens",
        type=int,
        nargs="+",
        default=[8],
        help="Sequence lengths to train over (run separately per value)",
    )
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument(
        "--lr_list",
        type=float,
        nargs="+",
        default=[3e-3, 1e-3, 3e-4],
        help="Learning rates to sweep when --grid is enabled",
    )

    parser.add_argument(
        "--wd_list",
        type=float,
        nargs="+",
        default=[0.0, 1e-4, 1e-3],
        help="Weight decays to sweep when --grid is enabled",
    )

    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[128, 256],
        help="Hidden sizes to sweep when --grid is enabled",
    )

    parser.add_argument(
        "--num_layers_list",
        type=int,
        nargs="+",
        default=[1, 2],
        help="GRU layer counts to sweep when --grid is enabled",
    )

    parser.add_argument(
        "--dropouts",
        type=float,
        nargs="+",
        default=[0.0, 0.1],
        help="Dropout rates to sweep when --grid is enabled (ignored for 1 layer)",
    )

    parser.add_argument("--grid", action="store_true", help="Run lr/wd/arch sweep")
    parser.add_argument(
        "--es_patience",
        type=int,
        default=15,  # you can even tighten this a bit from 25
        help="Early stopping patience on validation loss",
    )
    args = parser.parse_args()

    dir_ = args.dir
    skills_dir = os.path.join(dir_, args.skills_name)
    files = os.listdir(skills_dir)

    episodes = get_bc_pca_by_episode(
        dir_=dir_,
        files=files,
        skill=args.skill,
        feature_dir_name=args.feature_dir_name,
        skill_dir_name=args.skills_name,
    )

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

    use_mps = torch.backends.mps.is_available()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if use_mps else "cpu")
    )
    pin_mem = not use_mps
    print("Using device:", device)

    for seq_len in args.seq_lens:
        train_for_seq_len(
            seq_len=seq_len,
            train_eps=train_eps,
            val_eps=val_eps,
            test_eps=test_eps,
            args=args,
            device=device,
            pin_mem=pin_mem,
        )


if __name__ == "__main__":
    main()


