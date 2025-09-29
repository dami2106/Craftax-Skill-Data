import os
import json
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_recall_curve,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from skill_helpers import *
from joblib import dump
import pandas as pd 
from math import isnan
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GroupKFold
from pulearn import ElkanotoPuClassifier, BaggingPuClassifier
from sklearn.svm import SVC  # use SVC (probability=True); LinearSVC has no predict_proba

SEED = 42
rng = np.random.default_rng(SEED)

dir_ = 'Traces/stone_pickaxe_easy'

# Directory to save trained start models & metadata
models_dir = os.path.join(dir_, 'pu_start_models')
os.makedirs(models_dir, exist_ok=True)
files = os.listdir(os.path.join(dir_, 'groundTruth'))


def make_pu_clf(
    method: str = "elkanoto",         # "elkanoto" | "bagging"
    C: float = 10.0,
    kernel: str = "rbf",              # "linear" or "rbf" are good starts
    gamma: str | float = "scale",
    hold_out_ratio: float = 0.2,      # used by Elkanoto
    n_estimators: int = 15,           # used by BaggingPuClassifier
    seed: int = SEED,
):
    """
    Builds a PU-learning estimator that exposes predict_proba.
    Any scikit-learn classifier can be used underneath; we pick SVC(probability=True).
    """
    base = make_pipeline(
        StandardScaler(),
        SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=seed)
    )

    if method.lower() == "bagging":
        # Bagging over subsamples of unlabeled examples
        pu_estimator = BaggingPuClassifier(base_estimator=base, n_estimators=n_estimators, random_state=seed)
    else:
        # Classic Elkan–Noto PU
        pu_estimator = ElkanotoPuClassifier(estimator=base, hold_out_ratio=hold_out_ratio, random_state=seed)

    return pu_estimator

def best_threshold_from_pr(y_true, p_scores):
    prec, rec, thr = precision_recall_curve(y_true, p_scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)

    if len(thr) == 0:
        # degenerate (all scores same); fall back
        best_idx = int(np.nanargmax(f1s))
        return 0.5, float(f1s[best_idx])

    # thresholds correspond to points 1..n in (prec, rec)
    valid = f1s[1:]
    best_idx = int(np.nanargmax(valid)) + 1  # shift back into full f1s indexing
    return float(thr[best_idx - 1]), float(f1s[best_idx])


def fit_with_threshold_grouped_pu(X, y, groups, *,
                                  method="elkanoto",
                                  C=10.0,
                                  kernel="rbf",
                                  gamma="scale",
                                  hold_out_ratio=0.2,
                                  n_estimators=15,
                                  seed=SEED):
    """
    Train a PU model on a group-split train/val; pick threshold by max F1 on the val set,
    then refit on *all* provided (X, y) for final model.
    y must be 1 for positive, 0 for unlabeled (no true negatives).
    """
    # Inner val split by episode (no test leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(gss.split(X, y, groups))
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # Build & fit PU estimator on the training fold
    pu = make_pu_clf(
        method=method, C=C, kernel=kernel, gamma=gamma,
        hold_out_ratio=hold_out_ratio, n_estimators=n_estimators, seed=seed
    )
    pu.fit(X_tr, y_tr)

    # Pick threshold on group-held-out validation
    val_proba = pu.predict_proba(X_val)[:, 1]
    thr, val_f1 = best_threshold_from_pr(y_val, val_proba)

    # Refit on *all* provided data (train split of outer loop), same hyperparams
    pu_final = make_pu_clf(
        method=method, C=C, kernel=kernel, gamma=gamma,
        hold_out_ratio=hold_out_ratio, n_estimators=n_estimators, seed=seed
    )
    pu_final.fit(X, y)

    return pu_final, float(thr), float(val_f1)


results = {}
skills = get_unique_skills(dir_, files)

# Pick a PU method and base SVC hyperparams once (tweak as needed)
PU_METHOD = "elkanoto"   # or "bagging"
PU_C      = 10.0
PU_KERNEL = "rbf"        # try "linear" if you want speed/simplicity
PU_GAMMA  = "scale"

for skill in skills:
    X, y, groups = build_startability_dataset(dir_, skill, files, features_dirname='pca_features_750')

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X))
    X, y, groups = X[perm], y[perm], groups[perm]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train    = groups[train_idx]

    print(f"Skill: {skill}")
    print("train balance:", np.bincount(y_train))
    print("test  balance:",  np.bincount(y_test))

    # NEW: PU fit + threshold
    clf, thr, val_f1 = fit_with_threshold_grouped_pu(
        X_train, y_train, groups_train,
        method=PU_METHOD, C=PU_C, kernel=PU_KERNEL, gamma=PU_GAMMA,
        hold_out_ratio=0.2, n_estimators=15, seed=SEED
    )

    # Inspect probabilities and counts at some thresholds
    proba_test = clf.predict_proba(X_test)[:, 1]
    print("min/max prob:", float(proba_test.min()), float(proba_test.max()))
    for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
        preds_t = (proba_test >= t).astype(int)
        print(t, int(preds_t.sum()))
    print("Chosen threshold (from PR/F1 on val):", thr, " (val F1=", f"{val_f1:.4f}", ")")

    # Evaluate at chosen threshold
    y_pred = (proba_test >= thr).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    results[skill] = {
        "threshold": thr,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "val_f1": float(val_f1)
    }

    # Persist model + metadata (unchanged)
    model_path = os.path.join(models_dir, f"{skill}_clf.joblib")
    meta_path  = os.path.join(models_dir, f"{skill}_meta.json")
    try:
        dump(clf, model_path)
        with open(meta_path, 'w') as f:
            json.dump({
                'skill': skill,
                'threshold': thr,
                'val_f1': val_f1,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'n_train_pos': int((y_train == 1).sum()),
                'n_train_neg': int((y_train == 0).sum()),  # these are UNLABELED in PU terms
                'n_test_pos': int((y_test == 1).sum()),
                'n_test_neg': int((y_test == 0).sum()),
                'seed': SEED,
                'pu_method': PU_METHOD,
                'svc_kernel': PU_KERNEL,
                'svc_C': PU_C,
                'svc_gamma': PU_GAMMA
            }, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save model/metadata for skill {skill}: {e}")

rows = []
tot_tn = tot_fp = tot_fn = tot_tp = 0

for skill, res in results.items():
    cm = res["confusion_matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]
    tot_tn += int(tn); tot_fp += int(fp); tot_fn += int(fn); tot_tp += int(tp)

    support_pos = int(tp + fn)
    support_neg = int(tn + fp)
    support_all = support_pos + support_neg
    acc = (tp + tn) / support_all if support_all else float("nan")

    rows.append({
        "skill": skill,
        "pos_support": support_pos,
        "neg_support": support_neg,
        "threshold": res["threshold"],
        "precision": res["precision"],
        "recall": res["recall"],
        "f1": res["f1"],
        "accuracy": acc,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    })

# Overall (micro) metrics
overall_support = tot_tp + tot_fp + tot_fn + tot_tn
overall_precision = (tot_tp / (tot_tp + tot_fp)) if (tot_tp + tot_fp) else 0.0
overall_recall    = (tot_tp / (tot_tp + tot_fn)) if (tot_tp + tot_fn) else 0.0
if (overall_precision + overall_recall) > 0:
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
else:
    overall_f1 = 0.0
overall_accuracy  = (tot_tp + tot_tn) / overall_support if overall_support else float("nan")

# Macro (mean across skills)
macro_precision = float(np.mean([r["precision"] for r in rows])) if rows else float("nan")
macro_recall    = float(np.mean([r["recall"]    for r in rows])) if rows else float("nan")
macro_f1        = float(np.mean([r["f1"]        for r in rows])) if rows else float("nan")
macro_accuracy  = float(np.mean([r["accuracy"]  for r in rows])) if rows else float("nan")

# Pretty print
print("\n" + "="*80)
print("PER-SKILL METRICS (sorted by F1 desc)")
print("="*80)


df = pd.DataFrame(rows)
df = df.sort_values("f1", ascending=False)
disp_cols = ["skill", "pos_support", "neg_support", "threshold",
                "precision", "recall", "f1", "accuracy", "tp", "fp", "fn"]
# Round numeric columns for readability
for c in ["threshold", "precision", "recall", "f1", "accuracy"]:
    df[c] = df[c].astype(float).round(3)
print(df[disp_cols].to_string(index=False))

# Save per-skill metrics table & overall summary
metrics_csv = os.path.join(models_dir, 'per_skill_metrics.csv')
metrics_json = os.path.join(models_dir, 'summary_metrics.json')
try:
    df.to_csv(metrics_csv, index=False)
    with open(metrics_json, 'w') as f:
        json.dump({
            'overall': {
                'support': overall_support,
                'tp': tot_tp, 'fp': tot_fp, 'fn': tot_fn, 'tn': tot_tn,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'accuracy': overall_accuracy
            },
            'macro': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1,
                'accuracy': macro_accuracy
            }
        }, f, indent=2)
except Exception as e:
    print(f"[WARN] Failed to save aggregate metrics: {e}")

print("\n" + "="*80)
print("OVERALL (MICRO) METRICS — pooled over all skills")
print("="*80)
print(f"Support (all skills): {overall_support}")
print(f"TP={tot_tp}  FP={tot_fp}  FN={tot_fn}  TN={tot_tn}")
print(f"Precision: {overall_precision:.3f}  Recall: {overall_recall:.3f}  F1: {overall_f1:.3f}  Accuracy: {overall_accuracy:.3f}")

print("\n" + "="*80)
print("MACRO AVERAGES — mean of per-skill metrics")
print("="*80)
print(f"Precision: {macro_precision:.3f}  Recall: {macro_recall:.3f}  F1: {macro_f1:.3f}  Accuracy: {macro_accuracy:.3f}")
print("="*80 + "\n")