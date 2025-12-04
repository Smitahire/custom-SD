#!/usr/bin/env python3
"""
train_features.py

Train a classifier to predict step-bin (25/35/45) using features JSONL
produced by your generation/eval pipeline (fields: prompt_id, prompt, methods->{25,35,45}...).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import sys

# ensure project root on path
proj_root = Path(__file__).resolve().parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from ans_model.models import get_model
from utils.schedules import STEP_BINS, bin_index

def load_jsonl(path):
    arr = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            arr.append(json.loads(line))
    return arr

def build_feature_vector(rec, steps_list):
    methods = rec.get("methods", {})
    feats = []
    # clipscore per step
    for s in steps_list:
        feats.append(float(methods.get(str(s), {}).get("clipscore", 0.0)))
    # latency per step
    for s in steps_list:
        feats.append(float(methods.get(str(s), {}).get("latency_s", 0.0)))
    # pairwise lpips (i<j)
    for i in range(len(steps_list)):
        for j in range(i+1, len(steps_list)):
            a = steps_list[i]; b = steps_list[j]
            ma = methods.get(str(a), {})
            val = ma.get(f"lpips_vs_{b}", None)
            if val is None:
                mb = methods.get(str(b), {})
                val = mb.get(f"lpips_vs_{a}", 0.0)
            feats.append(float(val if val is not None else 0.0))
    # aggregates
    clip_vals = [float(methods.get(str(s), {}).get("clipscore", 0.0)) for s in steps_list]
    lat_vals = [float(methods.get(str(s), {}).get("latency_s", 0.0)) for s in steps_list]
    clip_max = float(max(clip_vals)) if clip_vals else 0.0
    clip_argmax = float(np.argmax(clip_vals)) if clip_vals else 0.0
    clip_mean = float(np.mean(clip_vals)) if clip_vals else 0.0
    feats += [clip_max, clip_argmax, clip_mean, clip_max - clip_mean]
    feats += [float(np.min(lat_vals)) if lat_vals else 0.0, float(np.mean(lat_vals)) if lat_vals else 0.0]
    return np.array(feats, dtype=np.float32)

def prepare_dataset(features_jsonl, label_mode='best_clip', external_labels=None):
    recs = load_jsonl(features_jsonl)
    label_map = {}
    if label_mode == 'from_labels':
        if not external_labels:
            raise ValueError("external_labels file required for label_mode=from_labels")
        for j in load_jsonl(external_labels):
            pid = int(j.get("prompt_id", -1))
            if pid >= 0:
                label_map[pid] = int(j.get("steps", j.get("label_steps", -1)))
    X_list, y_list, meta = [], [], []
    steps_list = list(STEP_BINS)
    for rec in recs:
        pid = int(rec.get("prompt_id", -1))
        if pid < 0: continue
        # determine label
        label_steps = None
        if label_mode == 'best_clip':
            best_s, best_sc = None, -1e9
            for s in steps_list:
                sc = rec.get("methods", {}).get(str(s), {}).get("clipscore", None)
                if sc is None:
                    continue
                if float(sc) > best_sc:
                    best_sc = float(sc); best_s = int(s)
            if best_s is None:
                label_steps = steps_list[0]
            else:
                label_steps = best_s
        else:
            if pid not in label_map: continue
            label_steps = label_map[pid]
        x = build_feature_vector(rec, steps_list)
        X_list.append(x)
        y_list.append(bin_index(label_steps))
        meta.append(rec.get("prompt", ""))
    if len(X_list) == 0:
        raise RuntimeError("No training samples prepared. Check inputs and label_mode.")
    return np.stack(X_list), np.array(y_list), meta

def train_nn(model, X_train, y_train, X_val, y_val, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()
    best_val = -1.0; best_state = None
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(X_train))
        total_loss = 0.0
        for i in range(0, len(X_train), args.batch_size):
            idx = perm[i:i+args.batch_size]
            xb = torch.tensor(X_train[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        val_acc, _ = evaluate_model(model, X_val, y_val, device)
        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()
            torch.save({'model_type': args.model_type, 'state_dict': best_state, 'input_dim': X_train.shape[1]}, args.out)
        print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss/len(X_train):.4f} val_acc={val_acc:.4f} best={best_val:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, model

def evaluate_model(model, X, y, device='cpu'):
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=device)
            logits = model(t)
            preds = logits.argmax(dim=1).cpu().numpy()
    else:
        preds = model.predict(X)
    return (preds == y).mean(), preds

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True, help='features JSONL path (e.g., data/results_features.jsonl)')
    p.add_argument('--labels', required=False, help='(optional) external labels JSONL when label_mode=from_labels')
    p.add_argument('--out', required=True, help='output model path (.pth for NN, .joblib/.pkl for ML)')
    p.add_argument('--model_type', default='tiny_mlp', choices=['tiny_mlp','deep_mlp','cnn','transformer','logistic','xgboost','random_forest'])
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--val_split', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--label_mode', choices=['best_clip','from_labels'], default='best_clip')
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    STEP_LIST = list(STEP_BINS)
    print("Using STEP_BINS:", STEP_LIST)

    X, y, meta = prepare_dataset(args.features, label_mode=args.label_mode, external_labels=args.labels)
    print(f"Prepared dataset: X.shape={X.shape}, y.shape={y.shape}")

    # split
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_split, random_state=args.seed, stratify=y)
    except Exception:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_split, random_state=args.seed)

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model_obj = get_model(args.model_type, dim_in=X.shape[1], dim_hidden=args.hidden_dim, num_classes=len(STEP_BINS))

    if isinstance(model_obj, nn.Module):
        print("Training NN model:", args.model_type)
        best_val, trained_model = train_nn(model_obj, X_train, y_train, X_val, y_val, args)
        print("Best val acc:", best_val)
        torch.save({'model_type': args.model_type, 'state_dict': trained_model.state_dict(),
                    'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
                    'input_dim': X.shape[1], 'step_bins': STEP_LIST}, args.out)
    else:
        print("Training ML model:", args.model_type)
        model_obj.fit(X_train, y_train)
        val_acc, preds = evaluate_model(model_obj, X_val, y_val)
        print("Validation acc:", val_acc)
        joblib.dump({'model': model_obj, 'scaler': scaler, 'step_bins': STEP_LIST}, args.out)

    # final report
    if isinstance(model_obj, nn.Module):
        final_preds = evaluate_model(trained_model, X_val, y_val)[1]
    else:
        final_preds = model_obj.predict(X_val)
    print("Classification report (val):")
    print(classification_report(y_val, final_preds, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, final_preds))

    meta_out = Path(args.out).with_suffix('.meta.json')
    json.dump({'model_path': args.out, 'model_type': args.model_type, 'input_dim': X.shape[1], 'n_samples': int(X.shape[0])}, open(meta_out, 'w'), indent=2)
    print("Saved meta to", meta_out)

if __name__ == '__main__':
    main()
