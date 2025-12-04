#!/usr/bin/env python3
"""
predict_features.py
Run a saved model on a features JSONL and produce predictions JSONL.

Supports:
 - scikit/xgboost models saved as joblib (.joblib/.pkl) with {'model':..., 'scaler':..., 'step_bins':...}
 - torch NN checkpoints (.pth) saved by train_features.py (with keys 'state_dict', 'model_type', 'input_dim', 'scaler_mean', 'scaler_scale', 'step_bins')
Handles CNN lazy _fc initialization and falls back to strict=False when loading state dict fails.
"""
import argparse
import json
import numpy as np
from pathlib import Path
import joblib, torch, sys

proj_root = Path(__file__).resolve().parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from utils.schedules import STEP_BINS

# -------------------------
# Feature builder (unchanged)
# -------------------------
def load_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            out.append(json.loads(line))
    return out

def build_feature_vector(rec, steps_list):
    methods = rec.get("methods", {})
    feats = []
    for s in steps_list:
        feats.append(float(methods.get(str(s), {}).get("clipscore", 0.0)))
    for s in steps_list:
        feats.append(float(methods.get(str(s), {}).get("latency_s", 0.0)))
    for i in range(len(steps_list)):
        for j in range(i+1, len(steps_list)):
            a = steps_list[i]; b = steps_list[j]
            ma = methods.get(str(a), {})
            val = ma.get(f"lpips_vs_{b}", None)
            if val is None:
                mb = methods.get(str(b), {})
                val = mb.get(f"lpips_vs_{a}", 0.0)
            feats.append(float(val if val is not None else 0.0))
    clip_vals = [float(methods.get(str(s), {}).get("clipscore", 0.0)) for s in steps_list]
    lat_vals = [float(methods.get(str(s), {}).get("latency_s", 0.0)) for s in steps_list]
    clip_max = float(max(clip_vals)) if clip_vals else 0.0
    clip_argmax = float(np.argmax(clip_vals)) if clip_vals else 0.0
    clip_mean = float(np.mean(clip_vals)) if clip_vals else 0.0
    feats += [clip_max, clip_argmax, clip_mean, clip_max - clip_mean]
    feats += [float(np.min(lat_vals)) if lat_vals else 0.0, float(np.mean(lat_vals)) if lat_vals else 0.0]
    return np.array(feats, dtype=np.float32)

# -------------------------
# Robust torch checkpoint loader (handles lazy CNN _fc)
# -------------------------
def _init_cnn_fc_if_needed(model, input_dim):
    """
    If the CNNClassifier uses lazy _fc creation, run a dummy forward to initialize it.
    Uses model.padded_dim if present; falls back to input_dim.
    """
    try:
        if hasattr(model, "_fc") and model._fc is None:
            pad_dim = getattr(model, "padded_dim", None) or int(input_dim)
            model.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, pad_dim)
                _ = model(dummy)  # trigger lazy _make_fc in forward
            return True
    except Exception as e:
        print("Warning: failed to init CNN _fc via dummy forward:", e)
    return False

def load_checkpoint_model(model_path):
    """
    Load checkpoint (.pth) saved by train_features.py and return (model, scaler_callable, step_bins).
    scaler_callable: a function f(X_numpy) -> X_scaled (or None).
    """
    data = torch.load(str(model_path), map_location="cpu")
    model_type = data.get("model_type", "tiny_mlp")
    input_dim = int(data.get("input_dim", 0) or 0)
    step_bins = data.get("step_bins", None) or list(STEP_BINS)

    # lazy import of get_model
    from ans_model.models import get_model

    # Reconstruct model object
    model = get_model(model_type, dim_in=input_dim, dim_hidden=512, num_classes=len(step_bins))

    # If model has lazy fc (CNN) try to init it so state_dict keys match
    _init_cnn_fc_if_needed(model, input_dim)

    # Try to load the state dict; if strict fails, retry with strict=False
    try:
        model.load_state_dict(data["state_dict"])
    except Exception as e:
        print("Warning: strict load_state_dict failed, retrying with strict=False. Error:", e)
        try:
            model.load_state_dict(data["state_dict"], strict=False)
            print("Loaded checkpoint with strict=False.")
        except Exception as e2:
            print("Error: could not load state_dict even with strict=False:", e2)
            raise

    model.eval()

    # build scaler callable (if checkpoint saved mean/scale)
    scaler_callable = None
    if "scaler_mean" in data and "scaler_scale" in data:
        mean = np.array(data["scaler_mean"], dtype=np.float32)
        scale = np.array(data["scaler_scale"], dtype=np.float32)
        def scaler_fn(X):
            # X: numpy array (N, D)
            return (X - mean) / (scale + 1e-12)
        scaler_callable = scaler_fn

    return model, scaler_callable, step_bins

# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', default='predictions.jsonl')
    args = p.parse_args()

    recs = load_jsonl(args.features)
    steps_list = list(STEP_BINS)
    X = []
    ids = []
    for r in recs:
        X.append(build_feature_vector(r, steps_list))
        ids.append(int(r.get("prompt_id", -1)))
    X = np.stack(X, axis=0) if len(X) else np.zeros((0,0), dtype=np.float32)

    preds = None

    # joblib (ML models)
    if args.model.endswith('.joblib') or args.model.endswith('.pkl'):
        bag = joblib.load(args.model)
        # bag could be {'model':..., 'scaler':..., 'step_bins':...} or raw model
        scaler = bag.get('scaler', None) if isinstance(bag, dict) else None
        model = bag.get('model', bag) if isinstance(bag, dict) else bag
        # apply scaler if present
        if scaler is not None:
            # If scaler is sklearn StandardScaler object or dict
            if hasattr(scaler, 'transform'):
                Xs = scaler.transform(X)
            elif isinstance(scaler, dict) and 'mean' in scaler and 'scale' in scaler:
                Xs = (X - np.array(scaler['mean'])) / (np.array(scaler['scale']) + 1e-12)
            else:
                Xs = X
        else:
            Xs = X
        preds = model.predict(Xs)

    else:
        # torch checkpoint
        model_path = Path(args.model)
        model, scaler_fn, step_bins_saved = load_checkpoint_model(model_path)
        # If checkpoint has step_bins, use them instead of global STEP_BINS
        if step_bins_saved:
            steps_list = list(step_bins_saved)

        # apply scaler if present
        if scaler_fn is not None:
            Xs = scaler_fn(X)
        else:
            Xs = X

        # run model on CPU (checkpoint restored on CPU)
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(Xs, dtype=torch.float32)
            logits = model(tensor)
            preds = logits.argmax(dim=1).cpu().numpy()

    # write predictions
    out_lines = []
    for pid, cls in zip(ids, preds):
        step = int(steps_list[int(cls)])
        out_lines.append({'prompt_id': int(pid), 'pred_bin_index': int(cls), 'pred_steps': int(step)})
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in out_lines:
            f.write(json.dumps(r) + "\n")
    print("Wrote", args.out)

if __name__ == '__main__':
    main()
