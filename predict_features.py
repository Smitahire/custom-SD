#!/usr/bin/env python3
"""
predict_features.py
Run a saved model on a features JSONL and produce predictions JSONL.
"""
import argparse, json
import numpy as np
from pathlib import Path
import joblib, torch, sys

proj_root = Path(__file__).resolve().parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from utils.schedules import STEP_BINS

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
    X = np.stack(X, axis=0)

    # if joblib saved (non-NN)
    if args.model.endswith('.joblib') or args.model.endswith('.pkl'):
        bag = joblib.load(args.model)
        scaler = bag.get('scaler', None)
        if scaler is not None:
            Xs = scaler.transform(X)
        else:
            Xs = X
        model = bag.get('model', bag)
        preds = model.predict(Xs)
    else:
        data = torch.load(args.model, map_location='cpu')
        from ans_model.models import get_model
        mtype = data.get('model_type', 'tiny_mlp')
        input_dim = data.get('input_dim', X.shape[1])
        model = get_model(mtype, dim_in=input_dim, dim_hidden=512, num_classes=len(steps_list))
        model.load_state_dict(data['state_dict'])
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32)
            logits = model(tensor)
            preds = logits.argmax(dim=1).numpy()

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
