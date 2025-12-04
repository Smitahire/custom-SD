#!/usr/bin/env python3
"""
evaluate_all_models.py

Usage:
  # from project root
  python evaluate_all_models.py --pred_dir predictions_by_model --labels data/labels_small.jsonl

If --labels is not provided or file missing, the script will derive labels from
data/results_features.jsonl using the "best_clip" rule (same as training labels).
Outputs:
 - prints per-model accuracy / classification report / confusion matrix
 - saves per-model detailed report to logs/metrics_{model}.txt
 - saves combined summary to logs/metrics_summary.txt
"""
import argparse, json, os
from pathlib import Path
from collections import OrderedDict
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_predictions(pred_file):
    preds = {}
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            pid = int(r['prompt_id'])
            pred_step = int(r.get('pred_steps', r.get('pred_step', None) or r.get('steps', None) or -1))
            preds[pid] = pred_step
    return preds

def load_labels_from_file(labels_file):
    labels = {}
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            pid = int(r.get('prompt_id', r.get('id', -1)))
            step = r.get('steps', r.get('step', None))
            if pid >= 0 and step is not None:
                labels[pid] = int(step)
    return labels

def derive_labels_from_features(features_file, step_bins=(25,35,45)):
    labels = {}
    with open(features_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            pid = int(r.get('prompt_id', -1))
            methods = r.get('methods', {})
            clips = []
            for s in step_bins:
                clips.append(float(methods.get(str(s), {}).get('clipscore', 0.0)))
            # argmax
            idx = int(np.argmax(clips))
            labels[pid] = int(step_bins[idx])
    return labels

def map_steps_to_class(steps_list):
    return {s:i for i,s in enumerate(steps_list)}

def evaluate_model(preds, labels, step_bins=(25,35,45)):
    # build aligned arrays for common prompt_ids
    common = sorted(set(preds.keys()) & set(labels.keys()))
    if not common:
        return None
    y_true = np.array([labels[i] for i in common])
    y_pred = np.array([preds[i] for i in common])
    # map to class indices (0..K-1)
    step2idx = map_steps_to_class(step_bins)
    y_true_idx = np.array([step2idx.get(int(x), -1) for x in y_true])
    y_pred_idx = np.array([step2idx.get(int(x), -1) for x in y_pred])
    # filter out any -1s
    valid = (y_true_idx >= 0) & (y_pred_idx >= 0)
    y_true_idx = y_true_idx[valid]; y_pred_idx = y_pred_idx[valid]
    acc = accuracy_score(y_true_idx, y_pred_idx)
    report = classification_report(y_true_idx, y_pred_idx, zero_division=0, output_dict=False)
    crep_dict = classification_report(y_true_idx, y_pred_idx, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    return {
        'n_common': int(valid.sum()),
        'accuracy': float(acc),
        'report_text': str(report),
        'report_dict': crep_dict,
        'confusion_matrix': cm.tolist()
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_dir', default='predictions_by_model')
    p.add_argument('--labels', default='data/labels_small.jsonl', help='external labels JSONL (optional)')
    p.add_argument('--features', default='data/results_features.jsonl', help='features JSONL used to derive labels if external labels missing')
    p.add_argument('--out_dir', default='logs')
    p.add_argument('--steps', nargs='*', type=int, default=[25,35,45])
    args = p.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load labels if available, else derive from features
    labels = {}
    if Path(args.labels).exists():
        print("Loading external labels from", args.labels)
        labels = load_labels_from_file(args.labels)
    else:
        print("External labels not found. Deriving labels from features file using best_clip:", args.features)
        labels = derive_labels_from_features(args.features, tuple(args.steps))
    if not labels:
        print("No labels found or derived. Aborting.")
        return

    # find prediction files
    pred_files = sorted(pred_dir.glob("*.jsonl"))
    if not pred_files:
        print("No prediction files found in", pred_dir)
        return

    summary = OrderedDict()
    for pf in pred_files:
        model_name = pf.stem.replace("predictions_","")
        print(f"\n=== Evaluating model: {model_name} (file: {pf}) ===")
        preds = load_predictions(pf)
        res = evaluate_model(preds, labels, tuple(args.steps))
        if res is None:
            print("No overlapping prompt_ids between predictions and labels.")
            continue
        # print nicely
        print(f"Samples compared: {res['n_common']}")
        print(f"Accuracy: {res['accuracy']:.4f}")
        print("Classification report:")
        print(res['report_text'])
        print("Confusion matrix:")
        print(np.array(res['confusion_matrix']))
        # save per-model file
        out_file = out_dir / f"metrics_{model_name}.txt"
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Pred file: {pf}\n")
            f.write(f"Samples compared: {res['n_common']}\n")
            f.write(f"Accuracy: {res['accuracy']:.4f}\n\n")
            f.write("Classification report (dict):\n")
            json.dump(res['report_dict'], f, indent=2)
            f.write("\n\nConfusion matrix:\n")
            json.dump(res['confusion_matrix'], f)
        summary[model_name] = res

    # combined summary
    summary_file = out_dir / "metrics_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        for mn, r in summary.items():
            f.write(f"=== {mn} ===\n")
            f.write(f"Samples compared: {r['n_common']}\n")
            f.write(f"Accuracy: {r['accuracy']:.4f}\n")
            f.write("Classification report (dict):\n")
            json.dump(r['report_dict'], f, indent=2)
            f.write("\nConfusion matrix:\n")
            json.dump(r['confusion_matrix'], f)
            f.write("\n\n")
    print("\nSaved per-model metrics to", out_dir, "and summary to", summary_file)

if __name__ == '__main__':
    main()
