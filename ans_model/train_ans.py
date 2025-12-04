import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# ✅ Dynamically add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.schedules import STEP_BINS, bin_index
from models import get_model
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X, y, device='cpu'):
    if isinstance(model, (nn.Module)):
        model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            y_tensor = torch.tensor(y, dtype=torch.long, device=device)
            logits = model(x_tensor)
            predictions = logits.argmax(dim=1).cpu().numpy()
    else:
        predictions = model.predict(X)
    return (predictions == y).mean()

def train_model(model, X_train, y_train, X_val, y_val, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if isinstance(model, nn.Module):
        # Neural network training
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        crit = nn.CrossEntropyLoss()

        def to_t(t): return torch.tensor(t, dtype=torch.float32, device=device)

        best_val = -1.0
        for epoch in range(args.epochs):
            model.train()
            perm = np.random.permutation(len(X_train))
            total_loss = 0

            for i in range(0, len(X_train), args.batch_size):
                sl = perm[i:i+args.batch_size]
                xb = to_t(X_train[sl])
                yb = torch.tensor(y_train[sl], dtype=torch.long, device=device)
                
                opt.zero_grad()
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item()

            # Validation
            acc = evaluate_model(model, X_val, y_val, device)
            if acc > best_val:
                best_val = acc
                torch.save({'model': model.state_dict()}, args.out)
            print(f"Epoch {epoch+1:02d} | val_acc={acc:.3f} | best={best_val:.3f}")

        # Final detailed report on validation set (if requested)
        if getattr(args, 'debug', False):
            model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
                logits = model(x_tensor)
                preds = logits.argmax(dim=1).cpu().numpy()
            print("Validation classification report (NN):")
            print(classification_report(y_val, preds, zero_division=0))
            print("Confusion matrix:")
            print(confusion_matrix(y_val, preds))

        print(f"Saved best model to {args.out}")
        return best_val

    else:
        # Traditional ML model (XGBoost, RandomForest, etc.)
        model.fit(X_train, y_train)
        acc = evaluate_model(model, X_val, y_val)
        print(f"Validation accuracy: {acc:.3f}")

        import joblib
        # Detailed report for traditional ML
        if getattr(args, 'debug', False):
            preds = model.predict(X_val)
            print("Validation classification report (ML):")
            print(classification_report(y_val, preds, zero_division=0))
            print("Confusion matrix:")
            print(confusion_matrix(y_val, preds))

        joblib.dump(model, args.out)
        print(f"Saved best model to {args.out}")
        return acc

def main(args):
    # Load embeddings
    X = np.load(args.embeddings)   # shape [N, D]
    with open(args.ids, 'r', encoding='utf-8') as f:
        meta = json.load(f)

 
    ys = []
    with open(args.labels, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            ys.append((rec['prompt_id'], bin_index(rec['steps'])))
    ys = dict(ys)

    # Align X and y
    idxs = [i for i in range(len(meta['prompts'])) if i in ys]
    X = X[idxs]
    y = np.array([ys[i] for i in idxs], dtype=np.int64)

    # Try to stratify the split to preserve class balance when possible
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_split, random_state=args.seed, stratify=y)
    except Exception:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_split, random_state=args.seed)

    # Initialize model
    model = get_model(
        args.model_type,
        dim_in=X.shape[1],
        dim_hidden=args.hidden_dim,
        num_classes=len(STEP_BINS)
    )

    # Train model
    best_val_acc = train_model(model, X_train, y_train, X_val, y_val, args)
    
    # Save training metadata
    meta_path = args.out.rsplit('.', 1)[0] + '_meta.json'
    with open(meta_path, 'w') as f:
        json.dump({
            'model_type': args.model_type,
            'val_accuracy': best_val_acc,
            'hidden_dim': args.hidden_dim,
            'input_dim': X.shape[1],
            'num_classes': len(STEP_BINS),
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }, f, indent=2)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--labels', required=True)
    p.add_argument('--embeddings', required=True)
    p.add_argument('--ids', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model_type', default='tiny_mlp',
                  choices=['tiny_mlp', 'deep_mlp', 'cnn', 'transformer',
                          'logistic', 'xgboost', 'random_forest'])
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--learning_rate', type=float, default=2e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--val_split', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--debug', action='store_true', help='Print classification reports and debug info')
    args = p.parse_args()
    main(args)
