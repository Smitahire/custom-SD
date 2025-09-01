import argparse, json, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from utils.schedules import STEP_BINS, bin_index

class TinyMLP(nn.Module):
    def __init__(self, dim_in, dim_hidden=512, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, num_classes)
        )
    def forward(self, x): return self.net(x)

def main(args):
    # Load embeddings
    X = np.load(args.embeddings)   # shape [N, D]
    with open(args.ids, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # Load labels
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

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)#, stratify=y

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyMLP(dim_in=X.shape[1], dim_hidden=512, num_classes=len(STEP_BINS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    def to_t(t): return torch.tensor(t, dtype=torch.float32, device=device)

    best_val = -1.0
    for epoch in range(30):
        model.train()
        perm = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), args.batch_size):
            sl = perm[i:i+args.batch_size]
            xb = to_t(X_train[sl])
            yb = torch.tensor(y_train[sl], dtype=torch.long, device=device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            xv = to_t(X_val)
            yv = torch.tensor(y_val, dtype=torch.long, device=device)
            logits = model(xv)
            acc = (logits.argmax(dim=1) == yv).float().mean().item()
        if acc > best_val:
            best_val = acc
            torch.save({'model': model.state_dict()}, args.out)
        print(f"Epoch {epoch+1:02d} | val_acc={acc:.3f} | best={best_val:.3f}")

    print(f"Saved best model to {args.out}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--labels', required=True)
    p.add_argument('--embeddings', required=True)
    p.add_argument('--ids', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--batch_size', type=int, default=64)
    args = p.parse_args()
    main(args)
