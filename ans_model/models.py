import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

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

class DeepMLP(nn.Module):
    def __init__(self, dim_in, dim_hidden=512, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, num_classes)
        )

    def forward(self, x): return self.net(x)

class CNNClassifier(nn.Module):
    """
    CNN that accepts arbitrary 1D feature vectors by padding to the next
    perfect square and reshaping to [batch, 1, grid_size, grid_size].
    This keeps the rest of your pipeline simple and avoids changing run scripts.
    """
    def __init__(self, dim_in, dim_hidden=512, num_classes=3):
        super().__init__()
        # compute grid size as next integer >= sqrt(dim_in)
        gs = int(np.ceil(np.sqrt(dim_in)))
        self.grid_size = gs
        self.padded_dim = gs * gs  # total size after zero-padding

        # simple conv net (keeps model small)
        # use adaptive sizing so fc input dims work even when grid is small
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # grid //= 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # grid //= 2
            nn.Flatten(),
            # final linear will be created lazily in forward when we know spatial size
        )

        # keep a small head; we'll define the Linear dynamically in forward
        self._fc_hidden = dim_hidden
        self._num_classes = num_classes
        self._fc = None  # will be nn.Linear(in_features, dim_hidden) set on first forward

    def _make_fc(self, conv_out_dim):
        # conv_out_dim is flattened size after convs
        self._fc = nn.Sequential(
            nn.Linear(conv_out_dim, self._fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self._fc_hidden, self._num_classes)
        )

    def forward(self, x):
        # x: [batch, dim_in]
        b = x.shape[0]
        # pad to padded_dim
        if x.shape[1] < self.padded_dim:
            pad = x.new_zeros((b, self.padded_dim - x.shape[1]))
            x = torch.cat([x, pad], dim=1)
        elif x.shape[1] > self.padded_dim:
            # this shouldn't normally happen, but truncate to padded_dim
            x = x[:, :self.padded_dim]

        # reshape to grid
        x = x.view(b, 1, self.grid_size, self.grid_size)
        conv_out = self.conv_net(x)  # shape [b, C, H, W] flattened by Flatten
        conv_flat = conv_out.view(b, -1)

        # create fc if missing
        if self._fc is None:
            self._make_fc(conv_flat.shape[1])
            # move to same device as conv weights
            self._fc = self._fc.to(conv_flat.device)

        return self._fc(conv_flat)

class TinyTransformer(nn.Module):
    """
    Tiny transformer that accepts arbitrary 1D feature vectors by padding
    the input to (num_tokens * token_dim) where token_dim = ceil(dim_in / num_tokens).
    This allows using transformer even if dim_in isn't divisible by num_tokens.
    """
    def __init__(self, dim_in, dim_hidden=512, num_classes=3, num_heads=4):
        super().__init__()
        # For tiny dataset, use simpler architecture
        self.num_tokens = 4  # keep tokens small
        # token_dim is ceil so product >= dim_in
        self.token_dim = int(np.ceil(dim_in / self.num_tokens))
        self.padded_dim = self.token_dim * self.num_tokens  # total features after padding

        # Initial feature projection with dropout (project token_dim -> dim_hidden)
        self.input_proj = nn.Sequential(
            nn.Linear(self.token_dim, dim_hidden),
            nn.Dropout(0.2),
            nn.GELU()
        )

        # Smaller transformer with strong regularization
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_hidden,
                nhead=num_heads,
                dim_feedforward=dim_hidden * 2,
                dropout=0.2,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=1
        )

        # Output head with regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.Dropout(0.2),
            nn.Linear(dim_hidden, num_classes)
        )

    def forward(self, x):
        # x: [batch, dim_in]
        batch_size = x.shape[0]
        dim_in = x.shape[1]

        # pad up to padded_dim with zeros if needed
        if dim_in < self.padded_dim:
            pad = x.new_zeros((batch_size, self.padded_dim - dim_in))
            x = torch.cat([x, pad], dim=1)
        elif dim_in > self.padded_dim:
            # If input is larger than expected (unlikely), truncate
            x = x[:, :self.padded_dim]

        # reshape into tokens: [batch, num_tokens, token_dim]
        x = x.view(batch_size, self.num_tokens, self.token_dim)

        # project and apply transformer
        x = self.input_proj(x)            # [B, T, dim_hidden]
        x = self.transformer(x)           # [B, T, dim_hidden]

        # global average pool over tokens and classify
        x = x.mean(dim=1)
        return self.classifier(x)

class LogisticModel:
    def __init__(self, dim_in, num_classes=3):
        self.model = LogisticRegression(multi_class='multinomial', max_iter=1000)
        
    def to(self, device):
        return self
        
    def train(self):
        pass
        
    def eval(self):
        pass

    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)

class TreeModel:
    def __init__(self, dim_in, num_classes=3, model_type='xgboost'):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=50,   # Fewer trees for tiny dataset
                max_depth=2,       # Very shallow to prevent overfitting
                learning_rate=0.1, # Slightly faster learning
                objective='multi:softmax',
                num_class=num_classes,
                tree_method='hist',
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                enable_categorical=False,
                use_label_encoder=False
            )
        else:  # random forest
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=-1,
                class_weight='balanced_subsample'
            )
        
    def to(self, device):
        return self
        
    def train(self):
        pass
        
    def eval(self):
        pass

    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        # Convert labels to integers if needed
        y = y.astype(np.int32)
        
        # For tiny dataset, use all data for training
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        # Scale features using the same scaler
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def get_model(model_name, dim_in, dim_hidden=512, num_classes=3):
    """
    Factory that returns an instance of the requested model.
    Uses small wrapper lambdas so each model gets the arguments it expects.
    """
    factories = {
        'tiny_mlp': lambda d, h, n: TinyMLP(d, h, n),
        'deep_mlp': lambda d, h, n: DeepMLP(d, h, n),
        'cnn': lambda d, h, n: CNNClassifier(d, h, n),
        'transformer': lambda d, h, n: TinyTransformer(d, h, n),
        # LogisticModel expects (dim_in, num_classes)
        'logistic': lambda d, h, n: LogisticModel(d, n),
        # TreeModel wrapper: (dim_in, num_classes, model_type)
        'xgboost': lambda d, h, n: TreeModel(d, n, 'xgboost'),
        'random_forest': lambda d, h, n: TreeModel(d, n, 'random_forest')
    }

    if model_name not in factories:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(factories.keys())}")

    return factories[model_name](dim_in, dim_hidden, num_classes)
