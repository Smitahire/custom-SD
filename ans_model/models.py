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
    def __init__(self, dim_in, dim_hidden=512, num_classes=3):
        super().__init__()
        # Reshape input into 2D grid
        self.grid_size = int(np.sqrt(dim_in))
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (self.grid_size//4) * (self.grid_size//4), dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, num_classes)
        )

    def forward(self, x):
        # Reshape to [batch, 1, grid_size, grid_size]
        x = x.view(-1, 1, self.grid_size, self.grid_size)
        return self.conv_net(x)

class TinyTransformer(nn.Module):
    def __init__(self, dim_in, dim_hidden=512, num_classes=3, num_heads=4):
        super().__init__()
        # For tiny dataset, use simpler architecture
        self.num_tokens = 4  # Fewer tokens
        self.token_dim = dim_in // self.num_tokens
        
        # Initial feature projection with dropout
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
                dim_feedforward=dim_hidden*2,  # Smaller FFN
                dropout=0.2,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=1  # Single layer
        )
        
        # Output head with regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.Dropout(0.2),
            nn.Linear(dim_hidden, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Split input into tokens
        x = x.view(batch_size, self.num_tokens, self.token_dim)
        
        # Process tokens
        x = self.input_proj(x)
        
        # Apply transformer (no positional encoding for tiny dataset)
        x = self.transformer(x)
        
        # Global average pooling and classify
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
    models = {
        'tiny_mlp': TinyMLP,
        'deep_mlp': DeepMLP,
        'cnn': CNNClassifier,
        'transformer': TinyTransformer,
        'logistic': LogisticModel,
        'xgboost': lambda d, h, n: TreeModel(d, n, 'xgboost'),
        'random_forest': lambda d, h, n: TreeModel(d, n, 'random_forest')
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
        
    return models[model_name](dim_in, dim_hidden, num_classes)