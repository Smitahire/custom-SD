# utils/schedules.py
from typing import List
import numpy as np

# Simple mapping from a target step count to timesteps using linear spacing (DDIM-like)
def make_timesteps(num_inference_steps: int, total_train_timesteps: int = 1000) -> np.ndarray:
    return np.linspace(total_train_timesteps - 1, 0, num=num_inference_steps, dtype=np.int64)

# Step bins we classify over (keep small for 4 GB dev)
STEP_BINS = [25, 35, 45]

def nearest_step_bin(n: int) -> int:
    return min(STEP_BINS, key=lambda b: abs(b - n))

def bin_index(n: int) -> int:
    return STEP_BINS.index(nearest_step_bin(n))
