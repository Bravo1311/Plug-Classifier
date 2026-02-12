from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class SplitSpec:
    train: float
    val: float
    test: float
    seed: int

def make_indices(n: int, spec: SplitSpec) -> Tuple[List[int], List[int], List[int]]:
    assert abs((spec.train + spec.val + spec.test) - 1.0) < 1e-6 or spec.test == 0.0
    rng = np.random.default_rng(spec.seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * spec.train)
    n_val = int(n * spec.val)
    n_test = n - n_train - n_val

    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:n_train+n_val].tolist()
    test_idx = idx[n_train+n_val:].tolist() if n_test > 0 else []
    return train_idx, val_idx, test_idx
