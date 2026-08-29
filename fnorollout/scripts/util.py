import random

import numpy as np
import torch


def set_seeds(seed: int = 42):
    """
    Set seeds for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(42)
    random.seed(seed)
