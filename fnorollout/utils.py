import random
import time

import numpy as np
import torch
from omegaconf import OmegaConf


def set_seeds(seed: int = 42):
    """
    Set seeds for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(42)
    random.seed(seed)


def register_utcnow_resolver() -> None:
    """
    Register a UTC OmegaConf resolver
    """
    if not OmegaConf.has_resolver("utcnow"):
        OmegaConf.register_new_resolver(
            "utcnow", lambda pattern: time.strftime(pattern, time.gmtime())
        )
