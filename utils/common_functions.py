import random
from torch.utils.data import DataLoader
import torch
import sys
import math
import tqdm
import numpy as np


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_loader_seed(loader, seed):
    # for dataloader
    try:
        loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass