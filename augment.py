import torch
import numpy as np

from torch import Tensor
from typing import Tuple

def spec_aug(batch, gamma_t: int = 2, eta_t: int = 60, gamma_f: int = 2, eta_f: int = 12,):
    _, feats, targets = batch
    #print(feats.shape)
    #exit(0)
    B, T, F = feats.shape
    for i in range(B):
        for _ in range(gamma_f):
            len_f = np.random.randint(0, eta_f)
            f0 = np.random.randint(0, max(F - len_f, 1))
            feats[i, :, f0:f0 + len_f] = 0

        for _ in range(gamma_t):
            len_t = np.random.randint(0, eta_t)
            t0 = np.random.randint(0, max(T - len_t, 1))
            feats[i, t0:t0 + len_t, :] = 0
    return feats, targets

def time_shifting(batch,  std: int = 10):
    _, feats, targets = batch
    B , _ , _  = feats.shape
    for i in range(B):
        ts = np.random.normal(0, std)
        feats[i,:,:] = torch.roll(feats[i,:,:], int(ts), dims=1)
    return feats, targets

def time_shifting_batch(batch ,std: int = 10):
    _, feats, targets = batch
    ts = np.random.normal(0, std)
    feats = torch.roll(feats, int(ts), dims=1)
    return feats, targets

def mixup(batch, length: int = 32, alpha: float = 0.2):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    _, feats, targets = batch
    idx1 = torch.randperm(len(feats), dtype=torch.long)[:length]
    idx2 = torch.randperm(len(feats), dtype=torch.long)[:length]
    feats1,feats2 = feats[idx1],feats[idx2]
    targets1,targets2 = targets[idx1],targets[idx2]

    mixed_feats = lam * feats1 + (1 - lam) * feats2
    mixed_targets = lam * targets1 + (1 - lam) * targets2

    return mixed_feats, mixed_targets

