import torch
import numpy as np


# Warmup learning rate scheduler: increases learning rate to specified lr for the first
# warmup batch updates, then decreases it using the specified cosine-like functions, 
# reaching rate = 0 after max_iters iterations.
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters))
        if (epoch <= self.warmup) and (self.warmup != 0):
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor