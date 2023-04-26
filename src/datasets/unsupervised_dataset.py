import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from ..utils.masking import noise_mask
from torch.utils.data import Dataset, DataLoader
from ..configs import NUM_WORKERS, PIN_MEMORY


class UnsupervisedDataset(Dataset):
    def __init__(self, path, features, window_size=150, step_size=150, normalize=True, mean_mask_length=3, masking_ratio=0.15,
                mode='separate', distribution='geometric', exclude_feats=None):
        df = pd.read_csv(path).sort_values(by="timestamp")
        df = df[features].dropna().reset_index(drop=True)
        if normalize:
            df = (df - df.mean())/df.std()
        self.data = df
        self.indexes = [i for i in range(0, len(df) - window_size + 1, step_size)]

        self.window_size = window_size
        self.step_size = step_size

        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.mode = mode
        self.distribution = distribution
        self.exclude_feats = exclude_feats
    
    def __len__(self):
        return len(self.indexes)
        
    def __getitem__(self, idx):
        idx = self.indexes[idx]  
        x_true = self.data.iloc[idx:idx+self.window_size].to_numpy()
        mask = noise_mask(x_true, self.masking_ratio, self.mean_mask_length, self.mode, self.distribution, self.exclude_feats) 
        x_masked = np.zeros_like(x_true)
        x_masked[mask] = x_true[mask] 
        mask = ~mask  # inverse logic: 0 now means ignore, 1 means predict
        return torch.tensor(x_masked, dtype=torch.float), torch.tensor(x_true, dtype=torch.float), torch.from_numpy(mask)


class UnsupervisedDataModule(pl.LightningDataModule):
    def __init__(self, train_path, test_path, batch_size, config):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.config = config

    def setup(self, stage=None):
        self.train_dataset = UnsupervisedDataset(self.train_path, **self.config)
        self.test_dataset = UnsupervisedDataset(self.test_path, **self.config)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )