import os
import torch
import pandas as pd
import pytorch_lightning as pl

from glob import glob
from torch.utils.data import Dataset, DataLoader


MEANS = [0.3622545097138867,
    0.16703852412033413,
    -0.45727826582246506,
    0.2999706455731254,
    0.5272100134926099,
    -0.1810271347463666,
    -0.057480609696549424,
    -0.030943019507639686,
    -0.283822232699083,
    -0.02026107906567315,
    0.01492879999064142,
    0.01495330873344323,
    -0.21252611071953795,
    0.039235152754384134,
    0.19742117656367683,
    0.3943193989239747,
    0.38630370076059206,
    0.1468860065240374,
    -0.47270594432444973]

STDS = [0.5518096884567281,
    0.5625705800248362,
    0.34451601357834516,
    1.3713393113348322,
    0.8923300804747172,
    0.5618254122320622,
    1.516266924633877,
    0.6945322965337732,
    1.1301687825701092,
    0.1527278996160204,
    0.22412035839464275,
    0.1920085419273123,
    0.31433677094671186,
    0.3437047109919375,
    0.4336991999338251,
    0.5948073655570362,
    0.5485874996986969,
    0.47617988251393734,
    0.27937132075615456]


class SupervisedDataset(Dataset):
    def __init__(self, data_dir, users, features, label_encoder, normalize=False):
        sequences = []
        labels = []
        for subpath, subdir, files in os.walk(data_dir):
            user = subpath.split("/")[-1]
            if user not in users:
                continue
            
            for file in glob(os.path.join(subpath, "*.csv")):
                df = pd.read_csv(file)
                # Normalize data:
                if normalize:
                    df[features] = (df[features] - MEANS) / STDS
                sequences.append(df[features])
                labels.append(df['label'][0])
                
        self.sequences = sequences
        self.labels = label_encoder.transform([label for label in labels])
        self.label_encoder = label_encoder
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx].to_numpy()
        label = self.labels[idx]
        return torch.tensor(sequence, dtype=torch.float), torch.tensor(label).long()

    def get_labels(self):          
        return self.labels  


class SupervisedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, train_users, val_users, test_users, features, label_encoder, batch_size, normalize, num_workers, pin_memory):
        super().__init__()
        self.data_dir = data_dir
        self.train_users = train_users
        self.val_users = val_users
        self.test_users = test_users
        self.features = features
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.normalize = normalize
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        self.train_dataset = SupervisedDataset(self.data_dir, self.train_users, self.features, self.label_encoder, self.normalize)
        self.val_dataset = SupervisedDataset(self.data_dir, self.val_users, self.features, self.label_encoder, self.normalize)
        self.test_dataset = SupervisedDataset(self.data_dir, self.test_users, self.features, self.label_encoder, self.normalize)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory
        )