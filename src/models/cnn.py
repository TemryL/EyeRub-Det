import torch
import torch.nn as nn
from .supervised_model import SupervisedModel


class CNN(SupervisedModel):
    def __init__(self, n_features, n_classes, learning_rate, weight_decay, datamodule):
        super().__init__(n_classes)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.datamodule = datamodule
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=5),
            nn.Dropout1d(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
            nn.Dropout1d(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
            nn.Dropout1d(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
            nn.Dropout1d(p=0.5),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(8576, n_classes)

        self.hparams.update({'learning_rate':learning_rate})
        self.hparams.update({'weight_decay':weight_decay})
    
    def forward(self, x):
        out = self.conv(x.permute(0, 2, 1))
        return self.fc1(out)
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_pred = torch.argmax(y_hat, dim=1)
        return y_pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, self.learning_rate, epochs=self.trainer.max_epochs, steps_per_epoch=len(self.datamodule.train_dataloader())
        )
        return optimizer