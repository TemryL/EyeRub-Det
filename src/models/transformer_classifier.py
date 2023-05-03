import torch
import torch.nn as nn
from .supervised_model import SupervisedModel
from .global_temporal_attention import GlobalTemporalAttention
from ..utils.scheduler import CosineWarmupScheduler


class TransformerClassifier(SupervisedModel):
    def __init__(self, encoder, n_classes, learning_rate, warmup, weight_decay, datamodule, freeze=False):
        super().__init__(n_classes)
        self.learning_rate = learning_rate
        self.warmup = warmup
        self.weight_decay = weight_decay
        self.datamodule = datamodule

        self.encoder = encoder
        self.global_attn = GlobalTemporalAttention(encoder.d_model, encoder.dropout)
        self.classifier = nn.Linear(encoder.d_model, n_classes)

        self.hparams.update(encoder.hparams)
        self.hparams.update({'learning_rate':learning_rate})
        self.hparams.update({'warmup':warmup})
        self.hparams.update({'weight_decay':weight_decay})
    
    def forward(self, x):
        output = self.encoder.encode(x)
        output = self.global_attn(output)
        return self.classifier(output)
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_pred = torch.argmax(y_hat, dim=1)
        return y_pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup, max_iters=self.trainer.max_epochs*len(self.datamodule.train_dataloader())
        )

        return optimizer