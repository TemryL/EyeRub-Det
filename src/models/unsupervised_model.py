import torch
import pytorch_lightning as pl

from .masked_mse_loss import MaskedMSELoss
from ..utils.scheduler import CosineWarmupScheduler

class UnsupervisedModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.criterion = MaskedMSELoss()
        self.train_epoch_outputs = []
        self.val_epoch_outputs = []
    
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, logger=False, on_step=True)
        output = {"loss":loss}
        self.train_epoch_outputs.append(output)
        sch = self.lr_scheduler
        self.log("lr", sch.get_last_lr()[0], prog_bar=False, logger=True, on_step=True)
        sch.step()
        return output
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, logger=False, on_step=True)
        output = {"loss":loss}
        self.val_epoch_outputs.append(output)
        return output
    
    def _shared_step(self, batch, batch_idx):
        x_masked, x_true, mask = batch
        x_hat = self.forward(x_masked)
        loss = self.criterion(x_hat, x_true, mask)
        return loss
    
    def on_train_epoch_end(self):
        outputs = self.train_epoch_outputs
        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('train_loss', avg_loss, self.current_epoch)
        self.train_epoch_outputs = []
    
    def on_validation_epoch_end(self):
        outputs = self.val_epoch_outputs
        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('val_loss', avg_loss, self.current_epoch)
        self.val_epoch_outputs = []
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup, max_iters=self.trainer.max_epochs*len(self.datamodule.train_dataloader())
        )

        return optimizer