import torch
import torch.nn as nn
import seaborn as sns
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, F1Score, ConfusionMatrix
from sklearn.metrics import roc_auc_score


class SupervisedModel(pl.LightningModule):
    def __init__(self, n_classes):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.train_epoch_outputs = []
        self.val_epoch_outputs = []
        self.test_epoch_outputs = []
        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.f1 = F1Score(task="multiclass", num_classes=n_classes, average='macro')
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=n_classes)
        self.best_val_f1 = 0
        self.best_val_loss = float('inf')
    
    def training_step(self, batch, batch_idx):
        y_hat, y_pred, y_true, loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, logger=False, on_step=True)
        output = {"loss":loss, "y_pred": y_pred, "y_true": y_true}
        self.train_epoch_outputs.append(output)
        sch = self.lr_scheduler
        self.log("lr", sch.get_last_lr()[0], prog_bar=False, logger=True, on_step=True)
        sch.step()
        return output
    
    def validation_step(self, batch, batch_idx):
        y_hat, y_pred, y_true, loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, logger=False, on_step=True)
        output = {"loss":loss, "y_pred": y_pred, "y_true": y_true}
        self.val_epoch_outputs.append(output)
        return output
    
    def test_step(self, batch, batch_idx):
        y_hat, y_pred, y_true, loss = self._shared_step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True, logger=False, on_step=True)
        output = {"loss":loss, "y_pred": y_pred, "y_true": y_true, "y_hat": y_hat}
        self.test_epoch_outputs.append(output)
        return output
    
    def _shared_step(self, batch, batch_idx):
        x, y_true = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y_true)
        y_pred = self.predict_step(batch, batch_idx)
        return y_hat, y_pred, y_true, loss
    
    def on_train_epoch_end(self):
        outputs = self.train_epoch_outputs
        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        acc = self.accuracy(y_pred, y_true)
        fig_ = self._plot_cm(y_pred, y_true)
        self.logger.experiment.add_scalar('train_loss', avg_loss, self.current_epoch)
        self.logger.experiment.add_figure("Confusion matrix train", fig_, self.current_epoch)
        self.log("train_acc", acc, prog_bar=True, logger=False)
        self.logger.experiment.add_scalar('train_acc', acc, self.current_epoch)
        self.train_epoch_outputs = []
    
    def on_validation_epoch_end(self):
        outputs = self.val_epoch_outputs
        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        acc = self.accuracy(y_pred, y_true)
        f1 = self.f1(y_pred, y_true)
        fig_ = self._plot_cm(y_pred, y_true)
        self.log("val_acc", acc, prog_bar=True, logger=False)
        self.log("val_f1", f1, prog_bar=True, logger=False)
        self.logger.experiment.add_scalar('val_loss', avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar('val_acc', acc, self.current_epoch)
        self.logger.experiment.add_scalar('val_f1', f1, self.current_epoch)
        self.logger.experiment.add_figure("Confusion matrix val", fig_, self.current_epoch)
        self.val_epoch_outputs = []
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            self.best_cm = fig_
    
    def on_test_epoch_end(self):
        outputs = self.test_epoch_outputs
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        y_hat = torch.cat([x['y_hat'] for x in outputs], dim=0)
        probs = F.softmax(y_hat, dim=1)
        acc = self.accuracy(y_pred, y_true)
        f1 = self.f1(y_pred, y_true)
        fig_ = self._plot_cm(y_pred, y_true)
        roc_auc_ovr = roc_auc_score(y_true, probs, multi_class='ovr')
        roc_auc_ovo = roc_auc_score(y_true, probs, multi_class='ovo')
        print(roc_auc_ovr, roc_auc_ovo)
        return acc, f1, fig_, roc_auc_ovr, roc_auc_ovo
    
    def _plot_cm(self, y_pred, y_true):
        cm = self.confmat(y_pred, y_true)
        cm = cm / cm.sum(axis=1)[:, None]    # row normalization
        fig, ax = plt.subplots(figsize=(10,10))
        fig_ = sns.heatmap(cm.to('cpu').numpy(), annot=True, cmap='Blues', ax=ax).get_figure()
        return fig_
    
    def on_train_end(self):
        # self.save_hyperparameters(ignore=["encoder", "datamodule"])
        self.logger.log_hyperparams(self.hparams, {"hp/val_f1": self.best_val_f1, "hp/val_loss": self.best_val_loss})