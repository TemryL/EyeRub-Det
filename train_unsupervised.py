import src.configs as configs
import pytorch_lightning as pl

from src.models.label_encoder import LabelEncoder
from src.datasets.unsupervised_dataset import UnsupervisedDataModule
from src.models.transformer_encoder import TransformerEncoder
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def train(config, num_epochs=1):
    # Load data
    train_path = configs.DATA_DIR + "unsupervised/train.csv"
    val_path = configs.DATA_DIR + "unsupervised/val.csv"
    data_module = UnsupervisedDataModule(train_path, val_path, config['batch_size'], config['unsupervised_dataset'])
    
    # Load transformer encoder
    if configs.PRETRAINED_MODEL is not None:
        model = TransformerEncoder.load_from_checkpoint(
            configs.PRETRAINED_MODEL,
            data_module = datamodule
            **config['encoder_cfgs']
        )
    else:
        model = TransformerEncoder(datamodule=data_module, **config['encoder_cfgs'])
    
    # Set callbacks + logger + trainer
    val_loss_ckpt_callback = ModelCheckpoint(
        dirpath = f'{configs.OUT_DIR}/logs{model.__class__.__name__}', 
        filename = "best_val_loss",
        save_top_k = 1, verbose=True, 
        monitor = "val_loss", mode="min"
    )

    logger = TensorBoardLogger(save_dir=configs.OUT_DIR, name=f'logs{model.__class__.__name__}')
    
    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        devices=1,
        logger=logger, 
        callbacks=[val_loss_ckpt_callback],
        accelerator=configs.ACCELERATOR,
        enable_progress_bar=False
    )

    # Fit model
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    # Set seed
    pl.seed_everything(42)
    
    lr = 5e-4
    config = dict(
        batch_size = 8,
        unsupervised_dataset = dict(
            features = configs.FEATURES, 
            window_size = 150, 
            step_size = 150, 
            normalize = False, 
            mean_mask_length = 3, 
            masking_ratio = 0.15,
            mode = 'separate', 
            distribution = 'geometric', 
            exclude_feats = None
        ),
        encoder_cfgs = dict(
            learning_rate = lr,
            feat_dim = 19, 
            max_len = 150, 
            d_model = 128, 
            num_heads = 16,
            num_layers = 2, 
            dim_feedforward = 512, 
            dropout = 0.1,
            pos_encoding = 'learnable', 
            activation = 'gelu',
            norm = 'BatchNorm', 
            freeze = False,
            warmup = 400,
            weight_decay = 1e-6,
        )
    )
    
    train(config, num_epochs=20)