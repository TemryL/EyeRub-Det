import src.configs as configs
import pytorch_lightning as pl

from src.models.label_encoder import LabelEncoder
from src.datasets.supervised_dataset import SupervisedDataModule
from src.models.transformer_encoder import TransformerEncoder
from src.models.transformer_classifier import TransformerClassifier
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def train(config, num_epochs=1):
    # Load data
    train_path = configs.DATA_DIR + "supervised/train/"
    val_path = configs.DATA_DIR + "supervised/val/"
    
    label_encoder = LabelEncoder()
    data_module = SupervisedDataModule(train_path, val_path, configs.FEATURES, label_encoder, config['batch_size'], normalize=False)
    
    # Load transformer encoder
    if configs.PRETRAINED_MODEL is not None:
        encoder = TransformerEncoder.load_from_checkpoint(
            configs.PRETRAINED_MODEL,
            **config['encoder_cfgs']
        )
    else:
        encoder = TransformerEncoder(**config['encoder_cfgs'])
    
    # Load transformer classifier
    model = TransformerClassifier(
        datamodule=data_module,
        encoder=encoder,
        n_classes=len(list(data_module.label_encoder.decode_map.values())),
        **config['classifier_cfgs']
    )

    # Set callbacks + logger + trainer
    f1_ckpt_callback = ModelCheckpoint(
        dirpath = f'{configs.OUT_DIR}/logs{model.__class__.__name__}/reg', 
        filename = "best_f1",
        save_top_k = 1, verbose=True, 
        monitor = "f1", mode="max"
    )
    val_loss_ckpt_callback = ModelCheckpoint(
        dirpath = f'{configs.OUT_DIR}/logs{model.__class__.__name__}/reg', 
        filename = "best_val_loss",
        save_top_k = 1, verbose=True, 
        monitor = "val_loss", mode="min"
    )

    logger = TensorBoardLogger(save_dir=configs.OUT_DIR, name=f'logs{model.__class__.__name__}/reg')
    
    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        devices=1,
        logger=logger, 
        callbacks=[f1_ckpt_callback, val_loss_ckpt_callback],
        accelerator=configs.ACCELERATOR,
        enable_progress_bar=False
    )

    # Fit model
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    for weight_decay in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        for dropout in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            
            # Set seed
            pl.seed_everything(42)
            
            lr = 5e-4
            config = dict(
                batch_size = 16,
                encoder_cfgs = dict(
                    learning_rate = lr,
                    feat_dim = 19, 
                    max_len = 150, 
                    d_model = 128, 
                    num_heads = 4,
                    num_layers = 4, 
                    dim_feedforward = 512, 
                    dropout = dropout,
                    pos_encoding = 'learnable', 
                    activation = 'gelu',
                    norm = 'BatchNorm', 
                    freeze = False
                ),
                classifier_cfgs = dict(
                    learning_rate = lr,
                    warmup = 200,
                    weight_decay = weight_decay
                )
            )
            
            train(config, num_epochs=20)