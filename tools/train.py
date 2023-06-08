import pytorch_lightning as pl

from src.models.label_encoder import LabelEncoder
from src.datasets.unsupervised_dataset import UnsupervisedDataModule
from src.datasets.supervised_dataset import SupervisedDataModule
from src.models.transformer_encoder import TransformerEncoder
from src.models.transformer_classifier import TransformerClassifier
from src.models.cnn import CNN
from src.models.deep_conv_lstm import DeepConvLSTM
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def train_unsupervised(config, num_epochs=1):
    # Load data
    train_path = config.train_path
    val_path = config.val_path
    data_module = UnsupervisedDataModule(train_path,
                                        val_path, 
                                        config.batch_size, 
                                        config.unsupervised_dataset,
                                        config.num_workers,
                                        config.pin_memory)
    
    model = TransformerEncoder(datamodule=data_module, **config.encoder_cfgs)
    
    # Set callbacks + logger + trainer
    val_loss_ckpt_callback = ModelCheckpoint(
        dirpath = f'{config.out_dir}', 
        filename = "best_val_loss",
        save_top_k = 1, verbose=True, 
        monitor = "val_loss", mode="min"
    )

    logger = TensorBoardLogger(save_dir=config.out_dir, name='', version='logs')
    
    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        devices=1,
        logger=logger, 
        callbacks=[val_loss_ckpt_callback],
        accelerator=config.accelerator,
        enable_progress_bar=True
    )

    # Fit model
    trainer.fit(model, datamodule=data_module)


def train_supervised(config, num_epochs=1):
    # Load data
    train_path = config.train_path
    val_path = config.val_path
    
    label_encoder = LabelEncoder()
    data_module = SupervisedDataModule(train_path, 
                                    val_path, 
                                    config.features, 
                                    label_encoder, 
                                    config.batch_size,
                                    normalize=config.normalize, 
                                    num_workers=config.num_workers, 
                                    pin_memory=config.pin_memory)
    
    # Initialize model
    if config.model_name == 'Transformer':
        # Load transformer encoder
        if config.pretrained_model is not None:
            encoder = TransformerEncoder.load_from_checkpoint(
                config.pretrained_model,
                **config.model_cfgs['encoder_cfgs']
            )
        else:
            encoder = TransformerEncoder(**config.model_cfgs['encoder_cfgs'])
        
        # Load transformer classifier
        model = TransformerClassifier(
            datamodule=data_module,
            encoder=encoder,
            n_classes=len(list(data_module.label_encoder.decode_map.values())),
            **config.model_cfgs['classifier_cfgs']
        )
    
    elif config.model_name == 'CNN':
        model = CNN(n_features=len(config.features),
                    n_classes=len(list(data_module.label_encoder.decode_map.values())),
                    learning_rate=config.lr,
                    weight_decay=config.weight_decay, 
                    datamodule=data_module)
    
    elif config.model_name == 'DeepConvLSTM':
        model = DeepConvLSTM(n_features=len(config.features),
                    n_classes=len(list(data_module.label_encoder.decode_map.values())),
                    learning_rate=config.lr,
                    weight_decay=config.weight_decay, 
                    datamodule=data_module)

    # Set callbacks + logger + trainer
    f1_ckpt_callback = ModelCheckpoint(
        dirpath = f'{config.out_dir}', 
        filename = "best_f1",
        save_top_k = 1, verbose=True, 
        monitor = "f1", mode="max"
    )
    val_loss_ckpt_callback = ModelCheckpoint(
        dirpath = f'{config.out_dir}', 
        filename = "best_val_loss",
        save_top_k = 1, verbose=True, 
        monitor = "val_loss", mode="min"
    )

    logger = TensorBoardLogger(save_dir=config.out_dir, name='', version='logs')
    
    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        devices=1,
        logger=logger, 
        callbacks=[f1_ckpt_callback, val_loss_ckpt_callback],
        accelerator=config.accelerator,
        enable_progress_bar=True
    )

    # Fit model
    trainer.fit(model, datamodule=data_module)
