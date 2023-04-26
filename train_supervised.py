import argparse
import src.config as config
import pytorch_lightning as pl

from src.models.label_encoder import LabelEncoder
from src.datasets.supervised_dataset import SupervisedDataModule
from src.models.transformer_encoder import TransformerEncoder
from src.models.transformer_classifier import TransformerClassifier
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def main(out_dir):
    # Set seed
    pl.seed_everything(42)
    
    # Load data
    train_path = config.DATA_DIR + "supervised/train/"
    test_path = config.DATA_DIR + "supervised/val/"
    
    label_encoder = LabelEncoder()
    CLASSES = list(label_encoder.decode_map.values())
    data_module = SupervisedDataModule(train_path, test_path, config.FEATURES, label_encoder, config.BATCH_SIZE, normalize=False)
    
    # Load transformer encoder
    if config.PRETRAINED_MODEL is not None:
        encoder = TransformerEncoder.load_from_checkpoint(
            config.PRETRAINED_MODEL,
            **config.ENCODER_CFGS
        )
    else:
        encoder = TransformerEncoder(**config.ENCODER_CFGS)
    
    # Load transformer classifier
    model = TransformerClassifier(
        encoder=encoder,
        n_features=len(config.FEATURES), 
        n_classes=len(CLASSES),
        datamodule=data_module,
        **config.MODEL_CFGS
    )

    # Set callbacks + logger + trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath = out_dir + "/logs" + model.__class__.__name__, 
        filename = "best-checkpoint",
        save_top_k = 1, verbose=True, 
        monitor = "f1", mode="max")

    logger = TensorBoardLogger(save_dir=out_dir, name='logs' + model.__class__.__name__)

    trainer = pl.Trainer(
        max_epochs=config.NUM_EPOCHS, 
        devices=1,
        logger=logger, 
        callbacks=[checkpoint_callback],
        accelerator=config.ACCELERATOR
    )

    # Fit model
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train supervised model using either pre-trained model or from scratch."
    )
    parser.add_argument(
        "out_dir",
        metavar="out_dir",
        type=str,
        help="The path to write logs.",
    )
    
    args = parser.parse_args()
    
    main(args.out_dir)