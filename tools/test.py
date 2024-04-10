import pytorch_lightning as pl

from src.models.label_encoder import LabelEncoder
from src.datasets.supervised_dataset import SupervisedDataModule
from src.models.transformer_encoder import TransformerEncoder
from src.models.transformer_classifier import TransformerClassifier
from src.models.cnn import CNN
from src.models.deep_conv_lstm import DeepConvLSTM


def test(config, ckpt_path, test_users):
    # Load data
    train_users = []
    val_users = []
    
    label_encoder = LabelEncoder()
    data_module = SupervisedDataModule(config.data_dir,
                                    train_users,
                                    val_users,
                                    test_users,
                                    config.features,
                                    label_encoder,
                                    config.batch_size,
                                    config.normalize,
                                    config.num_workers,
                                    config.pin_memory)
    data_module.setup()
    
    # Initialize model
    if config.model_name == 'Transformer':
        # Load transformer encoder
        encoder = TransformerEncoder(**config.model_cfgs['encoder_cfgs'])
        
        # Load transformer classifier
        model = TransformerClassifier.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            datamodule=data_module,
            encoder=encoder,
            n_classes=len(list(data_module.label_encoder.decode_map.values())),
            **config.model_cfgs['classifier_cfgs']
        ).eval()
    
    elif config.model_name == 'CNN':
        model = CNN.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            n_features=len(config.features),
            n_classes=len(list(data_module.label_encoder.decode_map.values())),
            datamodule=data_module
        ).eval()
    
    elif config.model_name == 'DeepConvLSTM':
        model = DeepConvLSTM.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            n_features=len(config.features),
            n_classes=len(list(data_module.label_encoder.decode_map.values())),
            datamodule=data_module
        ).eval()
    
    trainer = pl.Trainer(enable_checkpointing=False, logger=False)
    trainer.test(model, dataloaders=data_module.test_dataloader())
    
    ax = model.cm_test.get_axes()[0]
    ax.set_xticklabels(label_encoder.decode_map.values())
    ax.set_yticklabels(label_encoder.decode_map.values())
    
    return model.acc_test, model.f1_test, model.cm_test, model.roc_auc_ovr_test, model.roc_auc_ovo