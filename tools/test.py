import torch
import torch.nn.functional as F
import src.configs as configs
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.label_encoder import LabelEncoder
from src.datasets.supervised_dataset import SupervisedDataset
from src.models.transformer_encoder import TransformerEncoder
from src.models.transformer_classifier import TransformerClassifier


def test(config, ckpt_path):
    # Load data
    #val_path = configs.DATA_DIR + "supervised/use_case/user62/test"
    val_path = configs.DATA_DIR + "supervised/test"
    
    label_encoder = LabelEncoder()
    val_dataset = SupervisedDataset(val_path, configs.FEATURES, label_encoder, normalize=False)
    
    # Load transformer encoder
    encoder = TransformerEncoder(**config['encoder_cfgs'])
    
    # Load transformer classifier
    model = TransformerClassifier.load_from_checkpoint(
        ckpt_path,
        datamodule=None,
        encoder=encoder,
        n_classes=len(list(val_dataset.label_encoder.decode_map.values())),
        **config['classifier_cfgs']
    ).eval()

    # Make inferences
    preds = []
    targets = []
    for sequence, label in val_dataset:
        y_hat = model(sequence.view(1, 150, 19))
        y_hat = F.softmax(y_hat, dim=1)
        
        # Decision
        y_pred = torch.argmax(y_hat, dim=1).item()
        preds.append(y_pred)
        targets.append(label)
    
    # Compute performances
    preds = torch.Tensor(preds)
    targets = torch.Tensor(targets)
    cm = model.confmat(preds, targets)
    cm = cm / cm.sum(axis=1)[:, None]    # row normalization

    print(f"Acc: {model.accuracy(preds, targets)}")
    print(f"F1: {model.f1(preds, targets)}")
    
    fig, ax = plt.subplots(figsize=(8,8))
    fig_ = sns.heatmap(cm.to('cpu').numpy(), annot=True, cmap='Blues', ax=ax, 
                    xticklabels=label_encoder.decode_map.values(), 
                    yticklabels=label_encoder.decode_map.values())
    plt.xticks(rotation=0) 
    plt.yticks(rotation=0) 
    plt.show()


if __name__ == '__main__':
    # Set seed
    pl.seed_everything(42)
    
    lr = None
    config = dict(
        batch_size = None,
        encoder_cfgs = dict(
            learning_rate = lr,
            feat_dim = 19, 
            max_len = 150, 
            d_model = 128, 
            num_heads = 4,
            num_layers = 4, 
            dim_feedforward = 512, 
            dropout = 0.1,
            pos_encoding = 'learnable', 
            activation = 'gelu',
            norm = 'BatchNorm', 
            freeze = False
        ),
        classifier_cfgs = dict(
            learning_rate = lr,
            warmup = None,
            weight_decay = 1e-6
        )
    )
    
    test(config, ckpt_path='out/logsTransformerClassifier/reg/best_val_loss-v25.ckpt')
    # test(config, ckpt_path='out/logsFineTunedTransformerClassifier/best_val_loss-v3.ckpt')