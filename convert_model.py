import torch
import coremltools as ct

from src.models.label_encoder import LabelEncoder
from src.models.transformer_encoder import TransformerEncoder
from src.models.transformer_classifier import TransformerClassifier
from src.models.supervised_model import SupervisedModel


class Predictor(SupervisedModel):
    def __init__(self, config, ckpt_path):
        super().__init__(n_classes=len(list(LabelEncoder().decode_map.values())))
        # Load transformer encoder
        encoder = TransformerEncoder(**config['encoder_cfgs'])
        
        # Load transformer classifier
        model = TransformerClassifier.load_from_checkpoint(
            ckpt_path,
            datamodule=None,
            encoder=encoder,
            n_classes=len(list(LabelEncoder().decode_map.values())),
            **config['classifier_cfgs']
        ).eval()
        
        self.model = model

    def forward(self, input):
        return torch.argmax(self.model(input.view(1, 150, 19)))


def convert(config, ckpt_path, out_path):
    predictor = Predictor(config, ckpt_path)

    # Trace the model with random data.
    example_input = torch.randn(150, 19)
    traced_model = predictor.to_torchscript(method="trace", example_inputs=example_input)
    out = traced_model(example_input)

    # Convert to Core ML neural network using the Unified Conversion API.
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape, name='input')],
        outputs=[ct.TensorType(name='output')]
    )

    # Save the converted model.
    mlmodel.save(out_path)


if __name__ == '__main__':
    
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
    
    convert(config, 
            ckpt_path='out/logsTransformerClassifier/reg/best_val_loss-v25.ckpt', 
            out_path='TransformerClassifier.mlmodel')