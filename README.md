# Hello

```bash
pip install -r requirements.txt
```

## CNN
To train CNN model over 100 epochs:
```
python scripts/train_supervised.py configs/cnn.py 100
```
Tensorboard and checkpoints will be saved in `out/CNN` directory.

## DeepConvLSTM
To train DeepConvLSTM model over 100 epochs:
```
python scripts/train_supervised.py configs/deep_conv_lstm.py 100
```
Tensorboard and checkpoints will be saved in `out/DeepConvLSTM` directory.


## Transformer Unsupervised Pretraining
To pretrain Transformer model v0 over 500 epochs:
```bash
python scripts/train_unsupervised.py configs/transformer_pretraining_v0.py 500
```
Tensorboard and checkpoints will be saved in `out/pretraining_v0` directory.


