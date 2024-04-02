# Hello

```bash
pip install -r requirements.txt
```

## CNN
To train CNN model over 100 epochs:
```
python scripts/train_supervised.py configs/cnn.py 100
```
Tensorboard and checkpoints will be saved in `logs/CNN` directory.

## DeepConvLSTM
To train DeepConvLSTM model over 100 epochs:
```
python scripts/train_supervised.py configs/deep_conv_lstm.py 100
```
Tensorboard and checkpoints will be saved in `logs/DeepConvLSTM` directory.


## Transformer Unsupervised Pretraining
```

```


