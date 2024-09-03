# Eye-Rubbing Detection Using a Smartwatch

This repository contains the code for the paper:

> **Eye-Rubbing Detection Using a Smartwatch: A feasibility study demonstrated high accuracy with machine learning methods based on Transformer**, TVST Sep. 2024
> - [Link to the paper](https://tvst.arvojournals.org/article.aspx?articleid=2800751)

Published in [Translational Vision Science & Technology](https://tvst.arvojournals.org/) (TVST)

This project implements various machine learning models for detecting eye-rubbing and other hand-face interactions using smartwatch sensor data.

## Setup Instructions

```bash
git clone https://github.com/TemryL/EyeRub-Det.git
cd EyeRub-Det
```

Install requirements (Python 3.10 recommended):

```bash
pip install -r requirements.txt
```

## Get Data

```bash
git clone https://huggingface.co/datasets/TemryL/hand-face-interactions data/
```

## Training Models

For all models, tensorboard logs and checkpoints will be saved in their respective output directories.

### CNN

Train the CNN model over 100 epochs:

```bash
python scripts/train_supervised.py \
configs/cnn.py \
splits/users_train.txt \
splits/users_val.txt \
100 \
out/CNN
```

### DeepConvLSTM

Train the DeepConvLSTM model over 100 epochs:

```bash
python scripts/train_supervised.py \
configs/deep_conv_lstm.py \
splits/users_train.txt \
splits/users_val.txt \
100 \
out/DeepConvLSTM
```

### Transformer

#### Unsupervised Pretraining

Pretrain Transformer model v0 over 500 epochs:

```bash
python scripts/train_unsupervised.py \
configs/transformer_pretraining_v0.py \
500 \
out/pretraining/v0/
```

#### Supervised Training from Scratch

Train the Transformer model from scratch:

```bash
python scripts/train_supervised.py \
configs/transformer_v0_scratch.py \
splits/users_train.txt \
splits/users_val.txt \
20 \
out/v0_scratch
```

#### Supervised Training from Pre-trained

Set the path to the pretrained checkpoint in the config file (e.g., `configs/transformer_v0_pretrained.py`) and run:

```bash
python scripts/train_supervised.py \
configs/transformer_v0_pretrained.py \
splits/users_train.txt \
splits/users_val.txt \
20 \
out/v0_pretrained
```

## Visualizing Results

To plot tensorboard logs for any model:

```bash
tensorboard --logdir=out/<model_directory> --bind_all
```

Replace `<model_directory>` with the appropriate output directory (e.g., CNN, DeepConvLSTM, pretraining/v0, v0_scratch, v0_pretrained).

## Additional Information

For more details about the dataset and the research behind this project, please refer to the paper linked above and the dataset description on [Hugging Face](https://huggingface.co/datasets/TemryL/hand-face-interactions).

## Citation

If you use this code or the associated dataset in your research, please cite our paper:

```bibtex
@article{
}
```
