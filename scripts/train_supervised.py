
import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import pytorch_lightning as pl
from tools.train import train_supervised
from importlib.machinery import SourceFileLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train superivsed model with given config file.'
    )
    parser.add_argument('config_file', help='path to model config file')
    parser.add_argument('train_users', help='list of train users as text file')
    parser.add_argument('val_users', help='list of val users as text file')
    parser.add_argument('num_epochs', help='number of epochs to train', type=int)
    parser.add_argument('out_dir', help='output directory')
    return parser.parse_args()


if __name__ == '__main__':
    # Set seed
    pl.seed_everything(42)
    
    args = parse_args()
    config = SourceFileLoader("config",args.config_file).load_module()
    
    with open(args.train_users, 'r') as f:
        train_users = f.read().splitlines()
        
    with open(args.val_users, 'r') as f:
        val_users = f.read().splitlines()
    
    train_supervised(
        config, 
        train_users=train_users,
        val_users=val_users,
        num_epochs=args.num_epochs,  
        out_dir=args.out_dir
    )