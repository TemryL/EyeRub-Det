
import sys
sys.path.append('.')
sys.path.append('..')

import os
import json
import argparse
import pytorch_lightning as pl
from tools.test import test
from importlib.machinery import SourceFileLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test trained model.'
    )
    parser.add_argument('config_file', help='path to model config file')
    parser.add_argument('ckpt_path', help='path to model checkpoints')
    parser.add_argument('test_users', help='list of test users as text file')
    parser.add_argument('out_dir', help='output directory to write performance metrics')
    return parser.parse_args()


if __name__ == '__main__':
    # Set seed
    pl.seed_everything(42)
    
    args = parse_args()
    config = SourceFileLoader("config", args.config_file).load_module()
    
    with open(args.test_users, 'r') as f:
        test_users = f.read().splitlines()
    
    acc_test, f1_test, cm_test, roc_auc_ovr, roc_auc_ovo, roc_auc = test(config, ckpt_path=args.ckpt_path, test_users=test_users)
    results = {'f1': f1_test, 'auc_ovr': roc_auc_ovr, 'auc_ovo': roc_auc_ovo, 'auc': roc_auc}
    
    # Ensure the output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save results in json formatted file in the output directory
    results_file_path = os.path.join(args.out_dir, 'results_test.json')
    with open(results_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    # Save confusion matrix figure in the output directory
    cm_fig_path = os.path.join(args.out_dir, 'cm_test.png')
    cm_test.savefig(cm_fig_path)
    