import argparse
import yaml
import pickle as pkl
from time import perf_counter

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from sgc.dataset import CustomDataset
from sgc.models import SGC, SGC_Lightning
from sgc.utils import load_citation, sgc_precompute, set_seed, load_reddit_data
from sgc import citation, reddit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to yaml config file.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--lightning', action='store_true', default=False,
                        help='execute with PyTorch Lightning version.')

    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def read_yaml(filepath):
    with open(filepath, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def run_citation(config, cuda=False, lightning=True):
    if config['tuned']:
        with open("sgc-tuned/{}.txt".format(config['dataset']), 'rb') as f:
            config['weight_decay'] = pkl.load(f)['weight_decay']
            print(f"Using tuned weight decay: {config['weight_decay']}")

    adj, features, labels, idx_train, idx_val, idx_test = load_citation(config['dataset'], cuda)
    model = SGC(nfeat=features.size(1), nclass=labels.max().item() + 1)
    features, precompute_time = sgc_precompute(features, adj, config['degree'], config['alpha'])

    if lightning:

        train_dataset = CustomDataset(feature_tensor=features[idx_train], label_tensor=labels[idx_train])
        train_loader = DataLoader(dataset=train_dataset, batch_size=140)

        test_dataset = CustomDataset(feature_tensor=features[idx_test], label_tensor=labels[idx_test])
        test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

        module = SGC_Lightning(
            nfeat=features.size(1), 
            nclass=labels.max().item() + 1, 
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=config['optimizer'],
            lr=config['lr'],
        )
        trainer = pl.Trainer(max_epochs=100)
        start_time = perf_counter()
        trainer.fit(module)
        train_time = perf_counter() - start_time
        trainer.test(module, test_loader)
        print(f'Training_time:{train_time}')
    else:
        model, acc_val, train_time = citation.train_regression(
            model, 
            features[idx_train], 
            labels[idx_train], 
            features[idx_val],
            labels[idx_val],
            config['epochs'], 
            config['optimizer'],
            config['weight_decay'], 
            config['lr']
        )
        acc_test = citation.test_regression(model, features[idx_test], labels[idx_test])
        print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
        print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time,
                                                                                        precompute_time + train_time))


def run_reddit(config, cuda=False, lightning=True):
    adj, train_adj, features, labels, idx_train, idx_val, idx_test = load_reddit_data(cuda=cuda)
    print("Finished data loading.")

    processed_features, precompute_time = sgc_precompute(features, adj, config['degree'], config['alpha'])
    if config['inductive']:
        train_features, _ = sgc_precompute(features[idx_train], train_adj, config['degree'], config['alpha'])
    else:
        train_features = processed_features[idx_train]

    test_features = processed_features[idx_test if config['test'] else idx_val]

    if lightning:
        train_dataset = CustomDataset(feature_tensor=train_features, label_tensor=labels[idx_train])
        train_loader = DataLoader(dataset=train_dataset, batch_size=140)
        test_dataset = CustomDataset(feature_tensor=test_features, label_tensor=labels[idx_test])
        test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

        module = SGC_Lightning(
            nfeat=features.size(1), 
            nclass=labels.max().item() + 1, 
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=config['optimizer'],
            lr=config['lr'],
        )
        trainer = pl.Trainer(max_epochs=100)
        start_time = perf_counter()
        trainer.fit(module)
        train_time = perf_counter() - start_time
        print(f'Training_time:{train_time}')
        trainer.test(module, test_loader)
    else:
        model = SGC(features.size(1), labels.max().item() + 1)
        model.cuda() if cuda else None
        model, train_time = reddit.train_regression(
            model, 
            train_features, 
            labels[idx_train], 
            config['epochs'],
            config['optimizer'],
            config['weight_decay'],
            config['lr'],
        )
        test_f1, _ = reddit.test_regression(model, test_features, labels[idx_test if config['test'] else idx_val])
        print("Total Time: {:.4f}s, {} F1: {:.4f}".format(
            train_time + precompute_time,
            "Test" if config['test'] else "Val",
            test_f1
        ))


if __name__ == "__main__":
    args = parse_args()
    config = read_yaml(args.config)

    if args.lightning:
        pl.seed_everything(args.seed)
    else:
        set_seed(args.seed, args.cuda)

    if config['dataset'] == 'reddit':
        run_reddit(config, args.cuda, args.lightning)
    else:
        run_citation(config, args.cuda, args.lightning)
