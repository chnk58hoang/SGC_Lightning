import argparse
from time import perf_counter

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CustomDataset
from metrics import accuracy, f1
from models import SGC
from models import get_model
from utils import load_reddit_data, sgc_precompute


class SGC_Lightning(pl.LightningModule):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """

    def __init__(self, nfeat, nclass, train_loader, test_loader):
        super(SGC_Lightning, self).__init__()
        self.model = SGC(nfeat, nclass)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def forward(self, x):
        return self.model(x)

    def cross_entropy_loss(self, output, train_labels):
        return F.cross_entropy(output, train_labels)

    def training_step(self, train_batch, batch_idx):
        train_features, train_labels = train_batch
        output = self.forward(train_features)
        loss_train = self.cross_entropy_loss(output, train_labels)
        self.log("acc_val", loss_train)
        return loss_train

    def validation_step(self, val_batch, batch_idx):
        train_features, val_labels = val_batch
        output = self.forward(train_features)
        acc_val = accuracy(output, val_labels)
        self.log("acc_val", acc_val)

    def test_step(self, test_batch, test_idx):
        test_feature, test_label = test_batch
        output = self.forward(test_feature)
        test_acc = f1(output, test_label)
        print(f'f1:{test_acc}')

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer


# Args
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.2,
                    help='Initial learning rate.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--inductive', action='store_true', default=False,
                    help='inductive training.')
parser.add_argument('--test', action='store_true', default=True,
                    help='inductive training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2,
                    help='Number of epochs to train.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                    choices=['NormLap', 'Lap', 'RWalkLap', 'FirstOrderGCN',
                             'AugNormAdj', 'NormAdj', 'RWalk', 'AugRWalk', 'NoNorm'],
                    help='Normalization method for the adjacency matrix.')
parser.add_argument('--degree', type=int, default=2,
                    help='degree of the approximation.')
parser.add_argument('--hidden', type=int, default=0,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

pl.seed_everything(args.seed)

adj, train_adj, features, labels, idx_train, idx_val, idx_test = load_reddit_data(args.normalization, cuda=args.cuda)
print("Finished data loading.")

processed_features, precompute_time = sgc_precompute(features, adj, args.degree)
if args.inductive:
    train_features, _ = sgc_precompute(features[idx_train], train_adj, args.degree)
else:
    train_features = processed_features[idx_train]

test_features = processed_features[idx_test if args.test else idx_val]

train_dataset = CustomDataset(feature_tensor=train_features, label_tensor=labels[idx_train])
train_loader = DataLoader(dataset=train_dataset, batch_size=140)
test_dataset = CustomDataset(feature_tensor=test_features, label_tensor=labels[idx_test])
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

module = SGC_Lightning(nfeat=features.size(1), nclass=labels.max().item() + 1, train_loader=train_loader,
                       test_loader=test_loader)
trainer = pl.Trainer(max_epochs=100)
start_time = perf_counter()
trainer.fit(module)
train_time = perf_counter() - start_time
trainer.test(module, test_loader)
print(f'Training_time:{train_time}')
