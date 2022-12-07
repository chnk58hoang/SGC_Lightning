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
from args import get_reddit_args
from utils import load_reddit_data, sgc_precompute, set_seed


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
        self.log("train_loss", loss_train)
        return loss_train

    def validation_step(self, val_batch, batch_idx):
        train_features, val_labels = val_batch
        output = self.forward(train_features)
        acc_val = accuracy(output, val_labels)
        f1_val = f1(output, val_labels)
        self.log("val_acc", acc_val)
        self.log("val_f1", f1_val)

    def test_step(self, test_batch, test_idx):
        test_feature, test_label = test_batch
        output = self.forward(test_feature)
        test_f1 = f1(output, test_label)
        print(f'test_f1:{test_f1}')

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        optimizer = optim.LBFGS(self.parameters(), lr=args.lr)
        return optimizer


def train_regression(model, train_features, train_labels, epochs):
    optimizer = optim.LBFGS(model.parameters(), lr=1)
    model.train()
    def closure():
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        return loss_train
    t = perf_counter()
    for epoch in range(epochs):
        loss_train = optimizer.step(closure)
    train_time = perf_counter() - t
    return model, train_time

def test_regression(model, test_features, test_labels):
    model.eval()
    return f1(model(test_features), test_labels)


# Args
args = get_reddit_args()

if args.implement == "pytorch-lightning":
    pl.seed_everything(args.seed)
else:
    pl.seed_everything(args.seed)
    set_seed(args.seed, args.cuda)

adj, train_adj, features, labels, idx_train, idx_val, idx_test = load_reddit_data(args.normalization, cuda=args.cuda)
print("Finished data loading.")

processed_features, precompute_time = sgc_precompute(features, adj, args.degree)
if args.inductive:
    train_features, _ = sgc_precompute(features[idx_train], train_adj, args.degree)
else:
    train_features = processed_features[idx_train]

test_features = processed_features[idx_test if args.test else idx_val]

if args.implement == "pytorch-lightning":
    train_dataset = CustomDataset(feature_tensor=train_features, label_tensor=labels[idx_train])
    train_loader = DataLoader(dataset=train_dataset, batch_size=140)
    test_dataset = CustomDataset(feature_tensor=test_features, label_tensor=labels[idx_test])
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    module = SGC_Lightning(
        nfeat=features.size(1), 
        nclass=labels.max().item() + 1, train_loader=train_loader,
        test_loader=test_loader
    )
    trainer = pl.Trainer(max_epochs=100)
    start_time = perf_counter()
    trainer.fit(module)
    train_time = perf_counter() - start_time
    print(f'Training_time:{train_time}')
    trainer.test(module, test_loader)
else:
    model = SGC(features.size(1), labels.max().item() + 1)
    model.cuda() if args.cuda else None
    model, train_time = train_regression(model, train_features, labels[idx_train], args.epochs)
    test_f1, _ = test_regression(model, test_features, labels[idx_test if args.test else idx_val])
    print("Total Time: {:.4f}s, {} F1: {:.4f}".format(
        train_time + precompute_time,
        "Test" if args.test else "Val",
        test_f1
    ))
