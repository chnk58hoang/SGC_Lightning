import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from sgc import metrics


class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)


class SGC_Lightning(pl.LightningModule):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """

    def __init__(self, nfeat, nclass, train_loader, test_loader, optimizer="Adam", lr=0.01, weight_decay=5e-6):
        super(SGC_Lightning, self).__init__()
        self.model = SGC(nfeat, nclass)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = self.get_optimizer(optimizer, lr, weight_decay)

    def forward(self, x):
        return self.model(x)

    def get_optimizer(self, optim, lr, weight_decay=5e-6):
        if optim.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim.lower() == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim.lower() == "lbfgs":
            optimizer = torch.optim.LBFGS(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer {optim} is not supported")
        return optimizer

    def cross_entropy_loss(self, output, train_labels):
        return F.cross_entropy(output, train_labels)

    def training_step(self, train_batch, batch_idx):
        train_features, train_labels = train_batch
        output = self.forward(train_features)
        loss_train = self.cross_entropy_loss(output, train_labels)
        self.log("loss_train", loss_train)
        return loss_train

    def validation_step(self, val_batch, batch_idx):
        train_features, val_labels = val_batch
        output = self.forward(train_features)
        acc_val = metrics.accuracy(output, val_labels)
        self.log("acc_val", acc_val)

    def test_step(self, test_batch, test_idx):
        test_feature, test_label = test_batch
        output = self.forward(test_feature)
        test_acc =  metrics.accuracy(output, test_label)
        test_f1 = metrics.f1(output, test_label)
        print(f'\nTest Acc: {test_acc}')
        print(f'Test F1: {test_f1}')

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        return self.optimizer
