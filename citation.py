import pickle as pkl
from time import perf_counter

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from args import get_citation_args
from dataset import CustomDataset
from metrics import accuracy
from models import SGC
from models import get_model
from utils import load_citation, sgc_precompute, set_seed

# Arguments
args = get_citation_args()

if args.tuned:
    with open("{}-tuning/{}.txt".format("SGC", args.dataset), 'rb') as f:
        args.weight_decay = pkl.load(f)['weight_decay']
        print("using tuned weight decay: {}".format(args.weight_decay))

if args.implement == "pytorch-lightning":
    pl.seed_everything(args.seed)
else:
    pl.seed_everything(args.seed)
    set_seed(args.seed, args.cuda)

adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda)
model = get_model("SGC", features.size(1), labels.max().item() + 1, args.hidden, args.dropout, args.cuda)
features, precompute_time = sgc_precompute(features, adj, args.degree)
print("{:.4f}s".format(precompute_time))


def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter() - t
    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)
    return model, acc_val, train_time


def test_regression(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features), test_labels)


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
        self.log("loss_train", loss_train)
        return loss_train

    def validation_step(self, val_batch, batch_idx):
        train_features, val_labels = val_batch
        output = self.forward(train_features)
        acc_val = accuracy(output, val_labels)
        self.log("acc_val", acc_val)

    def test_step(self, test_batch, test_idx):
        test_feature, test_label = test_batch
        output = self.forward(test_feature)
        test_acc = accuracy(output, test_label)
        print(f'Test Acc:{test_acc}')

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer


if args.implement == "pytorch-lightning":

    train_dataset = CustomDataset(feature_tensor=features[idx_train], label_tensor=labels[idx_train])
    train_loader = DataLoader(dataset=train_dataset, batch_size=140)

    test_dataset = CustomDataset(feature_tensor=features[idx_test], label_tensor=labels[idx_test])
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    module = SGC_Lightning(nfeat=features.size(1), nclass=labels.max().item() + 1, train_loader=train_loader,
                            test_loader=test_loader)
    trainer = pl.Trainer(max_epochs=100)
    start_time = perf_counter()
    trainer.fit(module)
    train_time = perf_counter() - start_time
    trainer.test(module, test_loader)
    print(f'Training_time:{train_time}')
else:
    model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val],
                                                    labels[idx_val],
                                                    args.epochs, args.weight_decay, args.lr, args.dropout)
    acc_test = test_regression(model, features[idx_test], labels[idx_test])
    print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
    print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time,
                                                                                    precompute_time + train_time))
