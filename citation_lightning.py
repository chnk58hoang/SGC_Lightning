import torch.optim as optim
from utils import load_citation, sgc_precompute
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter
import torch.nn.functional as F
import pytorch_lightning as pl
from models import SGC
from dataset import CustomDataset
from torch.utils.data import DataLoader


# Arguments
args = get_citation_args()

if args.tuned:
    if args.model == "SGC":
        with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
    else:
        raise NotImplemented

# setting random seeds
pl.seed_everything(args.seed)
adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda)

model = get_model(args.model, features.size(1), labels.max().item() + 1, args.hidden, args.dropout, args.cuda)

if args.model == "SGC": features, precompute_time = sgc_precompute(features, adj, args.degree)



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
        test_acc = accuracy(output, test_label)
        print(f'Test Acc:{test_acc}')

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer



if args.model == "SGC":
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


