from time import perf_counter
import torch
import torch.nn.functional as F
import torch.optim as optim

from sgc.metrics import f1


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
