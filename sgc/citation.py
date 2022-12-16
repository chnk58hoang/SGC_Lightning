from time import perf_counter

import torch
import torch.nn.functional as F

from sgc.metrics import accuracy
from sgc.utils import get_optimizer


def train_regression(
    model,
    train_features, 
    train_labels,
    val_features, 
    val_labels,
    epochs=100, 
    optimizer="adam",
    weight_decay=5e-6,
    lr=0.02, 
):
    optimizer = get_optimizer(optimizer, model.parameters(), lr, weight_decay)

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
