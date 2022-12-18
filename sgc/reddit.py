from time import perf_counter

import torch.nn.functional as F

from sgc.metrics import f1
from sgc.utils import get_optimizer


def train_regression(
    model,
    train_features,
    train_labels,
    epochs=100,
    optimizer="adam",
    weight_decay=5e-6,
    lr=0.02,
):
    optimizer = get_optimizer(optimizer, model.parameters(), lr, weight_decay)
    model.train()

    def closure():
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        return loss_train

    t = perf_counter()
    for _ in range(epochs):
        loss_train = optimizer.step(closure)
    train_time = perf_counter() - t
    return model, train_time


def test_regression(model, test_features, test_labels):
    model.eval()
    return f1(model(test_features), test_labels)
