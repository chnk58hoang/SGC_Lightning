import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='20ng', help='Dataset string.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='training batch size.')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight for L2 loss on embedding matrix.')
    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation.')
    parser.add_argument('--tuned', action='store_true', help='use tuned hyperparams')
    parser.add_argument('--preprocessed', action='store_true',
                        help='use preprocessed data')
    args = parser.parse_args()

    return args
