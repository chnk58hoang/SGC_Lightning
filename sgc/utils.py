import sys
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from time import perf_counter

import torch
import networkx as nx

from sgc.normalization import fetch_normalization, row_normalize


def get_optimizer(optim, params, lr=0.2, weight_decay=5e-6):
    if optim.lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay) 
    elif optim.lower() == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif optim.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS(params, lr=lr)
    else:
        raise ValueError("Unsupported optimizer: {}".format(optim))
    return optimizer


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, normalization="AugNormAdj", gamma=1):
    adj_normalizer = fetch_normalization(normalization)
    if 'Aug' in normalization:
        adj = adj_normalizer(adj, gamma=gamma)
    else:
        adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_citation(data_path='data/', dataset_str="cora", normalization="AugNormAdj", cuda=True, gamma=1):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(data_path, dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(f"{data_path}/ind.{dataset_str}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    adj, features = preprocess_citation(adj, features, normalization, gamma=gamma)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test


def sgc_precompute(features, adj, degree, alpha=0.1):
    org_features = features
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
   
    # Add balance weight
    features = (1 - alpha) * features + alpha * org_features

    precompute_time = perf_counter() - t
    return features, precompute_time


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir + "reddit_adj.npz")
    data = np.load(dataset_dir + "reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], \
           data['test_index']


def load_reddit_data(data_path="data/", normalization="AugNormAdj", cuda=True, gamma=1.0):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ(data_path)
    labels = np.zeros(adj.shape[0])
    labels[train_index] = y_train
    labels[val_index] = y_val
    labels[test_index] = y_test
    adj = adj + adj.T
    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features - features.mean(dim=0)) / features.std(dim=0)
    adj_normalizer = fetch_normalization(normalization)
    if "Aug" in normalization:
        adj = adj_normalizer(adj, gamma)
    else:
        adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    if "Aug" in normalization:
        train_adj = adj_normalizer(train_adj, gamma)
    else:
        train_adj = adj_normalizer(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
        
    return adj, train_adj, features, labels, train_index, val_index, test_index
