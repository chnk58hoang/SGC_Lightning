# Simplifying Graph Convolutional Networks

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

## Overview
This repo contains an implementation and an improved version of the Simple Graph Convolution
(SGC) model, described in the ICML2019 paper [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153).

In general, SGC removes the nonlinearities and collapes the weight matrices in Graph Convolutional Networks (GCNs) and is essentially a linear model. 
For an illustration, ![](./static/model.jpg "SGC")

SGC achieves competitive performance while saving much training time. For reference, on a Tesla T4 (Google Colab's GPU),

Dataset | Metric | Training Time 
:------:|:------:|:-----------:|
Cora    | Acc: 81.0 %     | 0.13s
Citeseer| Acc: 71.9 %     | 0.14s
Pubmed  | Acc: 78.9 %     | 0.29s
Reddit  | F1:  94.9 %     | 2.7s

This home repo contains the implementation for citation networks (Cora, Citeseer, and Pubmed) and social network (Reddit).

## Datasets
We provide the citation network datasets under `data/`, which corresponds to [the public data splits](https://github.com/tkipf/gcn/tree/master/gcn/data).
Due to space limit, please download reddit dataset from [FastGCN](https://github.com/matenure/FastGCN/issues/9) and put `reddit_adj.npz`, `reddit.npz` under `data/`.

Details about the dataset is shown below:

Dataset | # Nodes | # Edges | # Features | # Classes | Train/Dev/Test Nodes
:------:|:-------:|:-------:|:----------:|:---------:|:-------------------:
Cora    | 2,708   | 5,429   | 1,433      | 7         | 140/500/1,000
Citeseer| 3,327   | 4,732   | 4,732      | 6         | 120/500/1,000
Pubmed  | 19,717  | 44,338  | 500        | 3         | 60/500/1,000
Reddit  | 233K    | 11.6M   | 602        | 41        | 152K/24K/55K

## Installation
You should have Python 3.7 or higher. I highly recommend creating a virual environment like venv or [conda](https://docs.conda.io/en/latest/miniconda.html). After that, execute the following commands:

```
git clone https://github.com/chnk58hoang/SGC_Lightning.git
cd SGC_Lightning
pip install -e .
```

## Usage
To run the code, go the `egs` folder and execute the following command:

```
$ python run.py --help
usage: run.py [-h] [--seed SEED] [--no-cuda] [--lightning] config

positional arguments:
  config       Path to yaml config file.

optional arguments:
  -h, --help   show this help message and exit
  --seed SEED  Random seed.
  --no-cuda    Disables CUDA training.
  --lightning  Execute with PyTorch Lightning version.
```

Example of a `config.yaml` file can be found in `configs/`