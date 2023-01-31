#%%
import dgl
import ipdb
import time
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Any, Iterable, List, Optional, Tuple, Union
import scipy.sparse
from torch import Tensor
import os
import joblib

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def from_scipy_sparse_matrix(
        A: scipy.sparse.spmatrix) -> Tuple[Tensor, Tensor]:
    r"""Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.

    Examples:

        >>> edge_index = torch.tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> adj = to_scipy_sparse_matrix(edge_index)
        >>> # `edge_index` and `edge_weight` are both returned
        >>> from_scipy_sparse_matrix(adj)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([1., 1., 1., 1., 1., 1.]))
    """
    A = A.tocoo()
    row = torch.from_numpy(A.row).to(torch.long)
    col = torch.from_numpy(A.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = torch.from_numpy(A.data)
    return edge_index, edge_weight

import warnings
warnings.filterwarnings('ignore')

from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='loan',
                    choices=['nba','bail','loan', 'credit', 'german'])

# Load data
args = parser.parse_known_args()[0]
print(args.dataset)

# Example
createFolder(f"./{args.dataset}/")

# Load credit_scoring dataset
if args.dataset == 'credit':
	sens_attr = "Age"  # column number after feature process is 1
	sens_idx = 1
	predict_attr = 'NoDefaultNextMonth'
	label_number = 6000
	path_credit = "./dataset/credit"
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(args.dataset, sens_attr,
	                                                                        predict_attr, path=path_credit,
	                                                                        label_number=label_number
	                                                                        )
	norm_features = feature_norm(features)
	norm_features[:, sens_idx] = features[:, sens_idx]
	features = norm_features

# Load german dataset
elif args.dataset == 'german':
	sens_attr = "Gender"  # column number after feature process is 0
	sens_idx = 0
	predict_attr = "GoodCustomer"
	label_number = 100
	path_german = "./dataset/german"
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(args.dataset, sens_attr,
	                                                                        predict_attr, path=path_german,
	                                                                        label_number=label_number,
	                                                                        )
	norm_features = feature_norm(features)
	norm_features[:, sens_idx] = features[:, sens_idx]
	features = norm_features
else:
	print('Invalid dataset name!')
	exit(0)

# edge index
edge_index = from_scipy_sparse_matrix(adj)[0]

# summary
print(f"Number of nodes: {adj.shape[0]}")
print(f"Number of features: {features.shape[1]}")

num_class = labels.unique()
print(f"Number of classes: {num_class}")

edge_index_np = edge_index.cpu().numpy()
edge_index_df = pd.DataFrame(edge_index_np)

features_np = features.cpu().numpy()
features_df = pd.DataFrame(features_np)

labels_np = labels.cpu().numpy()
labels_df = pd.DataFrame(labels_np)

# pickle adjacency matrix
joblib.dump(adj, f'./{args.dataset}/adj.pkl')

# save datasets
edge_index_df.to_csv(f'./{args.dataset}/edge_index.csv', index=False)
features_df.to_csv(f'./{args.dataset}/features.csv', index=False)
labels_df.to_csv(f'./{args.dataset}/labels.csv', index=False)

