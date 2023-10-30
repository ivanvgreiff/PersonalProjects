from collections import OrderedDict
from itertools import chain
from typing import List, Tuple

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import torch
from torch import nn
from torch import sparse as sp
from torch.nn import functional as F

from task import *


# Reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# GPU
use_cuda = torch.cuda.is_available() # = False


# Load Data
X = torch.load('./X.pt')
N, D = X.shape

A_indices = torch.load('./A_indices.pt')
A = torch.sparse.FloatTensor(A_indices, torch.ones_like(A_indices[0]).float(), (N, N)).coalesce()
del A_indices

labels = torch.load('./labels.pt')
C = labels.max().item() + 1

if use_cuda:
    A, X, labels = A.cuda(), X.cuda(), labels.cuda()

A, X, labels, N, D, C


three_layer_gcn = GCN(n_features=D, n_classes=C, hidden_dimensions=[64, 64])
if use_cuda:
    three_layer_gcn = three_layer_gcn.cuda()
    
three_layer_gcn


# Training
def split(labels: np.ndarray,
          train_size: float = 0.025,
          val_size: float = 0.025,
          test_size: float = 0.95,
          random_state: int = 42) -> List[np.ndarray]:
    """Split the arrays or matrices into random train, validation and test subsets.

    Parameters
    ----------
    labels: np.ndarray [n_nodes]
        The class labels
    train_size: float
        Proportion of the dataset included in the train split.
    val_size: float
        Proportion of the dataset included in the validation split.
    test_size: float
        Proportion of the dataset included in the test split.
    random_state: int
        Random_state is the seed used by the random number generator;

    Returns
    -------
    split_train: array-like
        The indices of the training nodes
    split_val: array-like
        The indices of the validation nodes
    split_test array-like
        The indices of the test nodes

    """
    idx = np.arange(labels.shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=labels)

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=labels[idx_train_and_val])
    
    return idx_train, idx_val, idx_test

idx_train, idx_val, idx_test = split(labels.cpu().numpy())



trace_train, trace_val = train(three_layer_gcn, X, A, labels, idx_train, idx_val)

plt.plot(trace_train, label='train')
plt.plot(trace_val, label='validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)



three_layer_appnp = APPNP(n_features=D, n_classes=C, hidden_dimensions=[64, 64])
if use_cuda:
    three_layer_appnp = three_layer_appnp.cuda()
    
three_layer_appnp



trace_train, trace_val = train(three_layer_appnp, X, A, labels, idx_train, idx_val)

plt.plot(trace_train, label='train')
plt.plot(trace_val, label='validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)



n_hidden_dimensions = 64
n_propagations = [1,2,3,4,5,10]

test_accuracy_gcn = []
for n_propagation in n_propagations:
    model = GCN(n_features=D, n_classes=C, hidden_dimensions=n_propagation*[n_hidden_dimensions])
    if use_cuda:
        model = model.cuda()
    train(model, X, A, labels, idx_train, idx_val, display_step=-1)

# TODO #

    test_accuracy_gcn.append(accuracy)
    
test_accuracy_appnp = []
for n_propagation in n_propagations:
    model = APPNP(n_features=D, n_classes=C, n_propagation=n_propagation)
    if use_cuda:
        model = model.cuda()
    train(model, X, A, labels, idx_train, idx_val, display_step=-1)

# TODO #

    test_accuracy_appnp.append(accuracy)



plt.plot(n_propagations, test_accuracy_gcn, label='GCN', marker='.')
plt.plot(n_propagations, test_accuracy_appnp, label='APPNP', marker='.')
plt.xlabel('Message passing steps')
plt.ylabel('Accuracy')
plt.ylim(0.7, 0.9)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.legend()
plt.grid(True)