from typing import List
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from clustering import *


list_of_clusters = [[0, 1, 4], [2, 3]]
A = sp.csr_matrix(torch.tensor([[0, 1, 0, 0, 1], [1, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [1, 1, 1, 0, 0]]))
N = 5
cluster_cut_sum = np.zeros(len(list_of_clusters))
j = 0
summ = 0
k = 0
deg_node_sum = 0
L = construct_laplacian(A, True)
for clist in list_of_clusters:
    sizeC = len(clist)
    cut_sum_node = np.zeros(sizeC)
    for node in clist:
        deg_node_sum += L[node, node]
        for i in range(N):
            if i in clist:
                pass
            else:
                summ += A[node, i]
        cut_sum_node[k] = summ
        k += 1
        summ = 0
    cluster_cut_sum[j] = cut_sum_node.sum()/deg_node_sum
    deg_node_sum = 0
    j += 1
    k = 0
norm_cut = cluster_cut_sum.sum()
print(norm_cut)