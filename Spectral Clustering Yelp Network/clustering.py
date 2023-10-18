from typing import List
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def deterministic_eigsh(A, **kwargs):
    np.random.seed(0)
    kwargs['v0'] = np.random.rand(min(A.shape))
    return eigsh(A, **kwargs)


def eigsh_help():
    help(eigsh)


def labels_to_list_of_clusters(z: np.array) -> List[List[int]]:
    """Convert predicted label vector to a list of clusters in the graph.
    This function is already implemented, nothing to do here.
    
    Parameters
    ----------
    z : np.array, shape [N]
        Predicted labels.
        
    Returns
    -------
    list_of_clusters : list of lists
        Each list contains ids of nodes that belong to the same cluster.
        Each node may appear in one and only one partition.
    
    Examples
    --------
    >>> z = np.array([0, 0, 1, 1, 0])
    >>> labels_to_list_of_clusters(z)
    [[0, 1, 4], [2, 3]]
    
    """
    return [np.where(z == c)[0] for c in np.unique(z)]


def construct_laplacian(A: sp.csr_matrix, norm_laplacian: bool) -> sp.csr_matrix:
    """Construct Laplacian of a graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    norm_laplacian : bool
        Whether to construct the normalized graph Laplacian or not.
        If True, construct the normalized (symmetrized) Laplacian, L = I - D^{-1/2} A D^{-1/2}.
        If False, construct the unnormalized Laplacian, L = D - A.
        
    Returns
    -------
    L : scipy.sparse.csr_matrix, shape [N, N]
        Laplacian of the graph.
        
    """

    N = len(A[0, :].getnnz(axis=0))
    D = np.zeros([N, N])
    D_ns = np.zeros([N, N])
    D = sp.csr_matrix(D)
    D_ns = sp.csr_matrix(D_ns)

    for i in range(N):
        D[i, i] = A[i, :].sum()
        D_ns[i, i] = D[i, i] ** -0.5

    if norm_laplacian == False:
        L = D - A
    if norm_laplacian == True:
        L = sp.csr_matrix(np.identity(N)) - D_ns * A * D_ns

    return L


def spectral_embedding(A: sp.csr_matrix, num_clusters: int, norm_laplacian: bool) -> np.array:
    """Compute spectral embedding of nodes in the given graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    num_clusters : int
        Number of clusters to detect in the data.
    norm_laplacian : bool, default False
        Whether to use the normalized graph Laplacian or not.
        
    Returns
    -------
    embedding : np.array, shape [N, num_clusters]
        Spectral embedding for the given graph.
        Each row represents the spectral embedding of a given node.
        The rows have to be sorted in ascending order w.r.t. the corresponding eigenvalues.
    
    """
    if (A != A.T).sum() != 0:
        raise ValueError("Spectral embedding doesn't work if the adjacency matrix is not symmetric.")
    if num_clusters < 2:
        raise ValueError("The clustering requires at least two clusters.")
    if num_clusters > A.shape[0]:
        raise ValueError(f"We can have at most {A.shape[0]} clusters (number of nodes).")


    L = construct_laplacian(A, norm_laplacian)
    w, v = eigsh(L, k=num_clusters, which='SM')
    embedding = v

    return embedding


def compute_ratio_cut(A: sp.csr_matrix, z: np.array) -> float:
    """Compute the ratio cut for the given partition of the graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    z : np.array, shape [N]
        Cluster indicators for each node.
    
    Returns
    -------
    ratio_cut : float
        Value of the cut for the given partition of the graph.
        
    """
    
    N = len(A[0, :].getnnz(axis=0))
    list_of_clusters = labels_to_list_of_clusters(z)
    cluster_cut_sum = np.zeros(N)
    j = 0
    k = 0
    summ = 0
    for clist in list_of_clusters:
        sizeC = len(clist)
        cut_sum_node = np.zeros(sizeC)
        for node in clist:
            for i in range(N):
                if i in clist:
                    pass
                else:
                    summ += A[node, i]
            cut_sum_node[k] = summ
            summ = 0
            k += 1
        cluster_cut_sum[j] = cut_sum_node.sum()/sizeC
        j += 1
        k = 0
    ratio_cut = cluster_cut_sum.sum()

    return ratio_cut


def compute_normalized_cut(A: sp.csr_matrix, z: np.array) -> float:
    """Compute the normalized cut for the given partition of the graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    z : np.array, shape [N]
        Cluster indicators for each node.
    
    Returns
    -------
    norm_cut : float
        Value of the normalized cut for the given partition of the graph.
        
    """
    
    N = len(A[0, :].getnnz(axis=0))
    list_of_clusters = labels_to_list_of_clusters(z)
    cluster_cut_sum = np.zeros(N)
    summ = 0; deg_node_sum = 0; j = 0; k = 0
    L = construct_laplacian(A, False)
    for clist in list_of_clusters:
        sizeC = len(clist)
        cut_sum_node = np.zeros(sizeC)
        for node in clist:
            deg_node_sum += L[node, node] + A[node, node]
            for i in range(N):
                if i in clist:
                    pass
                else:
                    summ += A[node, i]
            cut_sum_node[k] = summ;
            summ = 0; k += 1
        cluster_cut_sum[j] = cut_sum_node.sum()/deg_node_sum
        deg_node_sum = 0; k = 0; j += 1
    norm_cut = cluster_cut_sum.sum()

    return norm_cut


def spectral_clustering(A: sp.csr_matrix, num_clusters: int, norm_laplacian: bool, seed: int = 42) -> np.array:
    """Perform spectral clustering on the given graph.
    
    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape [N, N]
        Adjacency matrix of the graph.
    num_clusters : int
        Number of clusters to detect in the data.
    norm_laplacian : bool, default False
        Whether to use the normalized graph Laplacian or not.
    seed : int, default 42
        Random seed to use for the `KMeans` clustering.
        
    Returns
    -------
    z_pred : np.array, shape [N]
        Predicted cluster indicators for each node.
    """
    model = KMeans(num_clusters, random_state=seed)

    embedding = spectral_embedding(A, num_clusters, norm_laplacian)
    if norm_laplacian == True:
        embedding = normalize(embedding, norm='l2', axis=1)
    z_pred = model.fit_predict(embedding)

    return z_pred