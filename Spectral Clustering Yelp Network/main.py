import numpy as np
import scipy.sparse as sp
from clustering import construct_laplacian
from scipy.sparse.linalg import eigsh
from typing import List
from clustering import spectral_embedding
from clustering import compute_ratio_cut, compute_normalized_cut, labels_to_list_of_clusters, spectral_clustering

A = sp.load_npz('A.npz')
F = np.load('F.npy')
categories = np.load('categories.npy', allow_pickle=True).tolist()

assert A.shape[0] == F.shape[0]
assert F.shape[1] == len(categories)

print(f'The adjacency matrix is {"symmetric" if (A != A.T).sum() == 0 else "asymmetric"}')

# help(eigsh)

num_clusters = 6
np.random.seed(12903)
norm_laplacian = False
z_unnorm = spectral_clustering(A, num_clusters, norm_laplacian)
print('When using L_unnorm:')
print(' ratio cut = {:.3f}'.format(compute_ratio_cut(A, z_unnorm)))
print(' normalized cut = {:.3f}'.format(compute_normalized_cut(A, z_unnorm)))
print(' sizes of partitions are: {}'.format([len(clust) for clust in labels_to_list_of_clusters(z_unnorm)]))

np.random.seed(12323)
norm_laplacian = True
z_norm = spectral_clustering(A, num_clusters, norm_laplacian)
print('When using L_norm:')
print(' ratio cut = {:.3f}'.format(compute_ratio_cut(A, z_norm)))
print(' normalized cut = {:.3f}'.format(compute_normalized_cut(A, z_norm)))
print(' sizes of partitions are: {}'.format([len(clust) for clust in labels_to_list_of_clusters(z_norm)]))


def print_top_categories_for_each_cluster(top_k: int, z: np.array, F: sp.csr_matrix, categories: List[str]):
    """Print the top-K categories among users in each cluster.
    For each cluster, the function prints names of the top-K categories,
    and number of users that like the respective category (separated by a comma).
    The function doesn't return anything, just prints the output.
    
    Parameters
    ----------
    top_k : int
        Number of most popular categories to print for each cluster.
    z : np.array, shape [N]
        Cluster labels.
    F : sp.csr_matrix, shape [N, C]
        Matrix that tells preferences of each user to each category.
        F[i, c] = 1 if user i gave at least one positive review to at least one restaurant in category c.
    categories : list, shape [C]
        Names of the categories.
        
    """

    list_of_clusters = labels_to_list_of_clusters(z)
    
    for ix, clust in enumerate(list_of_clusters):
        cat_ratings = sum(F[i] for i in clust)
        top_cats = np.argsort(-cat_ratings)[:top_k]
        print('Most popular categories in cluster {}'.format(ix))
        for t in top_cats:
            print(f' - {categories[t]}, {int(cat_ratings[t])}')
        print()


np.random.seed(23142)
z_norm = spectral_clustering(A, num_clusters, True)
r = print_top_categories_for_each_cluster(top_k=5, z=z_norm, F=F, categories=categories)
