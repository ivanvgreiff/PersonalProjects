import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from model import Embedding
from train import Optimizer

from data import load_data, build_vocabulary, compute_token_to_index, get_token_pairs_from_window


reviews_1star, reviews_5star = load_data('task03_data.npy')
corpus = reviews_1star + reviews_5star
corpus, vocabulary, counts = build_vocabulary(corpus)
token_to_idx, idx_to_token, idx_to_count = compute_token_to_index(vocabulary, counts)
data = np.array(sum((list(get_token_pairs_from_window(sequence, 3, token_to_idx)) 
                        for sequence in corpus), start = [])) # N, 2
# Should output
# Total number of pairs: 207462
print('Total number of pairs:', data.shape[0])


VOCABULARY_SIZE = len(vocabulary)
EMBEDDING_DIM = 32


print('Number of positive reviews:', len(reviews_1star))
print('Number of negative reviews:', len(reviews_5star))
print('Number of unique words:', VOCABULARY_SIZE)


# Compute sampling probabilities
probabilities = np.array([1 - np.sqrt(1e-3 / idx_to_count[token_idx]) for token_idx in data[:, 0]])
probabilities /= np.sum(probabilities)
# Should output: 
# [4.8206203e-06 4.8206203e-06 4.8206203e-06]
print(probabilities[:3])


rng = np.random.default_rng(123)
def get_batch(data, size, prob):
    x = rng.choice(data, size, p=prob)
    return x[:,0], x[:,1]


model = Embedding(VOCABULARY_SIZE, EMBEDDING_DIM)
optim = Optimizer(model, learning_rate=1.0, momentum=0.5)

losses = []

MAX_ITERATIONS = 15000
PRINT_EVERY = 1000
BATCH_SIZE = 1000

for i in range(MAX_ITERATIONS):
    x, y = get_batch(data, BATCH_SIZE, probabilities)
    
    loss = model.forward(x, y)
    grad = model.backward()
    optim.step(grad)
    
    assert not np.isnan(loss)
    
    losses.append(loss)

    if (i + 1) % PRINT_EVERY == 0:
        print(f'Iteration: {i + 1}, Avg. training loss: {np.mean(losses[-PRINT_EVERY:]):.4f}')


emb_matrix = model.U.T