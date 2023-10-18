import numpy as np


#vocabulary = ('This', 'is', 'the', 'to', 'sentence')
counts = (1, 1, 1, 2, 1)

sequence = ('This', 'is', 'the', 'to', 'be', 'to', 'sentence')
window_size = 2

token_to_idx = {}
b = 0
for word in sequence:
    if word in token_to_idx:
        a = 1
    else:
        token_to_idx[word] = b
        b += 1
print(token_to_idx)

idx_to_token = {}
for index in token_to_idx.values():
    #idx_to_token[index] = list(token_to_idx.keys())[list(token_to_idx.values()).index(index)] 
    idx_to_token[index] = list(token_to_idx.keys())[index]
print(idx_to_token)

idx_to_count = {}
for i in range(len(counts)):
    idx_to_count[i] = counts[i]
print(idx_to_count)

print(sequence)
