import numpy as np

sequence = ('This', 'is', 'the', 'to', 'be', 'to', 'sentence')

token_to_idx = {}
b = 0
for word in sequence:
    if word in token_to_idx:
        a = 1
    else:
        token_to_idx[word] = b
        b += 1
print(token_to_idx)

window_size = 2
newlist2 = []
c=0
d=0
e=0
for word in sequence:

    if sequence.index(word, d) < window_size:
        for x in range(sequence.index(word, d)+1):
            if x == 0:
                c = 1
            else:
                newlist2.append(( token_to_idx[word], token_to_idx[sequence[sequence.index(word, d) - x]] ))
    else:
        for x in range(window_size+1):
            if x == 0:
                c = 1
            else:
                newlist2.append(( token_to_idx[word], token_to_idx[sequence[sequence.index(word, d) - x]] ))

    if sequence.index(word, d) + window_size >= len(sequence):
        for x in range(len(sequence) - sequence.index(word, d)):
            if x == 0:
                c = 1
            else:
                newlist2.append(( token_to_idx[word], token_to_idx[sequence[sequence.index(word, d) + x]] ))
    else:
        for x in range(window_size+1):
            if x == 0:
                c = 1
            else:
                newlist2.append(( token_to_idx[word], token_to_idx[sequence[sequence.index(word, d) + x]] ))

    d += 1

print(newlist2)