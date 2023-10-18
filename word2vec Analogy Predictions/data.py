# Functionality to load the dataset
# We will be working with restaurant reviews in Las Vegas.

import numpy as np
from collections import Counter
from typing import List, Sequence, Tuple, Dict
from numpy.typing import NDArray

UNKNOWN = '_unk'

def load_data(path: str = 'task03_data.npy') -> Tuple[List[List[str]]]:
    """Loads the dataset and returns 1-star and 5-star reviews.

    Args:
        path (str, optional): Path to the dataset. Defaults to 'task03_data.npy'.

    Returns:
        List[List[str]]: 1-star reviews
        List[List[str]]: 5-star reviews
    """
    data = np.load(path, allow_pickle=True).item()
    reviews_1_star = [[token.lower() for token in sequence] for sequence in data['reviews_1star']]
    reviews_5_star = [[token.lower() for token in sequence] for sequence in data['reviews_5star']]
    return reviews_1_star, reviews_5_star
    
def build_vocabulary(corpus: List[List[str]]) -> Tuple[List[List[str]], Tuple[str], Tuple[int]]:
    """Builds the vocabulary and counts how often each token occurs. Also adds a token for uncommon words, UNKNOWN. 

    Args:
        corpus (List[List[str]]): All sequences in the dataset

    Returns:
        List[List[str]]: All sequences in the dataset, where infrequent tokens are replaced with UNKNOWN
        Tuple[str]: All unique tokens in the dataset
        Tuple[int]: The frequency of each token
    """
    corpus_flattened = [token for sequence in corpus for token in sequence]
    vocabulary, counts = zip(*Counter(corpus_flattened).most_common(200))
    # filter the corpus for the most common words
    corpus = [[token if token in vocabulary else UNKNOWN for token in sequence] for sequence in corpus]
    vocabulary += (UNKNOWN, )
    counts += (sum([token == UNKNOWN for sequence in corpus for token in sequence]),)
    return corpus, vocabulary, counts

def compute_token_to_index(vocabulary: Tuple[str], counts: Tuple[int]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Computes a mapping from tokens in the vocabulary to integeres and vice versa. 

    Args:
        vocabulary (Tuple[str]): All unique tokens in the vocabulary
        counts (Tuple[int]): How often each token appears

    Returns:
        Dict[str, int]: A mapping from token to its unique index
        Dict[int, str]: The inverse mapping from index to token
        Dict[int, int]: Mapping from unique token index to count
    """

    token_to_idx = {}
    b = 0
    for word in vocabulary:
        if word in token_to_idx:
            a = 1
        else:
            token_to_idx[word] = b
            b += 1

    idx_to_token = {}
    for index in token_to_idx.values():
        #idx_to_token[index] = list(token_to_idx.keys())[list(token_to_idx.values()).index(index)] 
        idx_to_token[index] = list(token_to_idx.keys())[index]

    idx_to_count = {}

    for i in range(len(counts)):
        idx_to_count[i] = counts[i]

    return token_to_idx, idx_to_token, counts


def get_token_pairs_from_window(sequence: Sequence[str], window_size: int, token_to_index: Dict[str, int]) -> Sequence[Tuple[int, int]]:
    """ Collect all ordered token pairs from a sentence (sequence) that are at most `window_size` apart.
    Note that duplicates should appear more than once, e.g. for "to be to", (to, be) should be returned more than once, as "be"
    is in the context of both the first and the second "to".

    Args:
        sequence (Sequence[str]): The sentence to get tokens from
        window_size (int): The maximal window size
        token_to_index (Dict[str, int]): Mapping from tokens to numerical indices

    Returns:
        Sequence[Tuple[int, int]]: A list of pairs (token_index, token_in_context_index) with pairs of tokens that co-occur, represented by their numerical index.
    """
    newlist2 = []
    c=0
    d=0
    e=0
    for word in sequence:

        if sequence.index(word, d) + window_size >= len(sequence):
            for x in range(len(sequence) - sequence.index(word, d)):
                if x == 0:
                    c = 1
                else:
                    newlist2.append(( token_to_index[word], token_to_index[sequence[sequence.index(word, d) + x]] ))
        else:
            for x in range(window_size+1):
                if x == 0:
                    c = 1
                else:
                    newlist2.append(( token_to_index[word], token_to_index[sequence[sequence.index(word, d) + x]] ))

        if sequence.index(word, d) < window_size:
            for x in range(sequence.index(word, d)+1):
                if x == 0:
                    c = 1
                else:
                    newlist2.append(( token_to_index[word], token_to_index[sequence[sequence.index(word, d) - x]] ))
        else:
            for x in range(window_size+1):
                if x == 0:
                    c = 1
                else:
                    newlist2.append(( token_to_index[word], token_to_index[sequence[sequence.index(word, d) - x]] ))
        d += 1

    return newlist2
