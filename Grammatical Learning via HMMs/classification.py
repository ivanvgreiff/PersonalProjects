import numpy as np


def classify_review(hmm_1, hmm_5, p, sentence_in):
    """Given the trained models `hmm_1` and `hmm_2` and frequency of
       1-star reviews, classifies `sentence_in`

    Parameters
    ----------
    hmm_1 : HMM_TxtGenerator
        The trained model on 1-star reviews.
    hmm_5 : HMM_TxtGenerator
        The trained model on 5-star reviews.
    p: a scalar in [0,1]
        frequency of 1-star reviews, (#1star)/(#1star + #5star)

    Returns
    -------
    c : int in {1,5}
        c=1 means sentence_in is classified as 1.
        similarly c=5 means sentence_in is classified as 5.
        If both sentences are equally likely, you can return either 1 or 5.
    """

    hmm1_loglik_sentence = hmm_1.loglik_sentence(sentence_in)
    hmm5_loglik_sentence = hmm_5.loglik_sentence(sentence_in)

    log_prior_1 = np.log(p)
    log_prior_5 = np.log(1 - p)

    log_post_1 = log_prior_1 + hmm1_loglik_sentence
    log_post_5 = log_prior_5 + hmm5_loglik_sentence

    return 1 if log_post_1 > log_post_5 else 5
