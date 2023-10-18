# Functionality to compute token embeddings

import numpy as np

from numpy.typing import NDArray
from typing import Dict

class Embedding():
    """
    Token embedding model.
    
    Args:
        vocabulary_size (int): The number of unique tokens in the corpus
        embedding_dim (int): Dimension of the token vector embedding
    """
    def __init__(self, vocabulary_size: int, embedding_dim: int):
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim

        self.ctx = None # Used to store values for backpropagation

        self.U = None
        self.V = None
        self.reset_parameters()

    def reset_parameters(self):
        """
        We initialize weight matrices U and V of dimension (D, N) and (N, D), respectively
        """
        self.ctx = None
        self.U = np.random.normal(0, np.sqrt(6. / (self.embedding_dim + self.vocabulary_size)), (self.embedding_dim, self.vocabulary_size))
        self.V = np.random.normal(0, np.sqrt(6. / (self.embedding_dim + self.vocabulary_size)), (self.vocabulary_size, self.embedding_dim))

    def one_hot(self, sequence: NDArray, num_classes: int) -> NDArray:
        """
        Given a vector returns a matrix with rows corresponding to one-hot encoding.
        
        Args:
            sequence (NDArray, shape [t]): A sequence of length t containing tokens represented by integers from [0, self.vocabulary_size - 1]
            num_classes (int): How many potential classes (i.e. tokens) there are
            
        Returns:
            NDArray, shape [vocabulary_size, t]: The one-hot encoded representation of `sequence`
        """

        one_hot = np.zeros((self.vocabulary_size, len(sequence)))
        for k in range(len(sequence)):
            one_hot[sequence[k], k] = 1

        return one_hot

    def softmax(self, x: NDArray, axis: int) -> NDArray:
        """
        Computes a numerically stable version of the softmax along an axis.
        
        Args:
            x (NDArray): The input to normalize, any non-empty matrix.
            axis (int): Along which axis to normalize, i.e. along which dimension the softmax is performed.
        
        Returns:
            y (NDArray): Array with same dimension as `input`, but with normalized values.
        """

        x_max = np.max(x, axis=axis, keepdims=True)
        summation = np.sum(np.exp(x - x_max), axis=axis, keepdims=True)
        y = np.exp(x - x_max) / summation

        return y

    def loss(self, y_true: NDArray, y_predicted: NDArray) -> float:
        """
        Computes the cross-entropy loss $-1 / M * sum_i(sum_j(y_ij * log(prob_ij)))$ for
        predicted probabilities and ground-truth probabilities. 
        
        Parameters
        ----------
        y: array
            (vocabulary_size, num_samples) matrix of M samples where columns are one-hot vectors for true values
        prob: array
            (vocabulary_size, num_samples) column of M samples where columns are probability vectors after softmax

        Returns
        -------
        loss: float
            Cross-entropy loss calculated as: -1 / M * sum_i(sum_j(y_ij * log(prob_ij)))
        """

        y_predicted = np.clip(y_predicted, 1e-8, None)

        m, n = y_true.shape
        inner = np.zeros(n)
        innersum = np.zeros(m)
        for i in range(m):
            for j in range(n):
                inner[j] = y_true[i,j] * np.log(y_predicted[i,j])
            innersum[i] = np.sum(inner)
        loss = -1 / n * np.sum(innersum)

        """
        FIGURE OUT WHAT THE ISSUE WAS BEFORE ADDING THESE INDICES IN J AND I : I THINK IT JUST KEPT MAKING BOTH BIGGER
        """
        
        return loss

    def forward(self, x: NDArray, y: NDArray) -> float:
        """
        Performs forward pass and saves activations for backward pass
        
        Args:
            x (NDArray, shape [sequence_length], dtype int): Mini-batch of token indices to predict contexts for
            y (NDArray, shape [sequence_length], dtype int): Mini-batch of output context tokens
        
        Returns:
            float: The cross-entropy loss
        """
        
        # Input transformation
        """
        Input is represented with M-dimensional vectors
        convert them to (vocabulary_size, sequence_length) matrices such that columns are one-hot 
        representations of the input
        """
        x = self.one_hot(x, self.vocabulary_size)
        y = self.one_hot(y, self.vocabulary_size)
        
        # Forward propagation, needs to compute the following
        """
        Returns
        -------
        embedding (NDArray, shape [embedding_dim, sequence_length]): matrix where columns are token embedding from U matrix
        logits (NDArray, shape [vocabulary_size, sequence_length]): matrix where columns are output logits
        prob (NDArray, shape [vocabulary_size, sequence_length]): matrix where columns are output probabilities

        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.U = None
        self.V = None
        U,V^T ~ (self.embedding_dim, self.vocabulary_size)
        """
        # x~(vocab_size, sqn_length)
        embedding = np.dot(self.U, x)
        logits = np.dot(self.V, embedding)
        prob = self.softmax(logits, axis=0)
        # axis=0 columns // axis=1 rows

        # Save values for backpropagation
        self.ctx = (embedding, logits, prob, x, y)
        
        # Loss calculation
        loss = self.loss(y, prob)
        
        return loss
        
    def backward(self) -> Dict[str, NDArray]:
        """
        Given parameters from forward propagation, returns gradient of U and V.
        
        Returns
        -------
        Dict: Gradients with the following keys:
            V (NDArray, shape [vocabulary_size, embedding_dim]) matrix of partial derivatives of loss w.r.t. V
            U (NDArray, shape [embedding_dim, vocabulary_size]) matrix of partial derivatives of loss w.r.t. U
        """

        embedding, logits, prob, x, y = self.ctx

        m, n = y.shape
        d_embedding = np.matmul(self.V.T, prob - y)
        prevalue = self.U@x
        d_V = np.matmul(prevalue, (prob - y).T) / m
        d_U = np.matmul(d_embedding, x.T) / m
        d_V = d_V.T
        """
        dL_dV = np.outer(np.dot(self.U, x), prob - y)
        dL_dU = np.outer(x, np.dot(self.V.transpose(), prob - y))
        d_V = dL_dV
        d_U = dL_dU
        """
        return { 'V': d_V, 'U': d_U }
