from typing import Tuple, Union

import torch
from torch.nn import functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


@typechecked
class NeuralTPP(torch.nn.Module):
    """Neural Temporal Point Process class
    Args:
        hidden_dim (int): Number of history_emb dimensions.
    """

    def __init__(self, hidden_dim: int = 16):
        super(NeuralTPP, self).__init__()

        self.hidden_dim = hidden_dim

        # Single layer RNN for history embedding with tanh nonlinearity
        self.embedding_rnn = torch.nn.RNN(input_size=2, hidden_size=hidden_dim, num_layers=1, nonlinearity='tanh', bias=True, batch_first=True)
        #######################################################

        # Single layer neural network to predict mu and log(sigma)
        self.linear = torch.nn.Linear(in_features=hidden_dim, out_features=2)
        #######################################################

        # value to be used for numerical problems
        self.eps = 1e-8

    def log_likelihood(
        self,
        times: TensorType[torch.float32, "batch", "max_seq_length"],
        mask: TensorType[torch.bool, "batch", "max_seq_length"],
    ) -> TensorType[torch.float32, "batch"]:
        """Compute the log-likelihood for a batch of padded sequences.
        Args:
            times (Tensor): Padded inter-event times,
                shape (batch_size, seq_len+1)
            mask (Tensor): Boolean mask that indicates which entries
                do NOT correspond to padding, shape (batch_size, seq_len+1)
        Returns:
            log_likelihood: Log-likelihood for each sample in the batch,
                shape (batch_size,)
        """
        # clamp for stability
        times = torch.clamp(times, min=self.eps)

        # get history_emb
        history_emb = self.embed_history(times)

        # get cond. distributions
        mu, sigma = self.get_distribution_parameters(history_emb)
        dist = self.get_distributions(mu, sigma)

        # calculate negative log_likelihood
        log_density = self.get_log_density(dist, times, mask)
        log_survival = self.get_log_survival_prob(dist, times, mask)

        log_likelihood = log_density + log_survival

        return log_likelihood

    def get_log_density(
        self,
        distribution: torch.distributions.LogNormal,
        times: TensorType[torch.float32, "batch", "max_seq_length"],
        mask: TensorType[torch.bool, "batch", "max_seq_length"],
    ) -> TensorType["batch"]:
        """Compute the log-density for a batch of padded sequences.
        Args:
            distribution (torch.distributions.LogNormal): instance of pytorch distribution class
            times (Tensor): Padded inter-event times,
                shape (batch_size, seq_len+1)
            mask (Tensor): Boolean mask that indicates which entries
                do NOT correspond to padding, shape (batch_size, seq_len+1)
            B (int): batch size
            seq_len (int): max sequence length
        Returns:
            log_density: Log-density for each sample in the batch,
                shape (batch_size,)
        """
        # calculate log density
        batch_size = len(times[:, 0])
        seq_len = len(times[0, :])

        newmask = mask[:, 1:]
        densities = distribution.log_prob(times)
        densities2 = densities[:, :-1]
        densities3 = densities2*newmask
        log_density = torch.sum(densities3, dim=1)

        return log_density

    def get_log_survival_prob(
        self,
        distribution: torch.distributions.LogNormal,
        times: TensorType[torch.float32, "batch", "max_seq_length"],
        mask: TensorType[torch.bool, "batch", "max_seq_length"],
    ) -> TensorType["batch"]:
        """Compute the log-intensities for a batch of padded sequences.
        Args:
            distribution (torch.distributions.LogNormal): instance of pytorch distribution class
            times (Tensor): Padded inter-event times,
                shape (batch_size, seq_len+1)
            mask (Tensor): Boolean mask that indicates which entries
                do NOT correspond to padding, shape (batch_size, seq_len+1)
            B (int): batch size
            seq_len (int): max sequence length
        Returns:
            log_surv_last: Log-survival probability for each sample in the batch,
                shape (batch_size,)
        """
        # calculate log survival probability
        #tau_Nplus = times[:, len(times[0, :])-1]
        seq_len = len(times[0, :])
        batch_size = len(times[:, 0])

        last_element = torch.zeros([batch_size])
        cdftimes = torch.log(1 - distribution.cdf(times))

        count1 = 0
        for i in range(batch_size):
            count1 = 0
            for j in range(seq_len):
                if mask[i, j] == True:
                    count1 += 1
                else:
                    last_element[i] = cdftimes[i, count1 - 1]
                    count1=0
                    break
        log_surv_last = last_element

        return log_surv_last

    def encode(
        self, times: TensorType[torch.float32, "batch", "max_seq_length"]
    ) -> TensorType[torch.float32, "batch", "max_seq_length", 2]:

        x = None
        batch_size = len(times[:, 1])
        max_seqlength = len(times[1, :])
        x = torch.zeros([batch_size, max_seqlength, 2])
        for i in range(batch_size):
            for j in range(max_seqlength):
                x[i, j, 0] = times[i, j]
                x[i, j, 1] = torch.log(times[i,j])

        return x

    def embed_history(
        self, times: TensorType[torch.float32, "batch", "max_seq_length"]
    ) -> TensorType[torch.float32, "batch", "max_seq_length", "history_emb_dim"]:
        """Embed history for a batch of padded sequences.
        Args:
            times: Padded inter-event times,
                shape (batch_size, max_seq_length)
        Returns:
            history_emb: history_emb embedding of the history,
                shape (batch_size, max_seq_length, embedding_dim)
        """
        batch_size = len(times[:, 1])

        x = self.encode(times)
        c_1 = torch.zeros([1, batch_size, self.hidden_dim])
        history_emb, c_n = self.embedding_rnn(x, c_1)
        history_emb = history_emb[:, 0:len(history_emb[1, :, 1])-1, :]
        c_new = torch.zeros([batch_size, 1, self.hidden_dim])
        history_emb = torch.cat((c_new, history_emb), dim=1)

        return history_emb

    def get_distributions(
        self,
        mu: TensorType[torch.float32, "batch", "max_seq_length"],
        sigma: TensorType[torch.float32, "batch", "max_seq_length"],
    ) -> Union[torch.distributions.LogNormal, None]:
        """Get log normal distribution given mu and sigma.
        Args:
            mu (tensor): predicted mu (batch, max_seq_length)
            sigma (tensor): predicted sigma (batch, max_seq_length)

        Returns:
            Distribution: log_normal
        """

        log_norm_dist = torch.distributions.LogNormal(mu, sigma)

        return log_norm_dist

    def get_distribution_parameters(
        self,
        history_emb: TensorType[
            torch.float32, "batch", "max_seq_length", "history_emb_dim"
        ],
    ) -> Tuple[
        TensorType[torch.float32, "batch", "max_seq_length"],
        TensorType[torch.float32, "batch", "max_seq_length"],
    ]:
        """Compute distribution parameters.
        Args:
            history_emb (Tensor): history_emb tensor,
                shape (batch_size, seq_len+1, C)
        Returns:
            Parameter (Tuple): mu, sigma
        """  
        seq_length = len(history_emb[0, :, 0])
        batch_size = len(history_emb[:, 0, 0])

        mu = torch.zeros([batch_size, seq_length])
        sigma = torch.zeros([batch_size, seq_length])
            
        for i in range(seq_length):
            output = self.linear(history_emb[:, i, :])
            print(output)
            mu[:, i] = output[:, 0]
            sigma[:, i] = torch.exp(output[:, 1])

        return mu, sigma

    def forward(self):
        """
        Not implemented
        """
        pass
