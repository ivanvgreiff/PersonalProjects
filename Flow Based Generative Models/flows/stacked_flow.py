from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from .nf_utils import Flow


class StackedFlows(nn.Module):
    """Stack a list of transformations with a given based distribtuion.

    Args:
        transforms: list fo stacked transformations. list of Flows
        dim: dimension of input/output data. int
        base_dist: name of the base distribution. options: ['Normal']
    """

    def __init__(
        self,
        transforms: List[Flow],
        dim: int = 2,
        base_dist: str = "Normal",
        device="cpu",
    ):
        super().__init__()

        if isinstance(transforms, Flow):
            self.transforms = nn.ModuleList(
                [
                    transforms,
                ]
            )
        elif isinstance(transforms, list):
            if not all(isinstance(t, Flow) for t in transforms):
                raise ValueError("transforms must be a Flow or a list of Flows")
            self.transforms = nn.ModuleList(transforms)
        else:
            raise ValueError(
                f"transforms must a Flow or a list, but was {type(transforms)}"
            )

        self.dim = dim
        if base_dist == "Normal":
            self.base_dist = MultivariateNormal(
                torch.zeros(self.dim).to(device), torch.eye(self.dim).to(device)
            )
        else:
            raise NotImplementedError

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability of a batch of data (slide 27).

        Args:
            x: input sample. shape [batch_size, dim]

        Returns:
            log_prob: Log probability of the data, shape [batch_size]
        """

        B, D = x.shape

        ##########################################################
        # YOUR CODE HERE
        for t in reversed(self.transforms):
            z = x
            inv_log_abs_det_jac_prev = 0
            z_prev, inv_log_abs_det_jac = t.inverse(z)
            inv_log_abs_det_jac_prev = inv_log_abs_det_jac_prev + inv_log_abs_det_jac
            x = z_prev
        z_prev
        final_prob = self.base_dist.log_prob(x)
        log_prob = final_prob + inv_log_abs_det_jac_prev
        ##########################################################

        assert log_prob.shape == (B,)

        return log_prob

    def rsample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample from the transformed distribution (slide 31).

        Returns:
            x: sample after forward transformation, shape [batch_size, dim]
            log_prob: Log probability of x, shape [batch_size]
        """
        ##########################################################
        # YOUR CODE HERE


        z = self.base_dist.rsample(sample_shape=[batch_size])
        log_det_jac_next = 0
        z_init = z
        for t in self.transforms:
            x_next, log_det_jac = t.forward(z)
            log_det_jac_next = log_det_jac_next + log_det_jac
            z = x_next
        init_prob = self.base_dist.log_prob(z_init)
        log_prob = init_prob - log_det_jac_next
        x = z

        # log_prob = final_prob + log_det_jac_next
        # beginning_prob = log_prob(x_end) - log_det_jac_next {

        ##########################################################

        assert x.shape == (batch_size, self.dim)
        assert log_prob.shape == (batch_size,)

        return x, log_prob
