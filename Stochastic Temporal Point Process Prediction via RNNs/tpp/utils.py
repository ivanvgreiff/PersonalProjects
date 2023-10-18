from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


@typechecked
def get_sequence_batch(
    inter_event_times: List[TensorType[torch.float32]],
) -> Tuple[
    TensorType[torch.float32, "batch", "max_seq_length"],
    TensorType[torch.bool, "batch", "max_seq_length"],
]:
    """
    Generate padded batch and mask for list of sequences.

        Args:
            inter_event_times (List): list of inter-event times

        Returns:
            batch: batched inter-event times. shape [batch_size, max_seq_length]
            mask: boolean mask indicating inter-event times. shape [batch_size, max_seq_length]
    """

    batch = None
    mask = None
    batch_size = len(inter_event_times)

    length_seq = torch.zeros(batch_size)
    a = 0
    for seq in inter_event_times:
        length_seq[a] = len(seq)
        a += 1
    max_seq_length = max(length_seq).int() #.item() returns float

    batch = torch.zeros([batch_size, max_seq_length])
    mask = torch.zeros([batch_size, max_seq_length], dtype=torch.bool)

    b = 0
    for seq in inter_event_times:
        for i in range(len(seq)):
            batch[b, i] = seq[i]
            mask[b, i] = True
        b += 1

    return batch, mask


@typechecked
def get_tau(
    t: TensorType[torch.float32, "sequence_length"], t_end: TensorType[torch.float32, 1]
) -> TensorType[torch.float32]:
    """
    Compute inter-eventtimes from arrival times

        Args:
            t: arrival times. shape [seq_length]
            t_end: end time of the temporal point process.

        Returns:
            tau: inter-eventtimes.
    """
    # compute inter-eventtimes
    tau = None
    tau = torch.zeros(len(t)+1, dtype=torch.float32)
    tau[0] = t[0]
    for i in range(len(t)-1):
        tau[i+1] = t[i+1] - t[i]
    tau[len(t)] = t_end - t[len(t)-1]


    return tau
