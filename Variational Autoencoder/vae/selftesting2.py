import torch
import torch.nn as nn
import torch.nn.functional as F


mu = torch.randn([5, 7])
logsigma = torch.randn([5, 7])
batch_size = len(mu[:, 0])
input_dim = 10
latent_dim = 7

x = torch.ones([batch_size, input_dim])
theta = torch.zeros([batch_size, input_dim])
BCE = torch.zeros([batch_size])
for i in range(batch_size):
    BCE[i]  = (F.binary_cross_entropy(theta[i, :], x[i, :], reduction='mean'))
print(BCE)

