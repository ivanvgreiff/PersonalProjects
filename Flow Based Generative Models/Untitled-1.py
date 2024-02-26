        
import torch

x = torch.tensor([0, 0, 0])

shift = torch.tensor([0, 0, 1])

y = torch.exp(x) + shift

print(y)
