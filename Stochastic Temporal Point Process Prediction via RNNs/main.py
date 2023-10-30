import torch
import tqdm
import matplotlib.pyplot as plt
# import utility functions
from tpp.utils import get_tau, get_sequence_batch
from tpp.model import NeuralTPP



# load toy data
data = torch.load("data/hawkes.pkl")

arrival_times = data["arrival_times"]
t_end = data["t_end"]

# compute interevent times and batch sequences
tau = [get_tau(t, t_end) for t in arrival_times]
times, mask = get_sequence_batch(arrival_times)

# normalize inter event times [0,1]
times = times/t_end



model = NeuralTPP(hidden_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

losses = []
epochs = 5000

with tqdm.tqdm(range(1, epochs), unit="epoch") as tepoch:
    for epoch in tepoch:
        optimizer.zero_grad()
        loss = model.log_likelihood(times, mask)
        loss = -loss.mean()
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        tepoch.set_postfix(NLL=loss.item())

plt.plot(range(1, epochs), losses)
plt.ylabel("NLL")
plt.xlabel("Epoch")
plt.show()