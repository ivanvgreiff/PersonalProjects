import torch
import numpy as np
"""
t = torch.ones(5)
t_end = 10
tau = torch.zeros(len(t)+1, dtype=torch.float32)
tau[0] = t[0]
for i in range(len(t)-1):
    print(i)
    tau[i+1] = t[i+1] - t[i]
tau[len(t)-1] = t_end - t[len(t)-2]
"""
seq1 = torch.zeros(3)
seq2 = torch.ones(6)
inter_event_times = [seq1, seq2]
print(inter_event_times)

batch_size = len(inter_event_times)
length_seq = torch.zeros(batch_size)

a = 0
for seq in inter_event_times:
    length_seq[a] = len(seq)
    a += 1
max_seq_length = max(length_seq).int() #.item() returns float
print(max_seq_length)

batch = torch.zeros([batch_size, max_seq_length])
mask = torch.zeros([batch_size, max_seq_length], dtype=torch.bool)
print(batch)
print(mask)


hidden_dim=7
batch_size = 4
seq_length = 10
history_emb = torch.randn([batch_size, seq_length, hidden_dim])
linear = torch.nn.Linear(in_features=hidden_dim, out_features=2)

seq_length = len(history_emb[0, :, 0])
mu = torch.zeros([batch_size, seq_length])
sigma = torch.zeros([batch_size, seq_length])
      
for i in range(seq_length):
    output = linear(history_emb[:, i, :])
    print(output)
    mu[:, i] = output[:, 0]
    sigma[:, i] = output[:, 1]


mu = torch.randn([batch_size, seq_length])
sigma = torch.randn([batch_size, seq_length])
distribution = torch.distributions.LogNormal(mu, sigma)

times = torch.randn([batch_size, seq_length])
print(times)
print(1)
log_surv_last = torch.zeros([batch_size])
log_surv = torch.log(1-distribution.cdf(times))
xrange = len(log_surv[0, :])
yrange = len(log_surv[:, 0])

count1 = 0
for i in range(xrange):
    for j in range(yrange):
        if mask[i, j] == True:
            count1 += 1
        else:
            log_surv_last[i] = log_surv[i, count1]
            break
print(log_surv_last)