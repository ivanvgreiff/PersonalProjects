import torch
from torchvision import datasets, transforms
import numpy as np
from ddpm import DDPM
from visualization import visualize_dataset_mnist, visualize_mnist_samples


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")


batch_size = 128
tfs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_data = datasets.MNIST("./data", train=True, download=True, transform=tfs)
test_data = datasets.MNIST("./data", train=False, download=True, transform=tfs)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=device.type == "cuda")
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=device.type == "cuda")


visualize_dataset_mnist(train_data);


max_epochs = 20
display_step = 100

ddpm = DDPM(N=1000, type="resnet", hidden_dim=16, n_layers=2).to(device)
opt = torch.optim.Adam(ddpm.parameters())


for epoch in range(max_epochs):
    print(f"Epoch {epoch}")
    losses = []
    for ix, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        loss = ddpm.loss(x)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if ix % display_step == 0:
            print(f"  loss = {loss.item():.2f}")
    print(f"  => mean(loss) = {np.mean(losses):.2f}")


samples = ddpm.sample(20, device)


visualize_mnist_samples(samples);