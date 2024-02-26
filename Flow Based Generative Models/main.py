import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.datasets import make_moons
from torch import Tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')
        
# Visualization functions
def plot_density(model, loader=[], batch_size=100, mesh_size=5., device="cpu"):
    """Plot the density of a normalizing flow model. If loader not empty, it plots also its data samples.

    Args:
        model: normalizing flow model. Flow or StackedFlows
        loader: loader containing data to plot. DataLoader
        bacth_size: discretization factor for the mesh. int
        mesh_size: range for the 2D mesh. float
    """
    with torch.no_grad():
        xx, yy = np.meshgrid(np.linspace(- mesh_size, mesh_size, num=batch_size), np.linspace(- mesh_size, mesh_size, num=batch_size))
        coords = np.stack((xx, yy), axis=2)
        coords_resh = coords.reshape([-1, 2])
        log_prob = np.zeros((batch_size**2))
        for i in range(0, batch_size**2, batch_size):
            data = torch.from_numpy(coords_resh[i:i+batch_size, :]).float().to(device)
            log_prob[i:i+batch_size] = model.log_prob(data.to(device)).cpu().detach().numpy()

        plt.scatter(coords_resh[:,0], coords_resh[:,1], c=np.exp(log_prob))
        plt.colorbar()
        for X in loader:
            plt.scatter(X[:,0], X[:,1], marker='x', c='orange', alpha=.05)
        plt.tight_layout()
        plt.show()


def plot_samples(model, num_samples=500, mesh_size=5.):
    """Plot samples from a normalizing flow model. Colors are selected according to the densities at the samples.

    Args:
        model: normalizing flow model. Flow or StackedFlows
        num_samples: number of samples to plot. int
        mesh_size: range for the 2D mesh. float
    """
    x, log_prob = model.rsample(batch_size=num_samples)
    x = x.cpu().detach().numpy()
    log_prob = log_prob.cpu().detach().numpy()
    plt.scatter(x[:,0], x[:,1], c=np.exp(log_prob))
    plt.xlim(-mesh_size, mesh_size)
    plt.ylim(-mesh_size, mesh_size)
    plt.show()

from flows.affine import Affine
from flows.radial import Radial
from flows.stacked_flow import StackedFlows
from flows.loss import likelihood

def train(model, dataset, batch_size=100, max_epochs=1000, frequency=250):
    """Train a normalizing flow model with maximum likelihood.

    Args:
        model: normalizing flow model. Flow or StackedFlows
        dataset: dataset containing data to fit. Dataset
        batch_size: number of samples per batch. int
        max_epochs: number of training epochs. int
        frequency: frequency for plotting density visualization. int
        
    Return:
        model: trained model. Flow or StackedFlows
        losses: loss evolution during training. list of floats
    """
    # Load dataset
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # Train model
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    for epoch in range(max_epochs + 1):
        total_loss = 0
        for batch_index, (X_train) in enumerate(train_loader):
            loss = likelihood(X_train, model, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)
        losses.append(total_loss)
        
        if epoch % frequency == 0:
            print(f"Epoch {epoch} -> loss: {total_loss:.2f}")
            plot_density(model, train_loader, device=device)
    
    return model, losses

class CircleGaussiansDataset(Dataset):
    """Create a 2D dataset with Gaussians on a circle.

    Args:
        n_gaussians: number of Gaussians. int
        n_samples: number of sample per Gaussian. int
        radius: radius of the circle where the Gaussian means lie. float
        varaince: varaince of the gaussians. float
        seed: random seed: int
    """
    def __init__(
        self,
        n_gaussians: int = 6,
        n_samples: int = 100,
        radius: float = 3.,
        variance: float = .3,
        seed: int = 0
    ):
        self.n_gaussians = n_gaussians
        self.n_samples = n_samples
        self.radius = radius
        self.variance = variance

        np.random.seed(seed)
        radial_pos = np.linspace(0, np.pi*2, num=n_gaussians, endpoint=False)
        mean_pos = radius * np.column_stack((np.sin(radial_pos), np.cos(radial_pos)))
        samples = []
        for _, mean in enumerate(mean_pos):
            sampled_points = mean[:,None] + (np.random.normal(loc=0, scale=variance, size=n_samples), np.random.normal(loc=0, scale=variance, size=n_samples ))
            samples.append(sampled_points)
        p = np.random.permutation(self.n_gaussians * self.n_samples)
        self.X = np.transpose(samples, (0, 2, 1)).reshape([-1,2])[p]

    def __len__(self) -> int:
        return self.n_gaussians * self.n_samples

    def __getitem__(self, item: int) -> Tensor:
        x = torch.from_numpy(self.X[item]).type(torch.FloatTensor)
        return x
    
dataset_1 = CircleGaussiansDataset(n_gaussians=1, n_samples=500)
plt.figure(figsize=(4, 4))
plt.scatter(dataset_1.X[:,0], dataset_1.X[:,1], alpha=.05, marker='x', c='C1')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()

transforms = [Affine()]
model = StackedFlows(transforms, base_dist='Normal', device=device).to(device)
model, losses = train(model, dataset_1, max_epochs=201)

# Plots
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plot_density(model, [], device=device)
plot_samples(model)

transforms = [Radial().get_inverse().to(device) for _ in range(4)]
model = StackedFlows(transforms, base_dist='Normal', device=device).to(device)
model, losses = train(model, dataset_1, max_epochs=501)

# Plots
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plot_density(model, [], device=device)

dataset_2 = CircleGaussiansDataset(n_gaussians=3, n_samples=400, variance=.4)
plt.figure(figsize=(4, 4))
plt.scatter(dataset_2.X[:,0], dataset_2.X[:,1], alpha=.05, marker='x', c='C1')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()

transforms = [Affine().to(device)]
model = StackedFlows(transforms, base_dist='Normal', device=device).to(device)
model, losses = train(model, dataset_2, max_epochs=201)

# Plots
plt.plot(losses, marker='*')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plot_density(model, [], device=device)
plot_samples(model)

transforms = [Radial().get_inverse() for _ in range(16)]
model = StackedFlows(transforms, base_dist='Normal', device=device).to(device)
model, losses = train(model, dataset_2, max_epochs=501, frequency=100)

# Plots
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plot_density(model, [], device=device)

class MoonsDataset(Dataset):
    """Create a 2D dataset with spirals.

    Args:
        n_samples: number of sample per spiral. int
        seed: random seed: int
    """
    def __init__(self, n_samples: int = 1200, seed: int = 0):

        self.n_samples = n_samples

        np.random.seed(seed)
        self.X, _ = make_moons(n_samples=n_samples, shuffle=True, noise=.05, random_state=None)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, item: int) -> Tensor:
        x = torch.from_numpy(self.X[item]).type(torch.FloatTensor)
        return x
    
dataset_3 = MoonsDataset()
plt.scatter(dataset_3.X[:,0], dataset_3.X[:,1], alpha=.05, marker='x', c='orange')
plt.xlim(-2.5, 3)
plt.ylim(-2.5, 3)
plt.show()

transforms = [Affine().to(device)]
model = StackedFlows(transforms, base_dist='Normal', device=device).to(device)
model, losses = train(model, dataset_3, max_epochs=500)

# Plots
plt.plot(losses, marker='*')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plot_density(model, [], device=device)
plot_samples(model)

transforms = [Radial().get_inverse().to(device) for _ in range(16)]
model = StackedFlows(transforms, base_dist='Normal', device=device).to(device)
model, losses = train(model, dataset_3, max_epochs=501, frequency=100)

# Plots
plt.plot(losses, marker='*')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plot_density(model, [], mesh_size=3, device=device)