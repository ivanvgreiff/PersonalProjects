# ZK Cryptographic Federated Learning

## Overview
This project implements a privacy-preserving, cryptographically enhanced Federated Learning (FL) framework using JAX, Flax, and Optax. It supports multiple neural network architectures (LeNet, CNN, ResNet) and simulates a federated environment with configurable data distributions, client activity, and quantization for communication efficiency. The project is designed for research and experimentation in distributed, privacy-aware machine learning.

---

## Mathematical and Technical Background

### Federated Learning (FL)
Federated Learning is a distributed machine learning paradigm where multiple clients (workers) collaboratively train a model under the orchestration of a central server, while keeping their data local. The server aggregates model updates from clients, improving privacy and reducing data transfer.

#### Federated Optimization Objective
The global objective in FL is to minimize the sum of local objectives:
$$
\min_{\theta} F(\theta) = \sum_{k=1}^K p_k F_k(\theta)
$$
where:
- $ K $: Number of clients
- $ p_k = \frac{n_k}{n} $: Proportion of data on client $ k $
- $ F_k(\theta) = \frac{1}{n_k} \sum_{i=1}^{n_k} \ell(\theta; x_i, y_i) $: Local loss on client $ k $
- $ \ell $: Loss function (e.g., cross-entropy)

#### Federated Averaging (FedAvg)
At each round (server epoch):
1. The server sends the current model parameters $ \theta $ to all clients.
2. Each client updates $ \theta $ using its local data for several epochs, producing $ \theta_k $.
3. The server aggregates the updates, typically by weighted averaging:
   $$
   \theta_{t+1} = \sum_{k=1}^K p_k \theta_k
   $$

#### Quantization for Communication Efficiency
To reduce communication cost, model updates are quantized before being sent to the server. This project implements **stochastic quantization**:
- For a value $ x \in [0, 1] $ and quantization level $ q $,
- Let $ \lfloor qx \rfloor $ be the lower quantization bin, and $ \alpha = qx - \lfloor qx \rfloor $ the fractional part.
- With probability $ \alpha $, round up; otherwise, round down:
$$
Q(x) = \begin{cases}
\frac{\lfloor qx \rfloor + 1}{q} & \text{with probability } \alpha \\
\frac{\lfloor qx \rfloor}{q} & \text{with probability } 1-\alpha
\end{cases}
$$
This preserves the expectation: $ \mathbb{E}[Q(x)] = x $.

#### Dirichlet Data Partitioning (Non-IID Data)
To simulate realistic, non-IID data distributions across clients, the project supports Dirichlet partitioning:
- For $ K $ clients and $ C $ classes, draw proportions $ \mathbf{p}_k \sim \text{Dirichlet}(\alpha) $ for each class.
- Assign data to clients according to these proportions, controlling heterogeneity with $ \alpha $:
  - Small $ \alpha $: Highly non-IID (clients see few classes)
  - Large $ \alpha $: Nearly IID

#### Client Activity and Dropout
Client activity is modeled as a Bernoulli process:
$$
A_{k,t} \sim \text{Bernoulli}(1 - p_{\text{inact}})
$$
where $ A_{k,t} = 1 $ if client $ k $ is active at round $ t $.

---

## Cryptographic Details: Secure Aggregation and Quantization

### Secure Aggregation (Conceptual)
While this codebase does not implement a full cryptographic protocol, it is designed to be extensible for secure aggregation, which is crucial for privacy in FL. The main idea is:
- Each client masks its update with random values (additive secret sharing) so that the server cannot see individual updates, only their sum.
- The server receives masked updates and, after all masks cancel out, recovers the aggregate update.

**Mathematical Formulation:**
- Each client $ k $ computes update $ \Delta_k $ and mask $ r_k $.
- Sends $ \Delta_k + r_k $ to the server.
- Masks are constructed so that $ \sum_k r_k = 0 $, thus:
$$
\sum_k (\Delta_k + r_k) = \sum_k \Delta_k
$$

### Quantization with Modular Arithmetic (for Secure Aggregation)
- Quantized updates can be mapped to a finite field (e.g., modulo a prime $ p $), enabling cryptographic protocols:
$$
Q(x) = \text{Quantize}(x) \mod p
$$
- This is useful for protocols like [Bonawitz et al., 2017](https://arxiv.org/abs/1611.04482), where aggregation is performed in a finite field.

### Zero-Knowledge Proofs (ZK) (Research Direction)
- The project is structured to allow future integration of ZK proofs, where clients can prove properties about their updates (e.g., correct computation, bounded norm) without revealing the updates themselves.
- ZK proofs are not yet implemented (in progress), but the quantization and modular arithmetic steps are compatible with such extensions.

---

## Directory Structure

```
ZK Cryptographic Federated Learning/
├── Main.py                # Main entry point, orchestrates FL workflow
├── Training.py            # Client/server training logic, quantization
├── Utils.py               # Data loading, client management, helper functions
├── Commons.py             # Shared types, config, and constants
├── Models/                # Model architectures (LeNet, CNN, ResNet)
│   ├── lenet.py
│   ├── cnn.py
│   ├── resnet.py
│   └── __init__.py
├── Params/                # Experiment configuration files
│   └── Parameters_1.py
├── Logs/                  # Output logs and data distribution plots
│   └── _Data_Distribution.png
├── wandb/                 # Weights & Biases experiment tracking
├── Playground.ipynb       # Interactive notebook for experimentation
└── ...
```

---

## Component Details

### 1. Main Workflow (`Main.py`)
- Loads experiment configurations from `Params/`.
- Initializes random seeds, datasets, and clients.
- Distributes data among clients according to configuration (IID, non-IID, Dirichlet, etc.).
- For each server epoch:
  - Selects active clients (can simulate client dropout/inactivity).
  - Each active client trains locally for a configurable number of epochs.
  - Aggregates client updates (FedAvg or other schemes).
  - Optionally applies quantization to updates.
  - Evaluates and logs validation/test metrics.
- Logs results to Weights & Biases (wandb) and saves CSV logs.

### 2. Models (`Models/`)
- **LeNet5 (`lenet.py`)**: Classic CNN for digit recognition. Layers: Conv → ReLU → Pool → Conv → ReLU → Pool → Dense → ReLU → Dense → ReLU → Dense.
- **CNNs (`cnn.py`)**: Several deeper CNNs for more complex datasets (e.g., CIFAR-10). Includes variants with 4, 6, or 10 dense layers.
- **ResNet (`resnet.py`)**: Modern deep residual networks (ResNet18, 34, 50, etc.) with skip connections, batch normalization, and flexible depth.
- All models are implemented using Flax's `nn.Module` and are selected via the `get_model` function in `Models/__init__.py`.

### 3. Training Logic (`Training.py`)
- **train_client**: Each client receives the global model, trains locally, and returns updated parameters (optionally quantized).
- **test_client**: Evaluates a model on test/validation data.
- **Quantization**: Implements stochastic quantization for communication-efficient FL. Quantizes gradients/updates to a fixed number of levels, optionally modulo a prime (for secure aggregation research).
- **JAX JIT**: Training and update steps are JIT-compiled for speed.

### 4. Utilities (`Utils.py`)
- **Client Class**: Represents a federated client, holding local data, labels, and state.
- **get_datasets**: Loads and normalizes datasets (MNIST, CIFAR-10, Fashion-MNIST, EMNIST, etc.).
- **Data Distribution**: Functions to split data by class, sample validation sets, and allocate data to clients (supports various non-IID schemes).
- **Client Activity**: Simulates client dropout/inactivity per epoch.
- **Logging and Plotting**: Logs metrics, plots data distributions, and manages experiment metadata.

### 5. Configuration (`Params/Parameters_1.py`)
- Uses `ml_collections.ConfigDict` for hierarchical, flexible configuration.
- Controls all aspects: model, optimizer, data, server/client settings, quantization, poisoning, etc.
- Example settings:
  - `config.train.model`: Model architecture (e.g., LeNet5)
  - `config.data.name`: Dataset name
  - `config.server.num_epochs`: Number of global rounds
  - `config.worker.num`: Number of clients
  - `config.quantization.levels`: Quantization granularity

### 6. Experiment Tracking (`wandb/`)
- Integrates with [Weights & Biases](https://wandb.ai/) for experiment tracking, logging, and visualization.
- Each run logs metrics, configuration, and outputs for reproducibility.

### 7. Playground (`Playground.ipynb`)
- Jupyter notebook for interactive experimentation, visualization, and debugging.
- Walks through the main workflow, data loading, client setup, and model training.

---

## How Components Interact

- **Main.py** orchestrates the entire workflow, calling utility functions for data and client setup, and invoking training logic for each round.
- **Models** are selected and instantiated based on configuration, and passed to training functions.
- **Training.py** handles all local and global training logic, including quantization and evaluation.
- **Utils.py** provides all data handling, client management, and logging utilities.
- **Parameters_1.py** (and other config files) allow easy switching between experimental setups.
- **wandb** logs all relevant metrics and artifacts for later analysis.

---

## Running the Project

### 1. Install Dependencies
Install all required packages (see `wandb/run-*/files/requirements.txt` for full list):
```bash
pip install -r wandb/run-20250216_221830-67fnv1p9/files/requirements.txt
```

### 2. Run Main Experiment
```bash
python Main.py
```

### 3. Experiment with Notebook
Open `Playground.ipynb` in Jupyter for interactive exploration:
```bash
jupyter notebook Playground.ipynb
```

---

## Key Dependencies
- JAX, Flax, Optax (core ML/optimization)
- TensorFlow, TensorFlow Datasets (data loading)
- ml_collections (configuration)
- Weights & Biases (experiment tracking)
- NumPy, Pandas, Matplotlib, SciPy (data and visualization)

---

## References
- [Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/abs/1610.05492)
- [Practical Secure Aggregation for Privacy-Preserving Machine Learning](https://arxiv.org/abs/1611.04482)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Optax Documentation](https://optax.readthedocs.io/)
- [Weights & Biases](https://wandb.ai/)
- [Zero-Knowledge Proofs: An Illustrated Primer](https://blog.goodaudience.com/zero-knowledge-proofs-an-illustrated-primer-7c0f5fd6878a)

---

## Acknowledgements
Developed by Ivan von Greiff and Ata Shaker for the "Coding for Private Reliable and Efficient Distributed Learning" course, Winter Semester 2024. 