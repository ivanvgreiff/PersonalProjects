import os, sys, importlib
import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree as jtr
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import dm_pix as pix
from typing import Sequence, Dict, Tuple
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist, cifar100 # type: ignore
import tensorflow_datasets as tfds
from itertools import cycle
from scipy.linalg import circulant
from datetime import datetime
import wandb
import gzip
import struct
from Commons import *

__all__=["Client", "get_datasets", "generate_local_epoch_distribution_JAX", "split_by_class_JAX", "sample_per_class_JAX", "generate_active_client_matrix_JAX",
         "allocate_client_datasets_JAX", "plot_data_distribution", "generate_client_class_map", "flatten_dict", "load_and_extract_configs",
         "flatten_dict", "log", "load_idx_gz", "get_run_name"]


class Client:
    # Shared class attributes initialized once
    data_shape = None
    test_data = None
    test_labels = None
    valid_data = None
    valid_labels = None 
    test_poisoned_idx = None

    def __init__(self, client_id: int, epochs: Sequence[int]=None, data: np.array=None, labels: np.array=None):
        self.client_id = client_id
        self.data = data if data is not None else np.empty(shape=(0, *Client.data_shape))
        self.labels = labels if labels is not None else np.empty(shape=(0,), dtype=np.int8)
        self.epochs = epochs
        self.poisoned_idx = []
        
        self.class_map = np.empty(shape=(0,), dtype=np.int8)
        self.active = np.empty(shape=(0,), dtype=np.bool_)
        self.state_train = None
        self.state_opt = None
        
def get_datasets(JKEY, name_dataset:str = 'mnist' ) -> Tuple[np.ndarray, np.ndarray]:
    
    def normalize(img, mean, std):
        img =  img / 255.0
        img = (img - mean) / std
        return img

                # Function to augment images
    def augment(JKEY, img):
        img = pix.pad_to_size(img, 40, 40)
        img = pix.random_crop(JKEY, img, (32, 32, 3))
        img = pix.random_flip_left_right(JKEY, img)
        return img
    
    print(f'Loading {name_dataset}...')
    match str.lower(name_dataset):
        case 'emnist':
            root = f"/Users/ata/Desktop/HiWi/data/emnist"
            train_images_path = "emnist-digits-train-images-idx3-ubyte.gz"
            train_labels_path = "emnist-digits-train-labels-idx1-ubyte.gz"
            test_images_path = "emnist-digits-test-images-idx3-ubyte.gz"
            test_labels_path = "emnist-digits-test-labels-idx1-ubyte.gz"

            train_data = load_idx_gz(root + "/" + train_images_path)
            train_labels = load_idx_gz(root + "/" + train_labels_path)
            test_data = load_idx_gz(root + "/" + test_images_path)
            test_labels = load_idx_gz(root + "/" + test_labels_path)

            data_shape = (28, 28, 1)
            data_mean, data_std = (0.1307, 0.3081)
            train_data, test_data = np.expand_dims(train_data, axis=-1), np.expand_dims(test_data, axis=-1)
            train_ds, test_ds = {'image': train_data, 'label': np.ravel(train_labels)}, {'image': test_data, 'label': np.ravel(test_labels)}
            train_ds['image'] = jnp.asarray(jax.vmap(normalize, in_axes=(0, None, None))(train_ds['image'], data_mean, data_std))
            test_ds['image'] = jnp.asarray(jax.vmap(normalize, in_axes=(0, None, None))(test_ds['image'], data_mean, data_std))
            train_ds["label"], test_ds["label"] = jnp.asarray(train_ds["label"]), jnp.asarray(test_ds["label"])
        case 'mnist':
            data_shape = (28, 28, 1)
            data_mean, data_std = (0.1307, 0.3081)
            (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
            train_data, test_data = jnp.expand_dims(train_data, axis=-1), jnp.expand_dims(test_data, axis=-1)
            train_ds, test_ds = {'image': train_data, 'label': jnp.ravel(train_labels)}, {'image': test_data, 'label': jnp.ravel(test_labels)}
            train_ds['image'] = jnp.asarray(jax.vmap(normalize, in_axes=(0, None, None))(train_ds['image'], data_mean, data_std))
            test_ds['image'] = jnp.asarray(jax.vmap(normalize, in_axes=(0, None, None))(test_ds['image'], data_mean, data_std))
            
        case 'fashion-mnist':
            data_shape = (28, 28, 1)
            data_mean, data_std = (0.2859, 0.3530)
            (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
            train_data, test_data = np.expand_dims(train_data, axis=-1), np.expand_dims(test_data, axis=-1)
            train_ds, test_ds = {'image': train_data, 'label': np.ravel(train_labels)}, {'image': test_data, 'label': np.ravel(test_labels)}     
            train_ds['image'] = np.asarray(jax.vmap(normalize, in_axes=(0, None, None))(train_ds['image'], data_mean, data_std))
            test_ds['image'] = np.asarray(jax.vmap(normalize, in_axes=(0, None, None))(test_ds['image'], data_mean, data_std))
            
        case 'cifar10':
            data_shape = (32, 32, 3)
            (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
            train_ds, test_ds = {'image': train_data, 'label': np.ravel(train_labels)}, {'image': test_data, 'label': np.ravel(test_labels)}
            data_mean, data_std = np.array([0.491, 0.482, 0.447]), np.array([0.247, 0.243, 0.262])
            train_ds['image'] = np.asarray(jax.vmap(normalize, in_axes=(0, None, None))(train_ds['image'], data_mean, data_std))
            train_ds['image'] = np.asarray(jax.vmap(augment, in_axes=(0, 0))(jax.random.split(JKEY, len(train_ds['image'])), train_ds['image']))
            test_ds['image'] = np.asarray(jax.vmap(normalize, in_axes=(0, None, None))(test_ds['image'], data_mean, data_std))
        case _:
            raise ValueError(f'{name_dataset} is not a valid dataset')
    return train_ds, test_ds, data_shape

def generate_local_epoch_distribution_JAX(JKEY: jax.random.PRNGKey, clients: Sequence[Client], type_epoch:str, random:bool, max_epoch:int, 
                                      mean: float, std: float, beta:float, coefficient: int) -> None:
    match type_epoch:
        case 'constant':
            local_epoch_distribution = np.full((len(clients), ), mean)
            #done
        case 'uniform':
            local_epoch_distribution = jr.uniform(JKEY, (len(clients), ), minval=mean-std, maxval=mean+std)
            #done
        case 'gaussian':
            if random:
                distribution_mean = np.repeat(np.round(jr.dirichlet(JKEY, np.full((client, ), beta)) * coefficient).reshape(-1, len(clients)), max_epoch, axis=0)
                local_epoch_distribution = np.maximum(distribution_mean + std * jr.normal(JKEY, (max_epoch, len(clients))), 1)
            else:
                local_epoch_distribution = np.maximum(mean + std * jr.normal(JKEY, (len(clients), )), 1)
            #done
        case 'exponential':
            if random:
                distribution_mean = np.repeat(np.round(jr.dirichlet(JKEY, np.full((client, ), beta)) * coefficient).reshape(-1, len(clients)), max_epoch, axis=0)
                local_epoch_distribution = np.maximum(distribution_mean + std * np.round(jr.exponential(JKEY, (max_epoch, len(clients)))), 1)
            else:
                local_epoch_distribution = np.maximum(jr.exponential(JKEY, mean, size=len(clients)).round(), 1)
            #done
        case 'dirichlet':
            local_epoch_distribution = np.round(jr.dirichlet(np.full((len(clients), ), beta)) * coefficient)
            #done
        case _:
            raise ValueError("Invalid iteration type")
            #done
    print(f"The local iteration distribution is {local_epoch_distribution.shape} shaped")
    if local_epoch_distribution.ndim == 1:
        local_epoch_distribution = np.repeat(local_epoch_distribution.reshape(-1, len(clients)), max_epoch, axis=0)
        print(f"The local iteration distribution is expanded to {local_epoch_distribution.shape}")
    
    for client in clients:
        client.epochs = np.int16(local_epoch_distribution[:, client.client_id])
        
def split_by_class_JAX(data: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    classes = np.unique(labels)
    classed_data = {clss: data[labels == clss, :, :] for clss in classes}
    return classed_data

def sample_per_class_JAX(JKEY: jax.random.PRNGKey, data_shape: Tuple[int], num_samples: int, classed_data_and_label: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    valid_data = np.empty(shape=(0, *data_shape))
    valid_labels = np.empty(shape=(0, ), dtype=np.int8)
    for clss, data in classed_data_and_label.items():
        JKEY, _ = jr.split(JKEY)
        sampled_indices = jr.choice(JKEY, jnp.arange(len(classed_data_and_label[clss])), (num_samples, ), replace=False)
        valid_data = np.concatenate([valid_data, data[sampled_indices]], axis=0)
        valid_labels = np.concatenate([valid_labels, np.full(len(sampled_indices), clss)], axis=0)
        jnp.delete(classed_data_and_label[clss], sampled_indices, axis=0)

    return valid_data, valid_labels

def sample_data_per_class(sample_number: int, RNG: np.random.Generator, split_data: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    sampled_split_data = {}
    for label, val in split_data.items():
        indices = RNG.choice(len(val), sample_number, replace=False)
        sampled_split_data[label] = val[indices]
        split_data[label] = np.delete(split_data[label], indices, axis=0)
    return sampled_split_data

def generate_client_class_map(num_clients: int, num_class: int, num_class_per_client: int = 2) -> Sequence[Sequence[int]]:
    if num_class_per_client > num_class:
        raise ValueError("The number of classes per client is greater than the total number of classes.")
    all_classes = list(range(num_class))
    class_cycle = cycle(all_classes)
    client_class_assignments = [[next(class_cycle) for _ in range(num_class_per_client)] for i in range(num_clients)]
    return client_class_assignments

def allocate_client_datasets_JAX(JKEY: jax.random.PRNGKey, clients: Sequence[Client],  allocation_type: str, class_ratio: int,
                              num_classes: int, classed_data: Dict[int, np.ndarray], beta: float, data_shape: Tuple[int, int, int]) -> None:
    total_dropped_samples = 0
    match allocation_type:
        case "class-clients": # Each client has data from a single class
            if len(clients) > len(classed_data):
                raise ValueError("The number of clients is greater than the number of classes")
            for idx, client in enumerate(clients):
                client.data = np.concatenate([client.data, classed_data[idx]], axis=0)
                client.labels= np.concatenate([client.labels, np.full((len(classed_data[idx]), ), idx, dtype=np.int8)])
                client.class_map = np.concatenate([client.class_map, np.array([idx])])
        
        case "n-class-clients": # Each client has data from only n classes
            num_classes_per_client = class_ratio
            class_map = generate_client_class_map(len(clients), num_classes, num_classes_per_client)
            print(class_map)
            for idx, client in enumerate(clients):
                client.class_map = class_map[idx]
            
            for clss, data in classed_data.items():
                num_samples = len(data)
                class_length = num_samples // (num_classes_per_client)
                excess_samples = num_samples % ((len(clients) // num_classes) * num_classes_per_client)
                class_data = data[:-excess_samples] if excess_samples else data[:]
                class_data =  class_data.reshape((len(clients) // num_classes) * num_classes_per_client, -1, *data_shape)
                    
                for idx, client in enumerate(clients):
                    if clss in client.class_map:
                        client.data = np.concatenate((client.data, class_data[0]), axis=0)
                        labels = np.full(len(class_data[0]), clss)
                        class_data = class_data[1:]
                        client.labels = np.concatenate((client.labels, labels), axis=0)
                        # done with this case
        case "1/n-class-clients": # Each client has data from n classes with one class being dominant, n = class_ratio
            # This doesn't work when the number of clients is higher than a certain number
            num_classes_per_client = len(clients) * class_ratio
            allocation_map = np.arange(len(clients) + class_ratio - 1, num_classes_per_client, class_ratio - 1, dtype=np.int32)
            allocation_order = circulant(np.arange(len(clients), dtype=np.int8)).T
            allocation_row = 0
            # print(allocation_map, '\n', allocation_order)   
            for clss, data in classed_data.items():
                num_samples = len(data)
                class_length = num_samples // (num_classes_per_client)
                excess_samples = num_samples % num_classes_per_client
                total_dropped_samples += excess_samples
                classed_data = data[:-excess_samples, :, :].reshape((num_classes_per_client), class_length, *data_shape)
                split_classed_data = np.split(classed_data, allocation_map, axis=0)
        
                for client in range(len(clients)):
                    client_idx = allocation_order[allocation_row, client]
                    temp = split_classed_data[client].reshape(-1, *data_shape)
                    clients[client_idx].data = np.concatenate((clients[client_idx].data, temp), axis=0)
                    clients[client_idx].labels = np.concatenate((clients[client_idx].labels, np.full(len(temp), clss, dtype=np.int8)), axis=0)
                    clients[client_idx].class_map = np.concatenate((clients[client_idx].class_map, np.array([clss])), axis=0)
                allocation_row += 1
            # done with this case
        
        case "uniform":
            num_classes_per_client = len(clients)
            for clss, data in classed_data.items():
                num_samples = len(data)
                class_length = num_samples // (num_classes_per_client)
                excess_samples = num_samples % num_classes_per_client
                total_dropped_samples += excess_samples
                uniformly_classed_data = data[:-excess_samples, :, :] if excess_samples != 0 else data .reshape((num_classes_per_client), class_length, *data_shape)
                uniformly_classed_data = uniformly_classed_data.reshape((num_classes_per_client), class_length, *data_shape)
                # print(uniformly_classed_data.shape)
                for idx, client in enumerate(clients):
                    client.data = jnp.concatenate((client.data, uniformly_classed_data[idx]), axis=0)
                    client.labels = jnp.concatenate((client.labels, np.full(len(uniformly_classed_data[idx]), clss, dtype=np.int8)), axis=0)
                    client.class_map = jnp.concatenate((client.class_map, np.array([clss])), axis=0)
            # done with this case
        
        case "random":
            for label, data in classed_data.items():
                JKEY, _ = jr.split(JKEY)
                allocation_map = np.sort(jr.randint(JKEY, (len(clients) - 1, ), 0, len(data)))
                randomly_classed_data = np.split(data, allocation_map, axis=0)

                for idx, client in enumerate(clients):
                    client.data = np.concatenate((client.data, randomly_classed_data[idx]), axis=0)
                    client.labels = np.concatenate((client.labels, np.full(len(randomly_classed_data[idx]), label, dtype=np.int8)), axis=0)
                    client.class_map = np.concatenate((client.class_map, np.array([label])), axis=0)
            # done with this case
        
        case "dirichlet":
            sample_rate = jr.dirichlet(JKEY, np.full(len(clients), beta), shape=(len(classed_data), ))
            for label, data in classed_data.items():
                allocation_map = np.cumsum(np.int32(sample_rate[label] * len(data)))[:-1]
                dirichlet_classed_data = np.split(data, allocation_map, axis=0)
                
                for idx, client in enumerate(clients):
                    client.data = np.concatenate((client.data, dirichlet_classed_data[idx]), axis=0)
                    client.labels = np.concatenate((client.labels, np.full(len(dirichlet_classed_data[idx]), label, dtype=np.int8)), axis=0)
                    client.class_map = np.concatenate((client.class_map, np.array([label])), axis=0)
                    
            #done with this case
        case _:
            raise ValueError("Invalid allocation type")
        
    print(f"The number of dropped samples is equal to {total_dropped_samples}")
    if allocation_type != "class-clients":
        # client_data, client_labels = flatten_nested_lists(client_data), flatten_nested_lists(client_labels)
        for idx, client in enumerate(clients):
            JKEY, _ = jr.split(JKEY)
            randomized_idx = jnp.arange(len(client.data))
            randomized_idx = jr.permutation(JKEY, randomized_idx)
            # print(len(randomized_idx), client.labels.shape)
            client.data, client.labels = client.data[randomized_idx], client.labels[randomized_idx]

def plot_data_distribution(worker_number: int, worker_labels: list[np.ndarray], run_name:str):
    if worker_number <= 20:
        plt.figure(1, figsize=(int(8 * worker_number / 10), 8))
    else:
        plt.figure(1, figsize=(int(4 * worker_number / 10), 16))
    for worker in range(len(worker_labels)):
        if worker_number <= 20:
            plt.subplot(5, int(np.ceil(worker_number / 5)), worker + 1)
        else:
            plt.subplot(10, int(np.ceil(worker_number / 10)), worker + 1)
        plt.hist(worker_labels[worker], color="lightblue", ec="red", align="left", bins=np.arange(11))
        plt.title("worker " + str(worker + 1))
    plt.suptitle(run_name)
    plt.tight_layout()
    
    logs_dir = os.path.join(os.path.dirname(__file__), 'Logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    plt.savefig(os.path.join(logs_dir, f"{run_name}_Data_Distribution.png"))
    plt.close()
 
def generate_active_client_matrix_JAX(JKEY: jax.random.PRNGKey, clients: Sequence[Client], active_probability: float, max_epoch: int) -> None:
    if active_probability < 0  or 1 < active_probability:
        raise ValueError("The probability of inactivity must be between 0 and 1")
    for client in clients:
        client.active = jr.choice(JKEY, np.array([False, True]), (max_epoch, ), True, np.array([active_probability, 1-active_probability]))

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key.upper(), sep=sep).items())
        else:
            items.append((new_key.upper(), v))
    return dict(items)

def load_and_extract_configs(dir_path, include=None, exclude=None):
    """
    Load Python modules dynamically from a specified directory and extract a 'config' variable from each,
    with options to include specific modules or exclude specified modules.

    Parameters:
    - dir_path (str): The local path to the directory containing Python modules.
    - include (list): Optional list of module filenames to specifically include (without '.py' extension).
    - exclude (list): Optional list of module filenames to exclude (without '.py' extension).

    Returns:
    - list: A list of all the 'config' variables extracted from the modules.
    """
    if exclude is None:
        exclude = []

    curr_path = os.path.dirname(os.path.realpath(__file__))
    param_files = os.listdir(os.path.join(curr_path, dir_path))

    if include is not None:
        param_files = [f.split('.')[0] for f in param_files if f.split('.')[0] in include]
    else:
        param_files = [f.split('.')[0] for f in param_files if (f.endswith('.py') and f.split('.')[0] not in exclude)]

    configs = []

    if param_files:
        for f in param_files:
            module_name = f"{dir_path.replace('/', '.')}.{f}"
            try:
                module = importlib.import_module(module_name)
                print(f"Loading Configs from {f}.py...")
                config = getattr(module, 'config')
                configs.append(config)
                print("Done.")
            except AttributeError as e:
                print(f"Could not find 'config' in {f}.py. Skipping.")
                print(f"The error is {e}")
            except ImportError as e:
                print(f"Failed to import {f}.py: {str(e)}")
            finally:
                if module_name in sys.modules:
                    del sys.modules[module_name]
    else:
        print("No Python files found based on the include/exclude criteria.")

    return configs

def log(row:pd.Series, step:int, logees:dict[str, any]):
    for key, data in logees.items():
        if isinstance(data, list):
            for idx, datum in enumerate(data):
                wandb.log(data = {key: np.mean(datum)}) # , step = step + idx)
        else:
            wandb.log(data = {key: data}) # , step = step)
    return pd.Series(logees) if row.empty \
    else pd.concat([row, pd.Series(logees)], axis=0)

def load_idx_gz(filename):
    """Loads an IDX file (gzipped) into a NumPy array."""
    with gzip.open(filename, 'rb') as f:
        magic, num_items = struct.unpack(">II", f.read(8))  # Read first 8 bytes
        if magic == 2051:  # Image file
            rows, cols = struct.unpack(">II", f.read(8))  # Read image dimensions
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, rows, cols)
        elif magic == 2049:  # Label file
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"Unknown IDX file with magic number {magic}")
    return data

def get_run_name(config, run):
    name = ""
    
    name += f"JAX|"
    
    #* Model
    name += f"{config.train.model.__name__.lower()}|"
    #* Data
    name += f"{config.data.name}|"
    name += f"BS-{config.data.batch_size}|"
    
    #* Server
    name += f"SNE-{config.server.num_epochs}|"
    name += f"STX-{config.server.tx.fn.__name__}|"
    name += f"SLR-{config.server.tx.lr}|"
    # name += f"SLRDR-{config.worker.tx.lr_decay_per_server_epoch}|"
    
    #* Worker
    name += f"WNE-{config.worker.epoch.mean}|"
    name += f"WN-{config.worker.num}|"
    name += f"WTX-{config.worker.tx.fn.__name__}|"
    name += f"WLR-{config.worker.tx.lr}|"
    # name += f"WLRDR-{config.worker.tx.lr_decay_per_worker_epoch}|"
    
    # #* Compressor
    # if config.compressor.enable:
    #     name += f"C-{config.compressor.enable}|"
    #     name += f"CS2W-{config.compressor.s2w.enable}|"
    #     name += f"CS2W-{config.compressor.s2w.comp.__name__}|"
    #     name += f"W2S-{config.compressor.w2s.enable}|"
    #     name += f"W2S-{config.compressor.w2s.comp.__name__}|"
    
    #* Run
    name += f"R#{run}|"
    name += datetime.now().strftime("%H%M%S")
    return name

