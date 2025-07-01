import os
import jax
from jax import numpy as jnp
from jax import random as jr
import tensorflow as tf
import flax
from flax import linen as nn
from flax.training import train_state
import optax
from datetime import datetime
import numpy as np
import pandas as pd
import ml_collections
import wandb

from Utils import *
from Training import *
from Models import *


def main(config: ml_collections.ConfigDict, log_df: pd.DataFrame):

    wandb.login()
    run = wandb.init(project=config.wandb.project, job_type=config.wandb.job_type,
                    config=config.to_dict(), name=config.wandb.name)
    
    tf.config.set_visible_devices([], 'GPU')
    
    # run = wandb.init(dir="/nas/lnt/stud/ge27yuv/", project=config.wandb.project,
    #                  job_type=config.wandb.job_type, name=config.wandb.name, config=logged_config)

    RNG = np.random.default_rng(seed=config.seed)
    JKEY = jr.PRNGKey(seed=config.seed)
    JKEY, data_key = jr.split(JKEY, 2)

    train_ds, test_ds, data_shape = get_datasets(data_key, config.data.name)
    config.data.shape = data_shape
    print('Data is of shape:', config.data.shape)
    train_images, train_labels = train_ds['image'], train_ds['label']
    test_images, test_labels = test_ds['image'], test_ds['label']
    print(f"train_images is of shape: {train_images.shape} and of type {type(train_images)}")
    print(f"train_labels is of shape: {train_labels.shape} and of type {type(train_labels)})")

    Client.data_shape = config.data.shape
    clients = [Client(client_id) for client_id in range(config.worker.num)]
    num_classes = len(np.unique(train_labels))
    config.data.num_classes = num_classes

    JKEY, epochs_key, validation_key, training_key, activity_key = jr.split(JKEY, 5)
    generate_local_epoch_distribution_JAX(epochs_key, clients, config.worker.epoch.type, config.worker.epoch.is_random, config.server.num_epochs,
                                        config.worker.epoch.mean, config.worker.epoch.std, config.worker.epoch.beta, config.worker.epoch.coef)
    classed_data_and_labels = split_by_class_JAX(train_images, train_labels)
    valid_data, valid_labels = sample_per_class_JAX(validation_key, config.data.shape, config.data.num_validation, classed_data_and_labels)
    allocate_client_datasets_JAX(training_key, clients, config.data.alloc_type, config.data.alloc_ratio,
                            num_classes, classed_data_and_labels, config.data.beta, config.data.shape)
    plot_data_distribution(config.worker.num, [client.labels for client in clients], config.wandb.name)
    generate_active_client_matrix_JAX(activity_key, clients, config.worker.inact_prob, config.server.num_epochs)

    Client.test_data = test_images
    Client.test_labels = test_labels
    Client.valid_data = valid_data
    Client.valid_labels = valid_labels

    #*##################################

    JKEY, model_key = jr.split(JKEY)
    server_model = get_model(config.train.model, config.data.num_classes)
    
    server_params = server_model.init(model_key, jnp.ones((config.data.batch_size, *config.data.shape)), train=True)
    server_tx = config.server.tx.fn(config.server.tx.lr)
    state_server_tx = server_tx.init(server_params)
    client_tx = config.worker.tx.fn(config.worker.tx.lr, config.worker.tx.moment)

    # state_server = train_state.TrainState.create(apply_fn=lenet5.apply, params=variables['params'], tx=tx_server)

    for server_epoch in range(config.server.num_epochs):
        epoch_start_time = datetime.now()
        print(f"Server Epoch {server_epoch + 1} Started...")
        log_row = pd.Series()
        num_active_clients = 0
        agg_params = jax.tree.map(lambda x: jnp.zeros_like(x), server_params)
        # agg_grads = jax.tree.map(lambda x: jnp.zeros_like(x), server_params)
        
        for client in clients:
            if client.active[server_epoch]:
                print(f"Training on Client {client.client_id + 1}...")
                JKEY, client_key = jr.split(JKEY)
                num_active_clients += 1
                state_client_tx = client_tx.init(server_params)
                client_params, log_row = train_client(client_key, client, server_model,
                                            server_params, client_tx,
                                            state_client_tx, config.train.loss_fn, 
                                            client.epochs[server_epoch],
                                            config.data.batch_size, num_classes,
                                             config.train.batchwise_update, log_row)
                agg_params = jax.tree.map(lambda x, y: x + y, agg_params, client_params)
                
                # client_grad = train_client(client_key, client, server_model,
                #                 server_params, client_tx,
                #                 state_client_tx, config.train.loss_fn,
                #                 client.epochs[server_epoch],
                #                 config.data.batch_size, num_classes,
                #                 config.train.batchwise_update)
                # agg_grads = jax.tree.map(lambda x, y: x + y, agg_grads, client_grad)
        
        epoch_end_time = datetime.now()

        if num_active_clients >= len(clients) * config.server.update_thresh:
            agg_params = jax.tree.map(lambda x: x / num_active_clients, agg_params)
            server_params = agg_params
            # agg_grads = jax.tree.map(lambda x, y: x - y, server_params, agg_params)
            # updates, state_server_tx = server_tx.update(agg_grads, state_server_tx, server_params)
            # server_params = optax.apply_updates(client_params, updates)
            
            # agg_grads = jax.tree.map(lambda x: x / num_active_clients, agg_grads)
            # updates, state_server_tx = server_tx.update(agg_grads, state_server_tx, server_params)
            # server_params = optax.apply_updates(server_params, updates)        
            
            test_loss, test_acc = test_client(server_model, server_params, config.train.loss_fn, Client.test_data, Client.test_labels, num_classes)
            valid_loss, valid_acc = test_client(server_model, server_params,config.train.loss_fn, Client.valid_data, Client.valid_labels, num_classes)
            log_row = log(log_row, server_epoch, 
                          {"Server Validation Loss": valid_loss, 
                           "Server Validation Accuracy": valid_acc,
                           "Server Test Loss": test_loss, 
                           "Server Test Accuracy": test_acc})
            
            # print(f"Server Validation Loss: {valid_loss}, Server Validation Accuracy: {valid_acc}")
            # print(f"Server Test Loss: {test_loss}, Server Test Accuracy: {test_acc}")
        else:
            print("Not enough active clients to update the server model")
            print(f"Server Epoch {server_epoch + 1} skipped...")
        
        print(f'Server Epoch {server_epoch + 1} is completed in {(epoch_end_time - epoch_start_time).total_seconds()} seconds.')
        log_row = log(log_row, server_epoch, {f'Server Run Time': (epoch_end_time - epoch_start_time).total_seconds()})
        if log_df.empty:
            log_df = log_row.to_frame().T
        else:
            log_df.loc[len(log_df)] = log_row

    log_df['Run'] = config.wandb.name
    wandb.config.update(config.to_dict(), allow_val_change=True)
    run.finish()
    return server_model, server_params, log_df

if __name__ == "__main__":
    configs = load_and_extract_configs('Params')
    #! Number of clients should be the same accross all configurations
    #! if we want to compare the performance of different runs with clinets stats
    
    for idx, config in enumerate(configs):
        print(f"Running with Config #{idx + 1}...")
        for run in range(int(config.train.num_runs)):
            run_name = get_run_name(config, run + 1)
            config.wandb.name = run_name
            curr_path = os.path.dirname(os.path.realpath(__file__))
            log_df = pd.DataFrame()
            print(f"Run #{run + 1} with Config #{idx + 1} started...")
            federator_start_time = datetime.now()
            server_model, server_params, log_df = main(config, log_df)
            federator_end_time = datetime.now()
            print(f'The federated learning process is completed in {(federator_end_time - federator_start_time).total_seconds()} seconds.')
            log_df.to_csv(f"{os.path.join(curr_path, 'Logs', run_name)}.csv", index=False)
            
             