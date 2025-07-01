import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree as jtr
import optax.losses._classification
from flax import linen as nn
import pandas as pd
import optax
from optax import tree_utils as otu
from typing import Dict
from Utils import Client, log
from copy import deepcopy
from functools import partial
from Commons import *

__all__ = ["train_client", "test_client", "step_test", "step_train", "quantize_stochatically"]

def train_client(JKEY: jax.random.PRNGKey, client:Client, server_model: nn.Module, server_params: Dict[str, jnp.array], client_tx: optax.GradientTransformation,
                state_client_tx: Dict, opt_loss_fn: optax.losses, num_epochs: int, batch_size: int, num_classes:int, update_model_in_batch: bool, log_row:pd.Series,
                qcfg:FrozenConfigDict) -> Dict[str, jnp.array]:
    
    @jax.jit
    def step_train(client_params: Dict[str, jnp.array], batch_data: jnp.array, batch_labels: jnp.array):
        def loss_fn(params):
            logits, batch_stats = server_model.apply(params, batch_data, train=True, mutable=['batch_stats'])
            one_hot_labels = jax.nn.one_hot(batch_labels, num_classes)
            # loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch_labels))
            loss = jnp.mean(opt_loss_fn(logits=logits, labels=one_hot_labels))
            # eps = 1e-12
            # loss = (-jnp.sum(one_hot_labels * jnp.log(jnp.clip(logits, eps, 1-eps)), axis=-1)).mean()
            
            # jax.debug.print(f"Batch Data: {batch_data.shape}")
            # jax.debug.print(f"Batch Labels: {batch_labels.shape}")
            # jax.debug.print(f"Logits: {logits.shape}")
            # jax.debug.print(f"Loss: {loss}")
            return loss, (batch_stats, logits)
        
        grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
        (loss, (batch_stats, logits)), grads = grad_fn(client_params)
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch_labels)
        return grads, loss, acc, batch_stats
    
    @jax.jit
    def update_model(client_params, grads, state_client_tx):
        updates, state_client_tx = client_tx.update(grads, state_client_tx, client_params)
        return optax.apply_updates(client_params, updates), state_client_tx
            
    def train_epoch(JKEY: jax.random.PRNGKey, client_params: Dict[str, jnp.array], state_client_tx: Dict):
        JKEY, perm_key = jr.split(JKEY)
        data, labels = client.data, client.labels
        num_batches = len(data) // batch_size
        
        perms = jr.permutation(perm_key, len(data))
        perms = perms[: num_batches * batch_size]
        perms = perms.reshape((num_batches, batch_size))
        
        batch_loss = jnp.array([])
        batch_acc = jnp.array([])
        
        for perm in perms:
            batch_data = data[perm]
            batch_labels = labels[perm]
            grads, loss, acc, batch_stats = step_train(client_params, batch_data, batch_labels)
            # print(batch_stats.keys())
            if len(batch_stats) > 0:
                client_params['batch_stats'] = batch_stats['batch_stats']
            # grad_norm = jnp.sqrt(np.sum(jax.tree.map(lambda y: jnp.sqrt(jnp.sum(jnp.square(y))), jax.tree_util.tree_leaves(grads))))
            # print(f"Gradient Norm: {grad_norm}")
            client_params, state_client_tx = update_model(client_params, grads, state_client_tx)
            # assert grad_norm != 0, "Gradient is zero"
            batch_loss = jnp.append(batch_loss, loss)
            batch_acc = jnp.append(batch_acc, acc)
        train_loss = jnp.mean(batch_loss)
        train_acc = jnp.mean(batch_acc)
        # print(f"Client {client.client_id + 1}'s Training Loss: {batch_loss},\n Training Accuracy: {batch_acc}")
        return client_params, state_client_tx, train_loss, train_acc
    
    
    client_params = deepcopy(server_params)
    epoch_loss, epoch_acc = [], []
    for epoch in range(num_epochs):
        # print(f"Training Epoch {epoch + 1} Started...")
        client_params, state_client_tx, train_loss, train_acc = train_epoch(JKEY, client_params, state_client_tx)
        epoch_loss.append(train_loss)
        epoch_acc.append(train_acc)
        
        # print(f"Training Loss: {train_loss}, Training Accuracy: {train_acc}")
        # print(f"Validation Loss: {valid_loss}, Validation Accuracy: {valid_acc}")
        # print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    
    
    test_loss, test_acc = test_client(server_model, client_params, opt_loss_fn, client.test_data, client.test_labels, num_classes)
    valid_loss, valid_acc = test_client(server_model, client_params, opt_loss_fn, client.valid_data, client.valid_labels, num_classes)
    
    log_row = log(log_row, epoch, 
                  {f"Worker {client.client_id + 1}'s Local Loss": epoch_loss, 
                   f"Worker {client.client_id + 1}'s Local Accuracy": epoch_acc,
                   f"Worker {client.client_id + 1}'s Validation Loss": valid_loss, 
                   f"Worker {client.client_id + 1}'s Validation Accuracy": valid_acc,
                   f"Worker {client.client_id + 1}'s Test Loss": test_loss, 
                   f"Worker {client.client_id + 1}'s Test Accuracy": test_acc})    

    # return client_params, log_row
    agg_grads = jax.tree.map(lambda server_params, client_params: server_params - client_params, server_params, client_params)
    # print(f"original norm of grads is {otu.tree_l2_norm(agg_grads)}")
    agg_grads_norm = otu.tree_l2_norm(agg_grads)
    norm_agg_grads = otu.tree_scalar_mul(1/agg_grads_norm, agg_grads)
    # print(f"norm of grads after Norm is {otu.tree_l2_norm(norm_agg_grads)}")
    JKEY, qkey = jr.split(JKEY)
    qagg_grads = quantize_stochatically(qkey, norm_agg_grads, qcfg.levels, qcfg.prime)
    # print(f"norm of grads after quan is {otu.tree_l2_norm(qagg_grads)}")
    qagg_grads = otu.tree_scalar_mul(agg_grads_norm, qagg_grads)
    # print(f"norm of grads after quan norm is {otu.tree_l2_norm(qagg_grads)}")
    return qagg_grads

def test_client(server_model: nn.Module, params: Dict[str, jnp.array], opt_loss_fn:optax.losses, test_data: jnp.array, test_labels: jnp.array, num_classes: int):
    @jax.jit
    def step_test(params: Dict[str, jnp.array], test_data: jnp.array, test_labels: jnp.array) -> tuple[jnp.array, jnp.array]:
        logits = server_model.apply(params, test_data, train=False)
        one_hot_labels = jax.nn.one_hot(test_labels, num_classes)
        # loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=test_labels))
        loss = jnp.mean(opt_loss_fn(logits=logits, labels=one_hot_labels))
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == test_labels)
        return loss, acc
    return step_test(params, test_data, test_labels)

@partial(jax.jit, static_argnums=[0, 2])
def step_test(smodel: nn.Module, variables: dict[str, ArrayLike], 
              loss_fn: OptaxLoss,
              test_data: ArrayLike,
              test_labels: ArrayLike) -> tuple[Scalar, Scalar]:
    mutuable_vars = {k: v for k, v in variables.items() if k != "params"}
    logits, _ = smodel.apply(
            {"params": variables["params"], **mutuable_vars},
            test_data,
            train=False,
            mutable=list(mutuable_vars.keys()))
    loss = jnp.mean(loss_fn(logits=logits, labels=test_labels))
    target = jnp.argmax(test_labels, axis=-1)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == target)
    return loss, acc
    # metrics.update(loss=loss, logits=logits, labels=target)

@partial(jax.jit, static_argnums=[0, 2])
def step_train(smodel:nn.Module, variables:dict[str, ArrayTree],
                optax_loss_fn: OptaxLoss, 
                batch_data:ArrayLike,
                batch_labels:ArrayLike) -> tuple[ArrayTree, ArrayTree, Scalar, Scalar]:

    def loss_fn(params:ArrayTree, mutable_vars:dict[str, ArrayTree]) -> tuple[Scalar, tuple[ArrayTree, ArrayLike]]:
        logits, new_mutuable_vars = smodel.apply(
            {"params": params, **mutable_vars},
            batch_data,
            train=True,
            mutable=list(mutable_vars.keys()))
        loss = jnp.mean(optax_loss_fn(logits, batch_labels))
        return loss, (new_mutuable_vars, logits)
    
    mutable_vars = {k: v for k, v in variables.items() if k != "params"}
    grad_fn = jax.value_and_grad(loss_fn, argnums=[0], has_aux=True)
    (loss, (mutable_vars, logits)), grads = grad_fn(variables['params'], mutable_vars)
    target = jnp.argmax(batch_labels, axis=-1)
    # metrics.update(loss=loss, logits=logits, labels=target)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == target)
    return grads, mutable_vars, loss, acc

def quantize_stochatically(key, svars, q, p):
    leaves, treedef = jtr.flatten(svars)
    shapes = jtr.map(jnp.shape, svars)
    # shapes, _ = jtr.flatten(shapes)
    sizes = [leaf.size for leaf in leaves]
    x_concat = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves], axis=0)
    floored = jnp.floor(q * x_concat)
    alpha = (q * x_concat) - floored
    random_vals = jr.uniform(key, shape=x_concat.shape)
    bernoulli_mask = (random_vals < alpha)
    quantized = jnp.where(bernoulli_mask, (floored + 1) / q, floored / q)
    start = 0
    new_leaves = []
    for sz in sizes:
        end = start + sz
        new_leaves.append(quantized[start:end])
        start = end

    qsvars = jtr.unflatten(treedef, new_leaves)
    qsvars = jtr.map(lambda qsvar, sh: jnp.reshape(qsvar, sh), qsvars, shapes)
    # neg_map = jtr.map(lambda leaf: leaf >= 0, qsvars) 
    # qsvars = jtr.map(lambda leaf: jnp.where(leaf >= 0, leaf, leaf + p), qsvars)
    return qsvars# , neg_map