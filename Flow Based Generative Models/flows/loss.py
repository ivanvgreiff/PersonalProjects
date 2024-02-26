def likelihood(X_train, model, device):
    ##########################################################
    # YOUR CODE HERE

    x_to_device = X_train.to(device)
    loss = - model.log_prob(x_to_device).mean()

    ##########################################################

    return loss
