import torch
from torch.optim import Adam
from matplotlib import pyplot as plt
from src.utils import get_mnist_data, get_device
from src.models import ConvNN
from src.training_and_evaluation import *
from src.attacks import gradient_attack
from torch.nn.functional import cross_entropy
from typing import Tuple
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)



mnist_trainset = get_mnist_data(train=True)
mnist_testset = get_mnist_data(train=False)
device = get_device()

model = ConvNN()
model.to(device)

epochs = 2
batch_size = 128
test_batch_size = 1000  # feel free to change this
lr = 1e-3

opt = Adam(model.parameters(), lr=lr)

attack_args = {'norm': "2", "epsilon": 5}



losses, accuracies = train_model(model, mnist_trainset, batch_size, device,
                                 loss_function=loss_function_adversarial_training, optimizer=opt, 
                                 loss_args=attack_args, epochs=epochs)

torch.save(model.state_dict(), "models/adversarial_training.checkpoint")

fig = plt.figure(figsize=(10,3))
plt.subplot(121)
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.subplot(122)
plt.plot(accuracies)
plt.xlabel("Iteration")
plt.ylabel("Training Accuracy")
plt.show()



clean_accuracy = predict_model(model, mnist_testset, batch_size, device,
                               attack_function=None)
print(clean_accuracy)
perturbed_accuracy = predict_model(model, mnist_testset, test_batch_size, device, 
                                   attack_function=gradient_attack, 
                                   attack_args=attack_args)
print(perturbed_accuracy)