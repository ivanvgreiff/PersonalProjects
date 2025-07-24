import torch
from torch.optim import Adam
from matplotlib import pyplot as plt
from src.utils import get_mnist_data, get_device
from src.models import ConvNN, SmoothClassifier
from src.training_and_evaluation import train_model, standard_loss
from torch.nn.functional import cross_entropy
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)



mnist_trainset = get_mnist_data(train=True)
mnist_testset = get_mnist_data(train=False)
device = get_device()
base_classifier = ConvNN().to(device)


sigma = 1
batch_size = 128
lr = 1e-3
epochs = 1


model = SmoothClassifier(base_classifier=base_classifier, num_classes=10, 
                         sigma=sigma)
opt = Adam(model.parameters(), lr=lr)
losses, accuracies = train_model(model, mnist_trainset, batch_size, device, loss_function=standard_loss, optimizer=opt)

torch.save(model.base_classifier.state_dict(), 
           "models/randomized_smoothing.checkpoint")


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