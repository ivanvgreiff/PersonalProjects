import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from matplotlib import pyplot as plt
from src.utils import get_mnist_data, get_device
from src.models import ConvNN
from src.training_and_evaluation import train_model, predict_model, standard_loss
from src.attacks import gradient_attack, attack
import os
if not os.path.isdir("models"):
    os.mkdir("models")
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)



mnist_trainset = get_mnist_data(train=True)
mnist_testset = get_mnist_data(train=False)
device = get_device()

model = ConvNN()
model.to(device)

epochs = 1
batch_size = 128
test_batch_size = 1000  # feel free to change this
lr = 1e-3

opt = Adam(model.parameters(), lr=lr)



losses, accuracies = train_model(model, mnist_trainset, batch_size, device, loss_function=standard_loss, optimizer=opt)



torch.save(model.state_dict(), "models/standard_training.checkpoint")



model.load_state_dict(torch.load("models/standard_training.checkpoint", 
                                 map_location="cpu"))



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



clean_accuracy = predict_model(model, mnist_testset, test_batch_size, device=device,
                               attack_function=None)
print(clean_accuracy)



attack_args_l2 = {"epsilon": 5, "norm": "2"}
attack_args_linf = {"epsilon": 0.3, "norm": "inf"}



test_loader = DataLoader(mnist_testset, batch_size = 10, shuffle=False)
x,y = next(iter(test_loader))

x_pert_l2, y_pert_l2 = attack(x,y,model, attack_args_l2)

x_pert_linf, y_pert_linf = attack(x,y,model, attack_args_linf)



for ix in range(len(x)):
    plt.subplot(131)
    plt.imshow(x[ix,0].detach().cpu(), cmap="gray")
    plt.title(f"Label: {y[ix]}")
    plt.subplot(132)
    plt.imshow(x_pert_l2[ix,0].detach().cpu(), cmap="gray")
    plt.title(f"Predicted: {y_pert_l2[ix]}")
    
    plt.subplot(133)
    plt.imshow(x_pert_linf[ix,0].detach().cpu(), cmap="gray")
    plt.title(f"Predicted: {y_pert_linf[ix]}")
    plt.show()



perturbed_accuracy_l2 = predict_model(model, mnist_testset, 
                                      batch_size=test_batch_size, 
                                      device=device, 
                                      attack_function=gradient_attack, 
                                      attack_args=attack_args_l2)

perturbed_accuracy_linf = predict_model(model, mnist_testset, 
                                        batch_size=test_batch_size, 
                                        device=device, 
                                        attack_function=gradient_attack, 
                                        attack_args=attack_args_linf)

print(perturbed_accuracy_l2)
print(perturbed_accuracy_linf)