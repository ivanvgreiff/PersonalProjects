import torch
from src.utils import get_mnist_data, get_device
from src.models import ConvNN
from src.training_and_evaluation import evaluate_robustness_smoothing
from src.models import *



mnist_testset = get_mnist_data(train=False)
device = get_device()

model = ConvNN()
model.to(device)
    
num_samples_1 = int(1e3)  # reduce this to 1e2 in case it takes too long, e.g. 
                          # because you don't have CUDA
num_samples_2 = int(1e4)  # reduce this to 1e3 in case it takes too long, e.g. 
                          # because you don't have CUDA
certification_batch_size = int(5e3)  # reduce this to 5e2 if required (e.g. not 
                                     # enough memory)
sigma = 1
alpha = 0.05
num_classes = 10

training_types = ["standard_training", 
                  "adversarial_training", 
                  "randomized_smoothing"]



results = {}

for training_type in training_types:
    model.load_state_dict(torch.load(f"models/{training_type}.checkpoint"))
    smoothed_classifier = SmoothClassifier(base_classifier=model, sigma=sigma, 
                             num_classes=num_classes)

    certification_results = \
        evaluate_robustness_smoothing(smoothed_classifier, sigma, mnist_testset, device,
                                      num_samples_1=num_samples_1,
                                      num_samples_2=num_samples_2, 
                                      alpha=alpha, 
                                      certification_batch_size=certification_batch_size)
    results[training_type] = certification_results



# Robustness comparison
for k,v in results.items():
    print(f"{k}: correct_certified {v['correct_certified']}, avg. certifiable "
          f"radius: {v['avg_radius']}")