from .lenet import *
from .resnet import *
from .cnn import *

def get_model(model_arch:callable, num_classes:int):
    return model_arch(num_classes=num_classes)