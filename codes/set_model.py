# CNN
from models.StandardCNN import CNN
from models.MLP import MLP


def CNN_base(model_name, num_class, input_size):
    """set Deep CNN base model"""
    return CNN(num_class, input_size)


def MLP_base(model_name, num_class, input_size):
    """set MLP base model"""
    return MLP(num_class, input_size)


def model_parameter_counter(model):
    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()
    return num_params
