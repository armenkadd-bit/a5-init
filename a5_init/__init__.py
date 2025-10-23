import torch
import torch.nn as nn
import numpy as np

phi = (1 + np.sqrt(5)) / 2

def light_a5_init(tensor, strength=0.0001):
    out_f, in_f = tensor.shape
    nn.init.xavier_uniform_(tensor)
    with torch.no_grad():
        for i in range(out_f):
            pattern = torch.sin(2 * np.pi * torch.arange(in_f) / in_f + i * phi)
            pattern = pattern * (strength / torch.std(pattern))
            tensor[i] += pattern

def apply_a5_init(model):
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            light_a5_init(module.weight)
