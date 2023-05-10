import torch
import numpy as np

# Function to filter top x% gradients
def filter_top_gradients(grads, top_percent):
    flat_grads = torch.abs(grads.flatten())
    num_to_keep = int(np.ceil(top_percent * flat_grads.numel()))
    threshold = flat_grads.topk(num_to_keep).values[-1]
    mask = torch.abs(grads) >= threshold
    return grads * mask