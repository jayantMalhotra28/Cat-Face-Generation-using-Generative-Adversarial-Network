import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def preprocess_img(x):
    return 2 * x - 1.0  #scales the pixel values from the range [0, 1] to the range [-1, 1] to improve stability

def deprocess_img(x):
    return (x + 1.0) / 2.0 

def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.
    
    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    to_return = torch.randn((batch_size, dim))
    return to_return/torch.max(to_return)


