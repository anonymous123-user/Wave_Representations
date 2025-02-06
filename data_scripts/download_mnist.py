import torch
import numpy as np
from torchvision import datasets

mnist1 = datasets.MNIST('data/mnist/', train=True, download=True)
mnist2 = datasets.MNIST('data/mnist/', train=False, download=True)