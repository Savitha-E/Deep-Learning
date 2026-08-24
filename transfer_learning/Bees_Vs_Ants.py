##section 1  : IMPORT

import torch                                     #create tensors ,build neural networks,perform mathematical operations,calculate gradients,train models,use GPUs/accelerators
import torch.nn as nn                            #tools for neural network
import torch.optim as optim                      #updates the neural network's weights
from torch.optim import lr_scheduler             #controls the learning rate
import torch.backends.cudnn as cudnn             
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()  # interactive mode
