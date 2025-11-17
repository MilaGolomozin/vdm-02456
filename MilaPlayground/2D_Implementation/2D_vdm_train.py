import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from vdm_2D import Model, ScoreNetwork, NoiseSchedule
import matplotlib.pyplot as plt

device = torch.device('cpu')
model=Model().to(device)

#Hyper-parameters
# Data hyper-parameters
N = 1024                 # nr of datapoints

# Model hyper-parameters
init_gamma_0 = -13.3    # initial gamma_0
init_gamma_1 = 5.       # initial gamma_1
hidden_units = 512
T_train = 0                   # nr of timesteps in model; T=0 means continuous-time
vocab_size = 256
act = nn.SiLU() #this is the Sigmoid Linear Unit function
# Optimization hyper-parameters
learning_rate = 3e-3
num_train_steps = 20000   # nr of training steps

