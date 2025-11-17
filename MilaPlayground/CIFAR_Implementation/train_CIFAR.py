import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from vdm_2d import Model, loss_fn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------
# Hyperparameters
# ---------------------------
N = 1024
init_gamma_0 = -13.3
init_gamma_1 = 5.
hidden_units = 512
T_train = 0
vocab_size = 256
learning_rate = 1e-3
num_train_steps = 20000  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading Data

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # subtract 0.5 and divide by 0.5
    ]
)

batch_size = 64  # both for training and testing

# Load datasets
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)


# Flatten images for VDM
def preprocess_batch(batch):
    x, _ = batch
    B, C, H, W = x.shape
    x_flat = x.permute(0, 2, 3, 1).reshape(B, -1) * 255.0  # scale to 0-255
    return x_flat.to(device)



# ---------------------------
# Training
# ---------------------------
input_dim = 32 * 32 * 3  # 3072
model = Model(input_dim=input_dim, hidden_units=hidden_units).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
# model = Model(input_dim=2, hidden_units=hidden_units).to(device)
# optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

losses = []
for step, batch in enumerate(train_loader):
    if step >= num_train_steps:
        break
    model.train()
    optimizer.zero_grad()
    x_flat = preprocess_batch(batch)
    loss, metrics = loss_fn(model, x_flat)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

print("Training finished!")
