import torch
import torch.nn as nn
import torch.nn.functional as F
from VDM_our_implementation import VDM  # import your implementation
from UNetModel import UNet
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim




transform = transforms.Compose([
    transforms.ToTensor(),   # -> [0,1]
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_dataset = datasets.CIFAR10(
    root="./data",
    train=False,       # test set
    download=True,
    transform=transforms.ToTensor()  # same preprocessing as training
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,      # no shuffle for validation
    num_workers=4,
    pin_memory=True
)
image_shape = (3, 32, 32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=3).to(device)

vdm = VDM(
    model=model,
    image_shape=image_shape,
    gamma_min=-5.0,
    gamma_max=5.0,
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

num_epochs=5
for epoch in range(num_epochs):
    running_loss = 0.0

    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(device)

        optimizer.zero_grad()
        loss = vdm.forward(x)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()  # accumulate batch loss

    # compute average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    

    print(f"Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_loss:.4f}")

vdm.eval()
mse_losses = []

with torch.no_grad():
    for x, _ in val_loader:  # validation DataLoader
        x = x.to(device)
        # sample images from the model
        samples = vdm.sample(batch_size=x.size(0), n_sample_steps=50)
        
        # rescale x to [-1,1] to match your VDM normalization if needed
        x_encoded = vdm.data_encode(x)

        # compute MSE per batch
        mse = ((samples - x_encoded) ** 2).mean()
        mse_losses.append(mse.item())

# average MSE over the validation set
val_mse = sum(mse_losses) / len(mse_losses)
print(f"Validation MSE: {val_mse:.6f}")