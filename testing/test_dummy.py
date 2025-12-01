import torch
import torch.nn as nn
import numpy as np
from research.VDM_Breakdown import VDM, FixedLinearSchedule, LearnedLinearSchedule  # import your implementation

# ======= Define a tiny dummy model (the denoiser) =======
#this is what the VDM class takes as argument and usually it is a U-Net
class DummyDenoiser(nn.Module):
    """A super simple CNN-like denoiser that mimics the UNet but tiny."""
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, in_channels, 3, padding=1)
        )

    def forward(self, x, gamma_t):
        # gamma_t is (B,) but we need to broadcast it to match x
        gamma_t = gamma_t.view(-1, 1, 1, 1) #gamma_t holds the SNR at time stemp t snd this line expands the dimensions from(B,) to (B,1,1,1)
        # Optionally inject gamma_t into the model (simple modulation)
        return self.net(x) * torch.tanh(gamma_t) #the output of the model (the predicted noise) is modulated by gamma, menaning taht the amount of predicted noise depends on how noise the input is supposed to be at this timestamp


#Here we are creating a config class to store hyperparameters
#in the actual implementation they excpet a yaml file

class Config:
    noise_schedule = "fixed_linear"  # or "learned_linear"
    gamma_min = -5.0
    gamma_max = 5.0
    antithetic_time_sampling = False

cfg = Config()

# ======= 3️⃣ Create the VDM =======
image_shape = (3, 8, 8)  # small RGB images
model = DummyDenoiser(in_channels=3)
vdm = VDM(model=model, cfg=cfg, image_shape=image_shape)

# ======= 4️⃣ Make a toy dataset =======
batch_size = 2
x = torch.rand(batch_size, *image_shape)  # fake images in [0, 1]
print("The shape of the input:",x.shape) #expected [2,3,8,8] so [B,C,H,W]
#print("x bfore vocab size scaling: \n", x)
batch = (x, None) #in the forward pass of the VDM we define the batch to contain x and labels so x is assigned to x and None is assigned to labels

# ======= 5️⃣ Run the forward pass =======
loss, metrics = vdm.forward(batch)
#print("Metrics: \n",metrics)
#metrics is a dictionary containing scalar values that summarize different components of the loss
print("\n===== Forward pass results =====")
print(f"Loss: {loss.item():.6f}")
for k, v in metrics.items():
    print(f"{k:15s}: {v}")

# ======= 6️⃣ Test sampling (reverse diffusion) =======

#Sampling is never used throughout the forward pass it is used after the forward pass to check if using the model's noise prediction we can actually predict the image from a noisy sample
print("\n===== Sampling (reverse diffusion) =====")
samples = vdm.sample(batch_size=2, n_sample_steps=5, clip_samples=True) #in a normal model 50-100 sample steps are used for a high quality reconstruction
print("Sampled output shape:", samples.shape)
print("Sampled pixel range: [{:.3f}, {:.3f}]".format(samples.min().item(), samples.max().item()))
mse = ((samples - x) ** 2).mean()
print("MSE between original and sampled image:", mse.item())


# ======= 7️⃣ Optional: print intermediate stages =======
print("\n===== Inspect intermediate behavior =====")
times = vdm.sample_times(batch_size) #times is a tensor of random time steps t sampled from the interval [0, 1]. t controls how much noise we add the closer it is to 1 the more noise we add
print("Sampled times t:", times)

x_t, gamma_t = vdm.sample_q_t_0(x, times) #For each image in the batch, you compute x_t, the noisy version at the corresponding time t.
print("x_t mean:", x_t.mean().item(), "std:", x_t.std().item())
print("gamma_t:", gamma_t)

log_probs = vdm.log_probs_x_z0(x)
print("log_probs shape:", log_probs.shape) #(B, C, H, W, vocab_size)
print("Example pixel distribution (first pixel, first channel):")
print(log_probs[0, 0, 0, 0, :5])  # print first 5 vocab values 
#the probabilities by definition are between 0 and 1 but the logarithm of something that is between 0 and 1 is negative so the logs are negative
probs = log_probs.exp() #transforming the logs into actual probabilities
print(probs.shape)
