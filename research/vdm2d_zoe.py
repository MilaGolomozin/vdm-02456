"""Paper: https://arxiv.org/abs/2107.00630

# This is a simple implementation of VDM for educational purposes.

# Data: 2D 'swirl' data, in 8-bit (uint8) precision.
# Score network: Fully connected MLP.
# Note: Reconstruction loss produces unreasonable results for this simple 2D data.
Initial code was done by myself and then adapted to add additional data to send to WandB,
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import wandb
import torch.optim as optim

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

### HYPER PARAMETERS ###
# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    # Set the wandb project where this run will be logged.
    project="PyTorch Implementation of VDM on 2D Data",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 3e-3,
        "architecture": "VDM with MLP Score Network",
        "dataset": "2D Swirl Data",
        "epochs": 5000,
        "hidden_units": 512,
        "vocab_size": 256,
        "gamma_min": -13.3,
        "gamma_max": 5.0,
        "N": 1024,
    },
)

# Optional: structure metrics vs a global step for clean W&B charts
try:
    wandb.define_metric("global_step")
    for m in ["loss", "bpd_latent", "bpd_recon", "bpd_diff", "step_time_s", "gpu_mem_MB"]:
        wandb.define_metric(m, step_metric="global_step")
except Exception:
    pass

N = 1024                 # nr of datapoints

# Model hyper-parameters
hidden_units = 512
T_train = 0                   # nr of timesteps in model; T=0 means continuous-time
vocab_size = 256
gamma_min = -13.3
gamma_max = 5.0

# Optimization hyper-parameters
learning_rate = 3e-3
num_train_steps = 5000   # nr of training steps



### CREATE DATASET ###
# Make 8-bit swirl dataset
theta = np.sqrt(np.random.rand(N))*3*np.pi # np.linspace(0,2*pi,100)
r_a = 2*theta + np.pi
x = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
# We use 8 bits, to make this a bit similar to image data, which has 8-bit
# color channels.
x = 4*(x + .25*np.random.randn(N,2) + 30)
x = x.astype('uint8')
plt.scatter(x[:,0],x[:,1], alpha=0.1)
plt.close()

# Get mean and standard deviation of 'x'
x_mean = x.mean(axis=0)
x_std = x.std(axis=0)

# Move dataset to torch tensor on the chosen device
x_torch = torch.tensor(x, dtype=torch.float32, device=device)


# ### LEARNABLE MODEL DEFINITION ###
class VDM2DModel(nn.Module):
    def __init__(self):
        super(VDM2DModel, self).__init__()
        self.score_net = ScoreNet(hidden_units)
        self.noise_schedule = FixedLinearSchedule(gamma_min, gamma_max)

    def forward(self, x, t):
        gamma = self.noise_schedule(t)
        score = self.score_net(x, gamma)
        return score

    # Provide an explicit score method (parity with JAX API style)
    def score(self, z, gamma_t):
        return self.score_net(z, gamma_t)


### Score network definition ###
## score tells the direction to move x to increase log probability the most. 
# function that predicts how to denoise x at various noise levels (gammas)
class ScoreNet(nn.Module):
    def __init__(self, hidden_units):
        super(ScoreNet, self).__init__()
        # Fourier features on concatenated [z, gamma_norm]
        self.ff = FourierFeatures()

        # Input dims: z in R^2 + gamma_norm in R^1 => 3
        base_in = 3
        ff_dims = self.ff.num_features * base_in  # 2*F*C where C=3
        in_features = base_in + ff_dims

        self.fc1 = nn.Linear(in_features, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 2)
        self.act = nn.SiLU()  # swish

    def forward(self, z, gamma_t):
        # Normalize gamma_t to [-1, 1] using fixed schedule bounds
        lb = torch.as_tensor(gamma_min, dtype=z.dtype, device=z.device)
        ub = torch.as_tensor(gamma_max, dtype=z.dtype, device=z.device)
        gamma_t = gamma_t.view(-1)
        gamma_t_norm = ((gamma_t - lb) / (ub - lb)) * 2.0 - 1.0  # [-1, +1]

        # Concatenate normalized gamma as extra feature
        h = torch.cat([z, gamma_t_norm[:, None]], dim=1)  # (B, 3)

        # Append base-2 Fourier features of the concatenated input
        h_ff = self.ff(h)
        h = torch.cat([h, h_ff], dim=1)

        # Three dense layers with swish
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        return h




## Fixed Linear Noise schedule
class FixedLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t):
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t
    

# Fourier feature adding
class FourierFeatures(nn.Module):
    def __init__(self, first=5.0, last=6.0, step=1.0):
        super().__init__()
        self.freqs_exponent = torch.arange(first, last + 1e-8, step)

    @property
    def num_features(self):
        return len(self.freqs_exponent) * 2

    def forward(self, x):
        assert len(x.shape) >= 2

        # Compute (2pi * 2^n) for n in freqs.
        freqs_exponent = self.freqs_exponent.to(dtype=x.dtype, device=x.device)  # (F, )
        freqs = 2.0**freqs_exponent * 2 * np.pi  # (F, )
        freqs = freqs.view(-1, *([1] * (x.dim() - 1)))  # (F, 1, 1, ...)

        # Compute (2pi * 2^n * x) for n in freqs.
        features = freqs * x.unsqueeze(1)  # (B, F, X1, X2, ...)
        features = features.flatten(1, 2)  # (B, F * C, X1, X2, ...)

        # Output features are cos and sin of above. Shape (B, 2 * F * C, H, W).
        return torch.cat([features.sin(), features.cos()], dim=1)



# encode values of x from discrete values to the domain (-1,1)
def data_encode(x):
    # Rounding here just a safeguard to ensure the input is discrete
    # (although typically, x is a discrete variable such as uint8)
    x = x.round()
    # Ensure mean/std are tensors on the same device and dtype
    mean_t = torch.as_tensor(x_mean, dtype=x.dtype, device=x.device)
    std_t = torch.as_tensor(x_std, dtype=x.dtype, device=x.device)
    return (x - mean_t) / std_t


# decode values of x from (-1,1) back to discrete values
def data_decode(z_0_rescaled, gamma_0):
    # Logits are exact if there are no dependencies between dimensions of x
     # Create x values (like jnp.arange)
    x_vals = torch.arange(0, vocab_size, device=z_0_rescaled.device)[:, None]
    
    # Repeat along second dimension
    x_vals = x_vals.repeat(1, z_0_rescaled.shape[-1])
    
    # Apply encoding and reshape
    x_vals = data_encode(x_vals).permute(1, 0).unsqueeze(0)
    
    # Compute inverse stddev
    inv_stdev = torch.exp(-0.5 * gamma_0[..., None])
    
    # Compute logits
    logits = -0.5 * ((z_0_rescaled[..., None] - x_vals) * inv_stdev) ** 2
    
    # Compute log-softmax over the vocab dimension (usually last)
    logprobs = F.log_softmax(logits, dim=-1)
    
    return logprobs

# This computes how likely the observed discrete data x is, given z₀ and γ₀,
# using the categorical decoder (as in the original VDM for 8-bit data).
def data_logprob(x, z_0_rescaled, gamma_0):
    x = x.round().to(torch.int64)
    x_onehot = F.one_hot(x, vocab_size).to(dtype=z_0_rescaled.dtype)
    logprobs = data_decode(z_0_rescaled, gamma_0)
    # Sum over channels and vocab axis
    logprob = torch.sum(x_onehot * logprobs, dim=(1, 2))
    return logprob

# This generates discrete data samples from the model, given latent z₀ and scale γ₀.
# Generate synthetic data by sampling from decoder distribution
def data_generate_x(z_0, gamma_0, rng):
    var_0 = torch.sigmoid(gamma_0)
    z_0_rescaled = z_0 / torch.sqrt(1. - var_0)
    logits = data_decode(z_0_rescaled, gamma_0)
    samples = torch.distributions.Categorical(logits=logits).sample()
    return samples

#### LOSS FUNCTION DEFINITION ###
# Computes VDM loss given model parameters and a batch of data x
def loss_fn(model, x):
    gamma_fn = model.noise_schedule
    device = x.device
    gamma_0 = gamma_fn(torch.as_tensor(0.0, device=device))
    gamma_1 = gamma_fn(torch.as_tensor(1.0, device=device))
    var_0, var_1 = torch.sigmoid(gamma_0), torch.sigmoid(gamma_1) # variance at time 0 and 1
    n_batch = x.shape[0]

    # encode
    f = data_encode(x)

    # 1. RECONSTRUCTION LOSS
    # Adds noise to the original data and reconstructs it.
    # Measures how well the model can decode the latent z_0 back into the original data x.

    # Generate Gaussian noise with the same shape as the input
    eps_0 = torch.randn_like(f)

    # Create a noisy latent sample z_0
    z_0 = torch.sqrt(1.0 - var_0) * f + torch.sqrt(var_0) * eps_0

    # Rescale the noisy latent to match the data distribution: z0 / sqrt(1 - var0)
    z_0_rescaled = z_0 / torch.sqrt(1.0 - var_0)

    # Compute the negative log-likelihood of x given the reconstructed latent (per-sample)
    loss_recon = -data_logprob(x, z_0_rescaled, gamma_0)

    # 2. LATENT LOSS
    # KL z1 with N(0,1) prior
    mean1_sqr = (1. - var_1) * torch.square(f)
    loss_klz = 0.5 * torch.sum(mean1_sqr + var_1 - torch.log(var_1) - 1., dim=1)

    # 3. DIFFUSION LOSS
    # sample time steps
    t = torch.rand(n_batch, device=x.device)

    # sample z_t
    gamma_t = gamma_fn(t)
    var_t = torch.sigmoid(gamma_t)[:, None]
    eps = torch.randn_like(f)
    z_t = torch.sqrt(1. - var_t) * f + torch.sqrt(var_t) * eps
    # compute predicted noise
    eps_hat = model.score_net(z_t, gamma_t)
    # compute MSE of predicted noise
    loss_diff_mse = torch.sum(torch.square(eps - eps_hat), dim=1)


    # loss for infinite depth T, i.e. continuous time
    t = t.clone().detach().requires_grad_(True)  # make t differentiable
    gamma_t = gamma_fn(t)                             # forward pass
    g_t_grad = torch.autograd.grad(
        outputs=gamma_t,
        inputs=t,
        grad_outputs=torch.ones_like(gamma_t),
        create_graph=True
    )[0]
    loss_diff = .5 * g_t_grad * loss_diff_mse
    
    # End of diffusion loss computation



    # Compute loss in terms of bits per dimension
    rescale_to_bpd = 1./(np.prod(x.shape[1:]) * np.log(2.))
    bpd_latent = torch.mean(loss_klz) * rescale_to_bpd
    bpd_recon = torch.mean(loss_recon) * rescale_to_bpd
    bpd_diff = torch.mean(loss_diff) * rescale_to_bpd
    bpd = bpd_recon + bpd_latent + bpd_diff
    loss = bpd
    metrics = [bpd_latent, bpd_recon, bpd_diff]
    return loss, metrics




model = VDM2DModel()
print(model)
run.log({"model_summary": str(model)})
# Move model to device
model = model.to(device)

# Watch gradients/params occasionally (can be verbose; adjust log_freq)
try:
    wandb.watch(model, log="gradients", log_freq=200)
except Exception:
    pass

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ----- W&B Visualization helpers -----

# Below was mainly developed by AI for logging various model aspects to W&B.
def log_noise_schedule(model):
    ts = torch.linspace(0, 1, 200, device=device)
    with torch.no_grad():
        gammas = model.noise_schedule(ts).detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(ts.cpu().numpy(), gammas)
    ax.set_xlabel("t"); ax.set_ylabel("gamma(t)")
    run.log({"noise_schedule": wandb.Image(fig)})
    plt.close(fig)


def log_forward_diffusion_grid(x, model, step, num_points=2048):
    model.eval()
    with torch.no_grad():
        # Sample a subset of points
        idx = torch.randperm(x.shape[0], device=x.device)[:num_points]
        x_sample = x[idx]
        f = data_encode(x_sample)
        ts = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=x.device)
        fig, axs = plt.subplots(1, len(ts), figsize=(12, 2.4))
        for j, t in enumerate(ts):
            gamma_t = model.noise_schedule(t)
            var_t = torch.sigmoid(gamma_t)
            eps = torch.randn_like(f)
            z_t = torch.sqrt(1 - var_t) * f + torch.sqrt(var_t) * eps
            z_np = z_t.detach().cpu().numpy()
            axs[j].scatter(z_np[:, 0], z_np[:, 1], s=2, alpha=0.3)
            axs[j].set_title(f"t={float(t):.2f}")
            axs[j].axis('equal'); axs[j].axis('off')
        run.log({"forward_diffusion_grid": wandb.Image(fig)}, step=step)
        plt.close(fig)


def log_score_quiver(model, step, t_vals=(0.25, 0.5, 0.75), lim=3.0, res=25):
    model.eval()
    with torch.no_grad():
        xs = torch.linspace(-lim, lim, res, device=device)
        ys = torch.linspace(-lim, lim, res, device=device)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        grid = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        fig, axs = plt.subplots(1, len(t_vals), figsize=(12, 3))
        for i, tval in enumerate(t_vals):
            t = torch.tensor(tval, device=device)
            gamma_t = model.noise_schedule(t)
            score = model.score_net(grid, gamma_t.expand(grid.shape[0]))
            U, V = score[:, 0].cpu().numpy(), score[:, 1].cpu().numpy()
            axs[i].quiver(X.cpu(), Y.cpu(), U.reshape(res, res), V.reshape(res, res),
                          angles='xy', scale_units='xy', scale=1)
            axs[i].set_title(f"Score field t={tval}")
            axs[i].axis('equal'); axs[i].axis('off')
        run.log({"score_field": wandb.Image(fig)}, step=step)
        plt.close(fig)


def log_real_vs_decoded(x, model, step, num_points=2048):
    model.eval()
    with torch.no_grad():
        idx = torch.randperm(x.shape[0], device=x.device)[:num_points]
        x_sample = x[idx]
        f = data_encode(x_sample)
        gamma_0 = model.noise_schedule(torch.tensor(0.0, device=x.device))
        var_0 = torch.sigmoid(gamma_0)
        eps_0 = torch.randn_like(f)
        z_0 = torch.sqrt(1 - var_0) * f + torch.sqrt(var_0) * eps_0
        z_0_rescaled = z_0 / torch.sqrt(1 - var_0)

        # Decode to discrete x_hat
        logits = data_decode(z_0_rescaled, gamma_0)  # (B, 2, vocab)
        x_hat = torch.argmax(logits, dim=-1).to(x.dtype)  # (B, 2)

        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        axs[0].scatter(x_sample[:, 0].cpu(), x_sample[:, 1].cpu(), s=3, alpha=0.4)
        axs[0].set_title("Real x"); axs[0].axis('equal'); axs[0].axis('off')

        axs[1].scatter(x_hat[:, 0].cpu(), x_hat[:, 1].cpu(), s=3, alpha=0.4, c='orange')
        axs[1].set_title("Decoded x_hat (t=0)"); axs[1].axis('equal'); axs[1].axis('off')

        run.log({"real_vs_decoded": wandb.Image(fig)}, step=step)
        plt.close(fig)


def log_t_binned_mse(model, x, step):
    model.eval()
    with torch.no_grad():
        n = x.shape[0]
        t = torch.rand(n, device=x.device)
        gamma_t = model.noise_schedule(t)
        var_t = torch.sigmoid(gamma_t)[:, None]
        f = data_encode(x)
        eps = torch.randn_like(f)
        z_t = torch.sqrt(1 - var_t) * f + torch.sqrt(var_t) * eps
        eps_hat = model.score_net(z_t, gamma_t)
        mse = torch.mean((eps - eps_hat) ** 2, dim=1).detach().cpu().numpy()
        t_np = t.detach().cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.scatter(t_np, mse, s=3, alpha=0.2)
        ax.set_xlabel("t"); ax.set_ylabel("MSE(eps)")
        run.log({"mse_vs_t": wandb.Image(fig)}, step=step)
        plt.close(fig)

        run.log({
            "mse_mean": float(np.mean(mse)),
            "mse_std": float(np.std(mse)),
        }, step=step)


# ----- W&B: generated samples and trajectories -----
def log_sampling_outputs(z_list, x_pred_list, x_sample, step):
    # Discrete decoded samples (if available)
    if x_sample is not None:
        xs = x_sample.detach().cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.scatter(xs[:, 0], xs[:, 1], s=3, alpha=0.5)
        ax.set_title("Decoded discrete samples")
        ax.axis('equal'); ax.axis('off')
        run.log({"samples_discrete": wandb.Image(fig)}, step=step)
        plt.close(fig)

    # Continuous x_pred at final step
    if len(x_pred_list) > 0:
        xp = x_pred_list[-1].detach().cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.scatter(xp[:, 0], xp[:, 1], s=3, alpha=0.5, c='orange')
        ax.set_title("x_pred at final step")
        ax.axis('equal'); ax.axis('off')
        run.log({"samples_continuous": wandb.Image(fig)}, step=step)
        plt.close(fig)

    # Trajectory snapshots from z_list
    if len(z_list) >= 2:
        K = min(4, len(z_list))
        indices = np.linspace(0, len(z_list) - 1, K).astype(int)
        fig, axs = plt.subplots(1, K, figsize=(3 * K, 3))
        for j, idx in enumerate(indices):
            z = z_list[idx].detach().cpu().numpy()
            axs[j].scatter(z[:, 0], z[:, 1], s=2, alpha=0.5)
            axs[j].set_title(f"step {idx}")
            axs[j].axis('equal'); axs[j].axis('off')
        run.log({"sampling_trajectory": wandb.Image(fig)}, step=step)
        plt.close(fig)

# training loop (should take ~20 mins)

def train_step(model, optimizer, x):
    """
    model: nn.Module
    optimizer: torch.optim.Optimizer
    x: input tensor
    returns: loss value, metrics dict
    """
    model.train()
    optimizer.zero_grad()           # reset gradients
    
    loss, metrics = loss_fn(model, x)  # compute loss (and metrics)
    loss.backward()                 # compute gradients
    
    optimizer.step()                # update parameters
    return loss.item(), metrics


viz_interval = 500
losses = []

# One-time noise schedule visualization
log_noise_schedule(model)

for i in range(num_train_steps):
    t0 = time.perf_counter()
    loss, metrics = train_step(model, optimizer, x_torch)
    step_time = time.perf_counter() - t0
    losses.append(loss)

    # Unpack bpd components
    bpd_latent, bpd_recon, bpd_diff = [m.detach().item() for m in metrics]

    # GPU memory if available
    gpu_mem = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

    global_step = i + 1
    run.log({
        "global_step": global_step,
        "loss": loss,
        "bpd_latent": bpd_latent,
        "bpd_recon": bpd_recon,
        "bpd_diff": bpd_diff,
        "step_time_s": step_time,
        "gpu_mem_MB": gpu_mem,
    }, step=global_step)

    # Periodic visualizations
    if global_step % viz_interval == 0:
        log_forward_diffusion_grid(x_torch, model, global_step)
        log_score_quiver(model, global_step)
        log_real_vs_decoded(x_torch, model, global_step)
        log_t_binned_mse(model, x_torch, global_step)


# Log final loss as a metric
run.log({"final_loss": losses[-1]})


### SAMPLING FROM THE TRAINED MODEL
from sampling import sample_fn

model.eval()
data_stats = {"mean": x_mean, "std": x_std}  # these are computed earlier in vdm2d.py

# Choose how many samples and steps you want
z_list, x_pred_list, x_sample = sample_fn(
    model,
    N_sample=2048,
    T_sample=250,
    device=device,
    stochastic=True,         # True = stochastic sampling, False = deterministic (DDIM-like)
    data_stats=data_stats,   # provides discrete decoding into 0..255
)

# Log samples and trajectories to W&B before finishing the run
final_step = (locals().get("global_step", 0) or 0) + 1
log_sampling_outputs(z_list, x_pred_list, x_sample, step=final_step)

# Finish the run and upload any remaining data.
run.finish()