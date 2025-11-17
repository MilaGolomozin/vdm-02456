import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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
device = torch.device('cpu')


# ---------------------------
# Fourier features
# ---------------------------
class Base2FourierFeatures(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        freqs = 2.0 ** torch.arange(num_frequencies)
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        # x: [B, D]
        B, D = x.shape
        w = self.freqs[None, :, None] * 2 * np.pi  # [1, num_freqs, 1]
        h = x.unsqueeze(1) * w  # [B, num_freqs, D]
        sin_feat = torch.sin(h)
        cos_feat = torch.cos(h)
        h = torch.cat([sin_feat, cos_feat], dim=1)  # [B, 2*num_freqs, D]
        return h.flatten(1)  # [B, 2*num_freqs*D]


# ---------------------------
# Score network
# ---------------------------
class ScoreNetwork(nn.Module):
    def __init__(self, input_dim, hidden_units, num_frequencies=16):
        super().__init__()
        ff_out_dim = input_dim * 2 * num_frequencies
        self.ff = Base2FourierFeatures(num_frequencies)
        self.dense1 = nn.Linear(input_dim + 1 + ff_out_dim, hidden_units)  # +1 for gamma_t
        self.dense2 = nn.Linear(hidden_units, hidden_units)
        self.dense3 = nn.Linear(hidden_units, input_dim)  # output same dim as input
        self.act = nn.SiLU()

    def forward(self, z, gamma_t):
        # z: [B, D], gamma_t: [B]
        B, D = z.shape
        gamma_t_norm = ((gamma_t - init_gamma_0) / (init_gamma_1 - init_gamma_0)) * 2 - 1
        # Fourier features only on z
        h_ff = self.ff(z)
        h = torch.cat([z, gamma_t_norm.unsqueeze(1), h_ff], dim=1)
        h = self.act(self.dense1(h))
        h = self.act(self.dense2(h))
        h = self.dense3(h)
        return h


# ---------------------------
# Noise schedule
# ---------------------------
# Simple scalar noise schedule, i.e. gamma(t) in the paper:
# gamma(t) = abs(w) * t + b
#so we are only using linear noise
class NoiseSchedule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.full((1,), init_gamma_1 - init_gamma_0))
        self.b = nn.Parameter(torch.full((1,), init_gamma_0))

    def forward(self, t):
        return torch.abs(self.w) * t + self.b


# ---------------------------
# VDM Model
# ---------------------------
class Model(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super().__init__()
        self.score_net = ScoreNetwork(input_dim, hidden_units)
        self.noise_schedule = NoiseSchedule()

    def forward(self, x, t):
        gamma_t = self.noise_schedule(t)
        return self.score_net(x, gamma_t)

    def score(self, x, t):
        return self.score_net(x, t)

    def gamma(self, t):
        return self.noise_schedule(t)


# ---------------------------
# Data encoding / decoding
# ---------------------------
def data_encode(x):
    # This transforms x from discrete values (0, 1, ...)
  # to the domain (-1,1).
  # Rounding here just a safeguard to ensure the input is discrete
  # (although typically, x is a discrete variable such as uint8)
    x = x.round()
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0) + 1e-6  # avoid div by zero
    return (x - x_mean) / x_std

#The goal of data_decode is to map a continuous latent z_0_rescaled back to a probability distribution over discrete data values.
#p(x|z_0)

def data_decode(z_0_rescaled, gamma_0):
    # z_0_rescaled: [B,D], gamma_0: scalar or [B]
    B, D = z_0_rescaled.shape
    x_vals = torch.arange(vocab_size, device=z_0_rescaled.device).float()[:, None]  # [vocab_size,1]
    x_vals = x_vals.repeat(1, D)  # [vocab_size, D]
    x_vals = data_encode(x_vals).T.unsqueeze(0)  # [1,D,vocab_size]
    inv_stdev = torch.exp(-0.5 * gamma_0[..., None]) #this is basically the inverse of the standard deviation which measures how "wide" the distribution is. if 1/std is small logits are flatter (less sharp distribution). if 1/std is large -> logits are sharper (more peaked distrib)
#    #In Gaussian likelihoods, you always scale the residual by the inverse of the standard deviation
#   #If the true σ is large, the same difference x−μ should be considered less surprising, so the probability should be higher.
#   #If σ is small, even a small difference x−μ should drastically reduce the probability
    logits = -0.5 * ((z_0_rescaled[..., None] - x_vals) * inv_stdev) ** 2
    return F.log_softmax(logits, dim=-1) #Softmax normalizes the logits across discrete values (vocab_size dimension), turning them into probabilities that sum to 1.

"""
Each element of z_0_rescaled represents a latent “noisy” version of the discrete data.

data_decode says “given this latent value, what is the probability of each possible discrete symbol?”

data_logprob then selects the probability of the actual observed symbol and sums across dimensions to get the total log-likelihood.
"""

def data_logprob(x, z_0_rescaled, gamma_0):
    x = x.round().long()
    x_onehot = F.one_hot(x, num_classes=vocab_size).float()
    logprobs = data_decode(z_0_rescaled, gamma_0)
    return torch.sum(x_onehot * logprobs, dim=(1,2))  #Multiplies the one-hot vectors by the logits or log-probabilities → selects the probability corresponding to the true symbol.
  #Then sums over features (D) and vocab dimension to get a single scalar per batch example.

#Below the function is the equivalent of the sample function in the original VDM implementation however it only works for z_0

def data_generate_x(z_0, gamma_0):
    var_0 = torch.sigmoid(gamma_0)
    z_0_rescaled = z_0 / torch.sqrt(1. - var_0).unsqueeze(1)
    logits = data_decode(z_0_rescaled, gamma_0)  # [B,D,vocab]
    logits = logits.permute(0,2,1)  # [B,vocab,D]
    dist = torch.distributions.Categorical(logits=logits)
    samples = dist.sample().T.float()
    return samples


# ---------------------------
# Loss function
# ---------------------------
def loss_fn(model, x, T_train=0):
    B, D = x.shape
    gamma_0 = model.gamma(torch.tensor(0., device=x.device)) #noise level at start fo the diffusion
    gamma_1 = model.gamma(torch.tensor(1., device=x.device)) #noise level at end of the diffusion
    var_0 = torch.sigmoid(gamma_0)
    var_1 = torch.sigmoid(gamma_1)

    f = data_encode(x)

    # Reconstruction loss
    eps_0 = torch.randn_like(f)
    z_0_rescaled = f + torch.exp(0.5 * gamma_0) * eps_0
    loss_recon = -data_logprob(x, z_0_rescaled, gamma_0)

    # Latent KL
    mean1_sqr = (1. - var_1) * f**2
    loss_klz = 0.5 * torch.sum(mean1_sqr + var_1 - torch.log(var_1) - 1., dim=1)

    # Diffusion loss
    t = torch.rand(B, device=x.device)
    if T_train > 0:
        t = torch.ceil(t * T_train) / T_train

    gamma_t = model.gamma(t)
    var_t = torch.sigmoid(gamma_t).unsqueeze(1)
    eps = torch.randn_like(f)
    z_t = torch.sqrt(1. - var_t) * f + torch.sqrt(var_t) * eps
    eps_hat = model.score(z_t, gamma_t)
    loss_diff_mse = torch.sum((eps - eps_hat)**2, dim=1)

    if T_train == 0:
        t.requires_grad_(True)
        grad = torch.autograd.grad(
            outputs=model.gamma(t).sum(),
            inputs=t,
            create_graph=True
        )[0]
        loss_diff = 0.5 * grad * loss_diff_mse
    else:
        s = t - (1./T_train)
        gamma_s = model.gamma(s)
        loss_diff = 0.5 * T_train * (torch.expm1(gamma_t - gamma_s)) * loss_diff_mse

    rescale_to_bpd = 1. / (D * np.log(2.))
    bpd_latent = torch.mean(loss_klz) * rescale_to_bpd
    bpd_recon = torch.mean(loss_recon) * rescale_to_bpd
    bpd_diff = torch.mean(loss_diff) * rescale_to_bpd
    bpd = bpd_latent + bpd_recon + bpd_diff

    return bpd, [bpd_latent, bpd_recon, bpd_diff]


# ---------------------------
# Create 2D swirl dataset
# ---------------------------
theta = np.sqrt(np.random.rand(N))*3*np.pi
r_a = 2*theta + np.pi
x = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
x = 4*(x + .25*np.random.randn(N,2) + 30)
x = torch.tensor(x, dtype=torch.float32, device=device)

#breakpoint()
#plt.scatter(x[:,0], x[:,1], alpha=0.1)
#plt.show()


# ---------------------------
# Training
# ---------------------------
model = Model(input_dim=2, hidden_units=hidden_units).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

losses = []
for i in range(num_train_steps):
    model.train()
    optimizer.zero_grad()
    loss, metrics = loss_fn(model, x)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss.item():.4f}")
