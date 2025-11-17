import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim

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
device = torch.device('cpu')


# Define learnable model
class Model(nn.Module):

  def __init__(self):
    super().__init__()
    self.score_net = ScoreNetwork() #in this case do we call the noise predicting model 
    self.noise_schedule = NoiseSchedule()


  def forward(self, x, t):
    gamma_t = self.noise_schedule(t)
    return self.score_net(x, gamma_t)
  
  def score(self, x, t):
    return self.score_net(x, t)
  
  def gamma(self, t):
    return self.noise_schedule(t)


















# A fully-connected MLP as the score network
class ScoreNetwork(nn.Module):

  def __init__(self):
        super().__init__()
        num_frequencies =16
        input_dim = vocab_size + 1  # +1 for gamma_t
        ff_out_dim = input_dim * 2 * num_frequencies  # sin+cos for each frequency
        self.dense1 = nn.Linear(input_dim + ff_out_dim , hidden_units)  # +1 for gamma_t, +32 for Fourier features
        self.dense2 = nn.Linear(hidden_units, hidden_units)
        self.dense3 = nn.Linear(hidden_units, 2)
        self.ff = Base2FourierFeatures(num_frequencies=16)

  def forward(self, z, gamma_t):
        """
        z: [batch_size, vocab_size] — input data
        gamma_t: [batch_size] — scalar noise level
        """
        # Normalize gamma_t to [-1, 1]
        lb = init_gamma_0
        ub = init_gamma_1
        gamma_t_norm = ((gamma_t - lb) / (ub - lb)) * 2 - 1

        # Concatenate normalized gamma_t
        h = torch.cat([z, gamma_t_norm.unsqueeze(1)], dim=1)

        # Append Fourier features
        h_ff = self.ff(z)
        h = torch.cat([h, h_ff], dim=1)

        # Feedforward network
        h = F.silu(self.dense1(h))  # Swish = SiLU
        h = F.silu(self.dense2(h))
        h = self.dense3(h)
        return h
  

# class Base2FourierFeatures(nn.Module):
#   # Create Base 2 Fourier features
#   def __init__(self,inputs):
#     super().__init__()
#     self.freqs = np.asarray(range(8), dtype=inputs.dtype) #[0, 1, ..., 7]
    
  
#   def forward(self, inputs):
    
#     w = 2.**self.freqs * 2 * np.pi
#     w = np.tile(w[None, :], (1, inputs.shape[-1]))
#     h = np.repeat(inputs, len(self.freqs), axis=-1)
#     h *= w
#     h = np.concatenate([np.sin(h), np.cos(h)], axis=-1)
#     return h

class Base2FourierFeatures(nn.Module):
    """
    Generate Base-2 Fourier features for a given input tensor.
    This transforms input x -> [sin(2^k * 2πx), cos(2^k * 2πx)] for k in [0, ..., num_frequencies-1].
    """
    def __init__(self, num_frequencies=8):
        super().__init__()
        self.num_frequencies = num_frequencies
        # register freqs as a buffer (not a parameter, but moves with model to GPU)
        freqs = 2.0 ** torch.arange(num_frequencies)
        self.register_buffer('freqs', freqs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: [batch_size, dim]
        returns: [batch_size, dim * num_frequencies * 2]
        """
        # shape: [num_frequencies] → [1, num_frequencies, 1] for broadcasting
        w = self.freqs[None, :, None] * 2 * torch.pi  # base-2 scaled frequencies

        # unsqueeze inputs for broadcasting: [B, D] → [B, 1, D]
        h = inputs.unsqueeze(1) * w  # [B, num_freqs, D]

        # compute sin and cos
        sin_feat = torch.sin(h)
        cos_feat = torch.cos(h)

        # concatenate along the frequency dimension
        h = torch.cat([sin_feat, cos_feat], dim=1)  # [B, 2*num_freqs, D]

        # flatten frequencies and dims together
        return h.flatten(1)


# def constant_init(value, dtype='float32'):
#   def __init__(key, shape, dtype=dtype):
#     return value * np.ones(shape, dtype)
#   return _init
 
def constant_init(value, dtype=torch.float32):
    """
    Returns a function that creates a tensor of a given shape filled with `value`.
    """
    def _init(shape, device=None):
        return torch.full(shape, value, dtype=dtype, device=device)
    return _init

# Simple scalar noise schedule, i.e. gamma(t) in the paper:
# gamma(t) = abs(w) * t + b
class NoiseSchedule(nn.Module):

  def __init__(self):
    super().__init__()
    init_bias = init_gamma_0
    init_scale = init_gamma_1 - init_gamma_0
    # self.w = self.param('w', constant_init(init_scale), (1,))
    # self.b = self.param('b', constant_init(init_bias), (1,))
    # Learnable parameters
    self.w = nn.Parameter(torch.full((1,), init_scale))
    self.b = nn.Parameter(torch.full((1,), init_bias))


  def forward(self, t):
    return abs(self.w) * t + self.b
  

def data_encode(x):
  # This transforms x from discrete values (0, 1, ...)
  # to the domain (-1,1).
  # Rounding here just a safeguard to ensure the input is discrete
  # (although typically, x is a discrete variable such as uint8)
  x = x.round()
  # Get mean and standard deviation of 'x'
  x_mean = x.mean(axis=0)
  x_std = x.std(axis=0)
  return (x-x_mean)/x_std




# def data_decode(z_0_rescaled, gamma_0):
#   # Logits are exact if there are no dependencies between dimensions of x
#   x_vals = np.arange(0, vocab_size)[:, None]
#   x_vals = np.repeat(x_vals, z_0_rescaled.shape[-1], 1) #If z_0_rescaled has shape [batch, D], then now x_vals has shape [vocab_size, D].
#   x_vals = data_encode(x_vals).transpose([1, 0])[None, :, :] #data_encode() maps each discrete value into the same continuous space as your model’s latent variables z_0_rescaled.
#   #gamma_0 is the log variance (or "noise level") at time t=0.
#   #np.exp(-0.5 * gamma_0) = 1 / standard deviation
#   #This scales how "sharp" or "broad" the distribution over discrete values should be.
#   inv_stdev = np.exp(-0.5 * gamma_0[..., None]) #this is basically the inverse of the standard deviation which measures how "wide" the distribution is. if 1/std is small logits are flatter (less sharp distribution). if 1/std is large -> logits are sharper (more peaked distrib)
#   #In Gaussian likelihoods, you always scale the residual by the inverse of the standard deviation
#   #If the true σ is large, the same difference x−μ should be considered less surprising, so the probability should be higher.
#   #If σ is small, even a small difference x−μ should drastically reduce the probability

#   logits = -0.5 * np.square((z_0_rescaled[..., None] - x_vals) * inv_stdev)

#   logprobs = nn.Softmax(logits) #Softmax normalizes the logits across discrete values (vocab_size dimension), turning them into probabilities that sum to 1.

#   return logprobs

def data_decode(z_0_rescaled, gamma_0):
    # z_0_rescaled: [B, D]
    # gamma_0: scalar or [B]
    device = z_0_rescaled.device
    x_vals = torch.arange(vocab_size, device=device).float()[:, None]  # [vocab_size,1]
    x_vals = x_vals.repeat(1, z_0_rescaled.shape[1])  # [vocab_size, D]
    x_vals = data_encode(x_vals).T.unsqueeze(0)  # [1, D, vocab_size]

    inv_stdev = torch.exp(-0.5 * gamma_0[..., None])
    logits = -0.5 * ((z_0_rescaled[..., None] - x_vals) * inv_stdev) ** 2
    logprobs = F.softmax(logits, dim=-1)
    return logprobs





"""
Each element of z_0_rescaled represents a latent “noisy” version of the discrete data.

data_decode says “given this latent value, what is the probability of each possible discrete symbol?”

data_logprob then selects the probability of the actual observed symbol and sums across dimensions to get the total log-likelihood.
"""

def data_logprob(x, z_0_rescaled, gamma_0): #Compute the log-probability of observed discrete data x under the model, given latent continuous variables z_0_rescaled and noise level gamma_0.
  x = torch.round(x).long()  # shape [B, D]
  x_onehot = F.one_hot(x, num_classes=vocab_size).float()  # shape [B, D, vocab_size]
  logprobs = data_decode(z_0_rescaled, gamma_0) #Maps the continuous latent z_0_rescaled into probabilities/logits over discrete symbols.
  logprob = torch.sum(x_onehot * logprobs, axis=(1, 2)) #Multiplies the one-hot vectors by the logits or log-probabilities → selects the probability corresponding to the true symbol.
  #Then sums over features (D) and vocab dimension to get a single scalar per batch example.
  return logprob



def data_generate_x(z_0, gamma_0, rng):
  var_0 = torch.sigmoid(gamma_0) #Sigmoid ensures variance is between 0 and 1.
  z_0_rescaled = z_0 / np.sqrt(1. - var_0).unsqueeze(1)  # [B, D]
  logits = data_decode(z_0_rescaled, gamma_0) #Converts the continuous latent to a logit distribution over discrete symbols.
  # Using torch.distributions.Categorical (logits can be unnormalized)
  dist = torch.distributions.Categorical(logits=logits) #Draws one sample per feature dimension according to the probabilities implied by the logits
  samples = dist.sample()  # [B, D]  
  #samples =jax.random.categorical(rng, logits)
  return samples


def loss_fn(model, x, T_train=0, device='cpu'):
    """
    Compute full VDM loss: reconstruction + latent KL + diffusion MSE.
    
    Args:
        model: PyTorch model with methods:
               - model.gamma(t)
               - model.score(z, gamma_t)
        x: [B, D] discrete input data
        T_train: number of discrete time steps (0 = continuous)
        device: 'cpu' or 'cuda'
    
    Returns:
        loss: scalar tensor
        metrics: list of [bpd_latent, bpd_recon, bpd_diff]
    """

    B, D = x.shape

    # ---------------------------
    # 1️⃣ gamma and variances
    # ---------------------------
    gamma_0 = model.gamma(torch.tensor(0., device=device))
    gamma_1 = model.gamma(torch.tensor(1., device=device))
    var_0 = torch.sigmoid(gamma_0)
    var_1 = torch.sigmoid(gamma_1)

    # ---------------------------
    # 2️⃣ Encode data
    # ---------------------------
    f = data_encode(x)  # [B, D]

    # ---------------------------
    # 3️⃣ Reconstruction loss
    # ---------------------------
    eps_0 = torch.randn_like(f, device=device)
    z_0 = torch.sqrt(1. - var_0) * f + torch.sqrt(var_0) * eps_0
    z_0_rescaled = f + torch.exp(0.5 * gamma_0) * eps_0
    loss_recon = -data_logprob(x, z_0_rescaled, gamma_0)  # [B]

    # ---------------------------
    # 4️⃣ Latent KL loss
    # ---------------------------
    mean1_sqr = (1. - var_1) * f**2
    loss_klz = 0.5 * torch.sum(mean1_sqr + var_1 - torch.log(var_1) - 1., dim=1)  # [B]

    # ---------------------------
    # 5️⃣ Diffusion loss
    # ---------------------------
    t = torch.rand(B, device=device)  # uniform [0,1]
    if T_train > 0:
        t = torch.ceil(t * T_train) / T_train

    gamma_t = model.gamma(t)
    var_t = torch.sigmoid(gamma_t).unsqueeze(1)  # [B, 1]
    eps = torch.randn_like(f, device=device)
    z_t = torch.sqrt(1. - var_t) * f + torch.sqrt(var_t) * eps

    eps_hat = model.score(z_t, gamma_t)  # predicted noise

    loss_diff_mse = torch.sum((eps - eps_hat)**2, dim=1)  # [B]

    if T_train == 0:
        # continuous-time case
        # Approximate gradient using autograd
        t.requires_grad_(True)
        gamma_t_val = model.gamma(t)
        grad = torch.autograd.grad(
            outputs=gamma_t_val.sum(),
            inputs=t,
            create_graph=True
        )[0]
        loss_diff = 0.5 * grad * loss_diff_mse
    else:
        # discrete-time case
        s = t - (1. / T_train)
        gamma_s = model.gamma(s)
        loss_diff = 0.5 * T_train * (torch.expm1(gamma_t - gamma_s)) * loss_diff_mse

    # ---------------------------
    # 6️⃣ Rescale to bits per dimension
    # ---------------------------
    rescale_to_bpd = 1. / (np.prod(x.shape[1:]) * np.log(2.))
    bpd_latent = torch.mean(loss_klz) * rescale_to_bpd
    bpd_recon = torch.mean(loss_recon) * rescale_to_bpd
    bpd_diff = torch.mean(loss_diff) * rescale_to_bpd
    bpd = bpd_recon + bpd_latent + bpd_diff

    loss = bpd
    metrics = [bpd_latent, bpd_recon, bpd_diff]
    return loss, metrics



# # define loss function
# def loss_fn(model, x):

# #   gamma = lambda t: model.apply(params, t, method=Model.gamma)
# #   gamma_0, gamma_1 = gamma(0.), gamma(1.)
# #   var_0, var_1 = F.sigmoid(gamma_0), F.sigmoid(gamma_1)
# #   n_batch = x.shape[0]

# # ---------------------------
#     # 1️⃣ Get gamma_0 and gamma_1
#     # ---------------------------
#   gamma_0 = model.gamma(torch.tensor(0., device=device))  # log-variance at t=0
#   gamma_1 = model.gamma(torch.tensor(1., device=device))  # log-variance at t=1

#     # sigmoid to get actual variances
#   var_0 = torch.sigmoid(gamma_0)
#   var_1 = torch.sigmoid(gamma_1)

#   # encode
#   f = data_encode(x)

#   # 1. RECONSTRUCTION LOSS
#   # add noise and reconstruct
#   eps_0 = torch.randn_like(f, device=device)  # standard Gaussian noise [B, D]
#   #z_0 = np.sqrt(1. - var_0) * f + np.sqrt(var_0) * eps_0 ## This is the "forward diffusion" step at t=0
#   z_0_rescaled = f + torch.exp(0.5 * gamma_0) * eps_0  # = z_0/sqrt(1-var)
#   loss_recon = - data_logprob(x, z_0_rescaled, gamma_0)

#   # 2. LATENT LOSS
#   # KL z1 with N(0,1) prior
#   mean1_sqr = (1. - var_1) * np.square(f)
#   loss_klz = 0.5 * np.sum(mean1_sqr + var_1 - np.log(var_1) - 1., axis=1)

#   # 3. DIFFUSION LOSS
#   # sample time steps
#   n_batch = x.shape[0]
# #   rng, rng1 = jax.random.split(rng)
# #   t = jax.random.uniform(rng1, shape=(n_batch,))
#   t = torch.rand(n_batch, device=device)  # uniform [0,1]

#   # discretize time steps if we're working with discrete time
#   if T_train > 0: 
#     t = torch.ceil(t * T_train) / T_train

#   # sample z_t
#   gamma_t = model.gamma(t)
#   var_t = torch.sigmoid(gamma_t).unsqueeze(1)  # [B, 1]
#   eps = torch.randn_like(f, device=device)
#   z_t = np.sqrt(1. - var_t) * f + np.sqrt(var_t) * eps
#   # compute predicted noise
#   eps_hat = model.apply(params, z_t, gamma_t, method=Model.score)
#   # compute MSE of predicted noise
#   loss_diff_mse = np.sum(np.square(eps - eps_hat), axis=1)

#   if T_train == 0:
#     # loss for infinite depth T, i.e. continuous time
#     t.requires_grad_(True)
#     gamma_t_val = model.gamma(t)
#     g_t_grad = torch.autograd.grad(
#             outputs=gamma_t_val.sum(),
#             inputs=t,
#             create_graph=True
#         )[0]
#     loss_diff = .5 * g_t_grad * loss_diff_mse
#   else:
#     # loss for finite depth T, i.e. discrete time
#     s = t - (1./T_train)
#     gamma_s = model.gamma(s)
#     loss_diff = .5 * T_train * torch.expm1(gamma_t - gamma_s) * loss_diff_mse

#   # End of diffusion loss computation

#   # Compute loss in terms of bits per dimension
#   rescale_to_bpd = 1./(np.prod(x.shape[1:]) * np.log(2.))
#   bpd_latent = np.mean(loss_klz) * rescale_to_bpd
#   bpd_recon = np.mean(loss_recon) * rescale_to_bpd
#   bpd_diff = np.mean(loss_diff) * rescale_to_bpd
#   bpd = bpd_recon + bpd_latent + bpd_diff
#   loss = bpd
#   metrics = [bpd_latent, bpd_recon, bpd_diff]
#   return loss, metrics



# Make 8-bit swirl dataset
theta = np.sqrt(np.random.rand(N))*3*np.pi # np.linspace(0,2*pi,100)
r_a = 2*theta + np.pi
x = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
# We use 8 bits, to make this a bit similar to image data, which has 8-bit
# color channels.
x = 4*(x + .25*np.random.randn(N,2) + 30)
x = x.astype('uint8')
x = torch.tensor(x, dtype=torch.float32, device=device)
plt.scatter(x[:,0],x[:,1], alpha=0.1)
plt.show()

def train_step(model, optimizer, x, device='cpu'):
    """
    Single PyTorch training step.

    Args:
        model: PyTorch model
        optimizer: torch.optim.Optimizer
        x: input batch [B, D]
        device: 'cpu' or 'cuda'
    Returns:
        loss: scalar tensor
        metrics: list of metrics
    """
    model.train()  # set model to training mode
    x = x.to(device)

    optimizer.zero_grad()  # reset gradients

    # Forward pass + compute loss
    loss, metrics = loss_fn(model, x)

    # Backward pass
    loss.backward()

    # Gradient step
    optimizer.step()

    return loss.item(), metrics


model = Model().to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# training loop (should take ~20 mins)
losses = []
for i in range(num_train_steps):
  loss, metrics  = train_step(model, optimizer, x)
  losses.append(loss)