import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

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
        self.dense1 = nn.Linear(vocab_size + 1 + 32, hidden_units)  # +1 for gamma_t, +32 for Fourier features
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
        h_ff = self.ff(h)
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
    self.w = self.param('w', constant_init(init_scale), (1,))
    self.b = self.param('b', constant_init(init_bias), (1,))

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



#The goal of data_decode is to map a continuous latent z_0_rescaled back to a probability distribution over discrete data values.
#p(x|z_0)
def data_decode(z_0_rescaled, gamma_0):
  # Logits are exact if there are no dependencies between dimensions of x
  x_vals = np.arange(0, vocab_size)[:, None]
  x_vals = np.repeat(x_vals, z_0_rescaled.shape[-1], 1) #If z_0_rescaled has shape [batch, D], then now x_vals has shape [vocab_size, D].
  x_vals = data_encode(x_vals).transpose([1, 0])[None, :, :] #data_encode() maps each discrete value into the same continuous space as your model’s latent variables z_0_rescaled.
  #gamma_0 is the log variance (or "noise level") at time t=0.
  #np.exp(-0.5 * gamma_0) = 1 / standard deviation
  #This scales how "sharp" or "broad" the distribution over discrete values should be.
  inv_stdev = np.exp(-0.5 * gamma_0[..., None]) #this is basically the inverse of the standard deviation which measures how "wide" the distribution is. if 1/std is small logits are flatter (less sharp distribution). if 1/std is large -> logits are sharper (more peaked distrib)
  #In Gaussian likelihoods, you always scale the residual by the inverse of the standard deviation
  #If the true σ is large, the same difference x−μ should be considered less surprising, so the probability should be higher.
  #If σ is small, even a small difference x−μ should drastically reduce the probability

  logits = -0.5 * np.square((z_0_rescaled[..., None] - x_vals) * inv_stdev)

  logprobs = nn.Softmax(logits) #Softmax normalizes the logits across discrete values (vocab_size dimension), turning them into probabilities that sum to 1.

  return logprobs

def data_logprob(x, z_0_rescaled, gamma_0):
  x = torch.round(x).long()  # shape [B, D]
  x_onehot = F.one_hot(x, num_classes=vocab_size).float()  # shape [B, D, vocab_size]
  logprobs = data_decode(z_0_rescaled, gamma_0)
  logprob = np.sum(x_onehot * logprobs, axis=(1, 2))
  return logprob

def data_generate_x(z_0, gamma_0, rng):
  var_0 = torch.sigmoid(gamma_0)
  z_0_rescaled = z_0 / np.sqrt(1. - var_0).unsqueeze(1)  # [B, D]
  logits = data_decode(z_0_rescaled, gamma_0)
  # Using torch.distributions.Categorical (logits can be unnormalized)
  dist = torch.distributions.Categorical(logits=logits)
  samples = dist.sample()  # [B, D]  
  #samples =jax.random.categorical(rng, logits)
  return samples

