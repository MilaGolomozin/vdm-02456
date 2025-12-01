"""
This file is AI generated and used as an example for VDM sampling implementation in PyTorch for a 2d toy dataset.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict

@torch.no_grad()
def sample_step(
  model,
  z_t: torch.Tensor,
  t: torch.Tensor,
  s: torch.Tensor,
  eps: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """One reverse step z_t -> z_s using VDM closed-form discretization.

  Args:
    model: VDM2DModel-like with .noise_schedule and .score_net.
    z_t: (B, 2) latent at time t.
    t: scalar tensor in [0,1] (current time).
    s: scalar tensor in [0,1] (next time, s < t).
    eps: optional (B, 2) Gaussian noise; if None, sampled from N(0,I).

  Returns:
    z_s: (B, 2) latent at time s.
    x_pred: (B, 2) continuous reconstruction proxy at time t.
  """
  device = z_t.device
  B = z_t.shape[0]

  gamma_s = model.noise_schedule(s.to(device))
  gamma_t = model.noise_schedule(t.to(device))

  # Expand gammas to batch
  gamma_s_b = gamma_s.expand(B)
  gamma_t_b = gamma_t.expand(B)

  # Predict epsilon at time t
  eps_hat = model.score_net(z_t, gamma_t_b)  # (B,2)

  # VDM update coefficients (PyTorch replacements for JAX ops)
  a = torch.sigmoid(-gamma_s_b)[:, None]  # (B,1)
  b = torch.sigmoid(-gamma_t_b)[:, None]  # (B,1)
  c = -torch.expm1(gamma_s_b - gamma_t_b)[:, None]  # (B,1)
  sigma_t = torch.sqrt(torch.sigmoid(gamma_t_b))[:, None]  # (B,1)

  if eps is None:
    eps = torch.randn_like(z_t)

  # Reverse update
  z_s = torch.sqrt(a / b) * (z_t - sigma_t * c * eps_hat) + torch.sqrt((1.0 - a) * c) * eps

  # x prediction at time t (continuous proxy for decoded x)
  alpha_t = torch.sqrt(1.0 - b)
  x_pred = (z_t - sigma_t * eps_hat) / alpha_t

  return z_s, x_pred


@torch.no_grad()
def sample_fn(
  model,
  N_sample: int,
  T_sample: int,
  device: Optional[torch.device] = None,
  stochastic: bool = True,
  data_stats: Optional[Dict[str, np.ndarray]] = None,
  vocab_size: int = 256,
  generator: Optional[torch.Generator] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor]]:
  """Run ancestral-style sampling from the trained model.

  Args:
    model: Trained VDM2DModel (already on the right device, in eval mode recommended).
    N_sample: Number of samples to generate in parallel.
    T_sample: Number of discretization steps from t=1 -> 0.
    device: Torch device; if None, inferred from model.
    stochastic: If True, add noise each step; if False, deterministic DDIM-like path (eps reused).
    data_stats: Optional dict with keys 'mean' and 'std' (arrays of shape (2,)) for decoding.
    vocab_size: Size of discrete vocabulary (default 256 for 8-bit).
    generator: Optional torch.Generator for reproducible randomness.

  Returns:
    z_list: List of z tensors at each step (length T_sample+1, starting from t=1 init).
    x_pred_list: List of continuous x_pred tensors per step (length T_sample).
    x_sample: Optional discrete samples (B,2) if data_stats provided, else None.
  """
  if device is None:
    try:
      device = next(model.parameters()).device
    except StopIteration:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Initialize z at t=1 from standard normal
  z_list: List[torch.Tensor] = [
    torch.randn((N_sample, 2), device=device, generator=generator)
  ]
  x_pred_list: List[torch.Tensor] = []

  for i in range(T_sample):
    # times: t -> s (descending)
    t = torch.tensor((T_sample - i) / T_sample, device=device)
    s = torch.tensor((T_sample - i - 1) / T_sample, device=device)

    if stochastic:
      # torch.randn_like may not support 'generator' in older PyTorch versions
      eps = torch.randn(
        z_list[-1].shape,
        device=z_list[-1].device,
        dtype=z_list[-1].dtype,
        generator=generator,
      )
    else:
      eps = torch.zeros_like(z_list[-1])  # deterministic path (eta=0)

    z_s, x_pred = sample_step(model, z_list[-1], t, s, eps)
    z_list.append(z_s)
    x_pred_list.append(x_pred)

  # Optional discrete decode at t=0 if stats are provided
  x_sample = None
  if data_stats is not None:
    mean = torch.as_tensor(data_stats["mean"], dtype=z_list[-1].dtype, device=device)
    std = torch.as_tensor(data_stats["std"], dtype=z_list[-1].dtype, device=device)

    def _data_decode(z_0_rescaled: torch.Tensor, gamma_0: torch.Tensor) -> torch.Tensor:
      # Build vocabulary grid and encode with provided stats
      x_vals = torch.arange(0, vocab_size, device=z_0_rescaled.device)[:, None]
      x_vals = x_vals.repeat(1, z_0_rescaled.shape[-1])  # (V, 2)

      # encode: (x - mean) / std, then reshape to (1, 2, V)
      x_vals = (x_vals - mean) / std
      x_vals = x_vals.permute(1, 0).unsqueeze(0)

      inv_stdev = torch.exp(-0.5 * gamma_0[..., None])
      logits = -0.5 * ((z_0_rescaled[..., None] - x_vals) * inv_stdev) ** 2
      return F.log_softmax(logits, dim=-1)

    gamma_0 = model.noise_schedule(torch.tensor(0.0, device=device))
    var_0 = torch.sigmoid(gamma_0)
    z_0 = z_list[-1]
    z_0_rescaled = z_0 / torch.sqrt(1.0 - var_0)
    logprobs = _data_decode(z_0_rescaled, gamma_0)
    x_sample = torch.distributions.Categorical(logits=logprobs).sample()

  return z_list, x_pred_list, x_sample


if __name__ == "__main__":
  # Minimal smoke test with a dummy model shape contract.
  # NOTE: Importing vdm2d.py directly will run training due to top-level code.
  # Prefer passing an already constructed model instance here from your training script.
  print("sampling.py loaded. Define your model and call sample_fn(model, ...) from your script.")
