import torch
from torch import allclose, argmax, autograd, exp, linspace, nn, sigmoid, sqrt
import torch.nn.functional as F
import numpy as np

vocab_size=256
####__________________________
#Linear Noise Implementation
###____________________________

class LinearGammaSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t):
        """
        t: [B] or [B, 1] tensor containing times in [0,1]
        returns: gamma_t with shape [B]
        """
        return self.gamma_min + t * (self.gamma_max - self.gamma_min)



###_________________
# Our VDM
##___________________


class VDM(nn.Module):
    def __init__(self, model, image_shape, gamma_min, gamma_max):
        super().__init__()
        self.model = model
        self.image_shape = image_shape
         # Store them directly for easy access
        # self.gamma_min = gamma_min
        # self.gamma_max = gamma_max
        device = next(model.parameters()).device
        self.gamma_min = torch.as_tensor(gamma_min, dtype=torch.float32, device=device)
        self.gamma_max = torch.as_tensor(gamma_max, dtype=torch.float32, device=device)

        # self.gamma_min = torch.tensor(gamma_min)
        # self.gamma_max = torch.tensor(gamma_max)

        self.gamma = LinearGammaSchedule(gamma_min, gamma_max)

    #helper function to see what is going on on the device
    @property
    def device(self):
        return next(self.model.parameters()).device


###______________
#q(z_t|t)
#####

    def sample_q_t_0(self, x, times, noise=None): #forward diffusion (in the paper we use z_t instead of x_t but during the forward pass they are equivalent)
        """Samples from the distributions q(x_t | x_0) at the given time steps."""
        """
        x: clean image (x0)
        times: [B] float32 times in [0,1]
        """
        with torch.enable_grad():  # Need gradient to compute loss even when evaluating
            gamma_t = self.gamma(times) 

        # Pad gamma to match image shape (B, C, H, W)
        gamma_t = gamma_t[:, None, None, None]
        
        # compute alpha and sigma from gamma
        alpha = torch.sqrt(torch.sigmoid(-gamma_t))       # √(sigmoid(-γ))
        sigma = torch.sqrt(torch.sigmoid(gamma_t))        # √(sigmoid(γ))

        if noise is None:
            #raise Warning("No Noise is applied")
            noise = torch.randn_like(x)

        return alpha * x + sigma * noise, gamma_t


    ########_______________
    #sampling t
    #######################
    def sample_times(self, batch_size):
        times = torch.rand(batch_size, device=self.device)
        return times


    # ---------------------------
    # Data encoding / decoding
    # ---------------------------
    def data_encode(self,x):
        # This transforms x from discrete values (0, 1, ...)
    # to the domain (-1,1).
    # Rounding here just a safeguard to ensure the input is discrete
    # (although typically, x is a discrete variable such as uint8)
        x = x.round()
        x_mean = x.mean(dim=0)
        x_std = x.std(dim=0) + 1e-6  # avoid div by zero
        return (x - x_mean) / x_std




    def data_decode(self, z_0_rescaled, gamma_0):
        # z_0_rescaled: [B,D], gamma_0: scalar or [B]
        B, D = z_0_rescaled.shape
        x_vals = torch.arange(vocab_size, device=z_0_rescaled.device).float()[:, None]  # [vocab_size,1]
        x_vals = x_vals.repeat(1, D)  # [vocab_size, D]
        x_vals = self.data_encode(x_vals).T.unsqueeze(0)  # [1,D,vocab_size]
        inv_stdev = torch.exp(-0.5 * gamma_0[..., None]) #this is basically the inverse of the standard deviation which measures how "wide" the distribution is. if 1/std is small logits are flatter (less sharp distribution). if 1/std is large -> logits are sharper (more peaked distrib)
    #    #In Gaussian likelihoods, you always scale the residual by the inverse of the standard deviation
    #   #If the true σ is large, the same difference x−μ should be considered less surprising, so the probability should be higher.
    #   #If σ is small, even a small difference x−μ should drastically reduce the probability
        logits = -0.5 * ((z_0_rescaled[..., None] - x_vals) * inv_stdev) ** 2
        return F.log_softmax(logits, dim=-1) #Softmax normalizes the logits across discrete values (vocab_size dimension), turning them into probabilities that sum to 1.


    def data_logprob(self, x, z_0_rescaled, gamma_0):
        x = x.round().long()
        x_onehot = F.one_hot(x, num_classes=vocab_size).float()
        logprobs = self.data_decode(z_0_rescaled, gamma_0)
        return torch.sum(x_onehot * logprobs, dim=(1,2))  #Multiplies the one-hot vectors by the logits or log-probabilities → selects the probability corresponding to the true symbol.
    #Then sums over features (D) and vocab dimension to get a single scalar per batch example.


    def sample(self, batch_size, n_sample_steps=50, clip_samples=True):
        """
        Generate samples from the trained VDM model.
        
        Args:
            batch_size (int): number of samples to generate
            n_sample_steps (int): number of reverse diffusion steps
            clip_samples (bool): whether to clip samples to [-1,1] for visualization

        Returns:
            x: [B,C,H,W] sampled images
        """
        device = self.device
        B, C, H, W = batch_size, *self.image_shape
        # Start from standard normal noise
        x_t = torch.randn(batch_size, *self.image_shape, device=device)
        
        # Linear time steps from 1 to 0
        times = torch.linspace(1.0, 0.0, n_sample_steps, device=device)

        for t in times:
            t_batch = torch.full((batch_size,), t, device=device)
            gamma_t = self.gamma(t_batch)[:, None, None, None]

            # Predict noise using the model
            with torch.no_grad():
                pred_noise = self.model(x_t, gamma_t)

            # Compute alpha and sigma
            alpha = torch.sqrt(torch.sigmoid(-gamma_t))
            sigma = torch.sqrt(torch.sigmoid(gamma_t))

            # Reverse diffusion step: simple ancestral step
            x0_pred = (x_t - sigma * pred_noise) / alpha
            x_t = alpha * x0_pred + sigma * pred_noise  # update x_t

            if clip_samples:
                x_t = x_t.clamp(-1, 1)

        return x_t



    def forward(self, x, *, noise=None):
        #breakpoint()
        bpd_factor = 1 / (np.prod(x.shape[1:]) * np.log(2)) #converts the loss into bits-per-dimension.This is standard in generative modeling to report likelihood per pixel in bits (Appendix C in VDM paper).
        #making sure the input is from -1 to 1
        x1=self.data_encode(x)
        # Sample from q(x_t | x_0) with random t.
        times = self.sample_times(x.shape[0]).requires_grad_(True)
        if noise is None:
            noise = torch.randn_like(x)
        x_t, gamma_t = self.sample_q_t_0(x=x1, times=times, noise=noise) #eq 1,3,4 in the paper. still have a question regarding th SNR #this line is sampling from q(x_t|x_0) which is the encoder distribution
        model_pred = self.model(x_t, gamma_t)

        #Diffusion loss
        # Simple MSE loss weighted by the noise schedule derivative
        #breakpoint()
        mse_loss = F.mse_loss(model_pred, noise, reduction='none')
        mse_loss = mse_loss.sum(dim=(1, 2, 3))  # Sum over spatial dims
        
        # Get dγ/dt analytically from your noise schedule
        gamma_grad = self.gamma_max -self.gamma_min
        
        # Final diffusion loss
        diffusion_loss = 0.5 * mse_loss * gamma_grad * bpd_factor

        # *** Latent loss (bpd): KL divergence from q(z_1 | x) to N(0, 1)
        gamma_1 = self.gamma(torch.ones(1, device=x.device))  # gamma at t=1
        sigma_1_sq = torch.sigmoid(gamma_1)
        alpha_1_sq = 1 - sigma_1_sq  # alpha^2 = sigmoid(-gamma) = 1 - sigmoid(gamma)
        
        # Mean of q(z_1 | x) is alpha_1 * x, variance is sigma_1^2
        mean_sq = alpha_1_sq * (x1 ** 2)  # shape: (B, C, H, W)
        
        kl_std =0.5 * (sigma_1_sq+ mean_sq - 1 - torch.log(sigma_1_sq))

        latent_loss = kl_std.sum(dim=(1, 2, 3)) * bpd_factor #need it to be 1,2,3 so that we can skip the batch size

        ## *** Reconstruction loss (bpd): - E_{q(z_0 | x)} [log p(x | z_0)]. equation 11
            # Compute log p(x | z_0) for all possible values of each pixel in x.

        # Reconstruction loss
        eps_0 = torch.randn_like(x1)
       
        z_0_rescaled = x1 + torch.exp(0.5 * self.gamma_min) * eps_0
        
        # loss_recon = -self.data_logprob(x, z_0_rescaled, self.gamma_min)
        # Gaussian log-likelihood: -0.5 * ||x - z_0||^2 / sigma_0^2 (ignoring constants)
        sigma_0_sq = torch.sigmoid(self.gamma_min)
        recon_mse = ((x1 - z_0_rescaled) ** 2).sum(dim=(1, 2, 3))
        loss_recon = 0.5 * recon_mse / sigma_0_sq * bpd_factor

        # *** Overall loss in bpd. Shape (B, ).
        loss = diffusion_loss + latent_loss + loss_recon #this is the VLB the latent loss is the prior loss
        return loss.mean()






