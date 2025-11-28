import numpy as np
import torch
from torch import allclose, argmax, autograd, exp, linspace, nn, sigmoid, sqrt
from torch.special import expm1
from tqdm import trange
from utils import maybe_unpack_batch, unsqueeze_right

#NB:In most VDM implemntations we have the option of adding noise ourselves and also learning the addition of notes
#The fixed schedule of noise is used to sample time steps during training
#You use the learned γₑ(t) for computing the model’s own predicted variance function, not for generating noisy inputs directly.

## Noise schedule with fixed linear interpolation between min and max gamma values.
class FixedLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, t):
        return self.gamma_min + (self.gamma_max - self.gamma_min) * t

#this is the trainable noise addition 
#It’s just a linear function of time, but now its slope (w) and offset (b) are trainable parameters instead of fixed constants.
#the .abs() ensures the schedule stays monotonic increasing (so noise doesn’t go backward as time progresses).

#it is important to remember that the learned linear schedule is computed to scale alpha and the SNR, so it does not add noise itself
#So the “forward pass” here doesn’t add noise itself — it just gives you the schedule (γ values) that tell you how much noise to add.
#it does not have an activation function so that the learning remains more scalable
class LearnedLinearSchedule(nn.Module):
    def __init__(self, gamma_min, gamma_max):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(gamma_min))
        self.w = nn.Parameter(torch.tensor(gamma_max - gamma_min))

    def forward(self, t):
        return self.b + self.w.abs() * t


class VDM(nn.Module): ## All code following lives under this class
    def __init__(self, model, cfg, image_shape):
        super().__init__()
        self.model = model
        self.cfg = cfg  # needed it breaks the implementation
        self.image_shape = image_shape
        self.vocab_size = 256
        if cfg.noise_schedule == "fixed_linear":
            self.gamma = FixedLinearSchedule(cfg.gamma_min, cfg.gamma_max)
        elif cfg.noise_schedule == "learned_linear":
            self.gamma = LearnedLinearSchedule(cfg.gamma_min, cfg.gamma_max)
        else:
            raise ValueError(f"Unknown noise schedule {cfg.noise_schedule}")

    #helper function to see what is going on on the device
    @property
    def device(self):
        return next(self.model.parameters()).device
    

    """
In VDMs, we can think of this process as progressively denoising a noisy sample (with noise at time t) 
by leveraging the reverse diffusion process, which is governed by parameters gamma_t and gamma_s
So the probability tries to predict x from its noisy version z_t, where z_s represent the target timestamp/result
    """


    @torch.no_grad()
    def sample_p_s_t(self, z, t, s, clip_samples):
        """Samples from p(z_s | z_t, x). Used for standard ancestral sampling."""
        gamma_t = self.gamma(t) #the noise at the timestamp t
        gamma_s = self.gamma(s) #the level of noise at the time stamp s (s in this case is the desired timestamp)
        c = -expm1(gamma_s - gamma_t) #the difference in the noise strength between the two times.
        alpha_t = sqrt(sigmoid(-gamma_t))
        alpha_s = sqrt(sigmoid(-gamma_s))
        sigma_t = sqrt(sigmoid(gamma_t))
        sigma_s = sqrt(sigmoid(gamma_s))

        #The model predicts the noise (or residual. a residual is the difference between the tru value and a model's prediction)
        # #for the sample z at time t, which we will use for reconstructing the clean sample.
        pred_noise = self.model(z, gamma_t)
        
        if clip_samples:
            x_start = (z - sigma_t * pred_noise) / alpha_t #this is formula number 10 in the vdm paper
            x_start.clamp_(-1.0, 1.0)
            mean = alpha_s * (z * (1 - c) / alpha_t + c * x_start) #??
        else:
            mean = alpha_s / alpha_t * (z - c * sigma_t * pred_noise) #Appendix A.4
        scale = sigma_s * sqrt(c)
        return mean + scale * torch.randn_like(z) #reparametrization
    
    @torch.no_grad()
    def sample(self, batch_size, n_sample_steps, clip_samples): #reverse diffusion
        z = torch.randn((batch_size, *self.image_shape), device=self.device) #Each element of z is drawn independently from a Gaussian with mean 0 and variance 1.
        #this z represents the latent variable at the last step of the forward pass as it is pure noise, meaning i does not contain x
        #shape of z is the batch size and the unpacked dimentions of the images so: B, C, H, W
        steps = linspace(1.0, 0.0, n_sample_steps + 1, device=self.device) #created a sequence of time steps from 1 back to 0
        #in the loop below we are donoising the current sample 
        for i in trange(n_sample_steps, desc="sampling"):
            z = self.sample_p_s_t(z, steps[i], steps[i + 1], clip_samples) #Appendix A.4 as well
        #after the loop, z is an estiamte of the clean sample z_0
        #logprobs below omputes the log-likelihood of the discrete image values given the predicted latent z_0 so log(p(x|z_0))
        logprobs = self.log_probs_x_z0(z_0=z)  # (B, C, H, W, vocab_size)
        #Since images are represented as discrete tokens (or pixel bins), the argmax converts the probability distribution over the discrete vocabulary into the actual discrete image.
        x = argmax(logprobs, dim=-1)  # (B, C, H, W)
        return x.float() / (self.vocab_size - 1)  # normalize to [0, 1]
    
    def sample_q_t_0(self, x, times, noise=None): #forward diffusion (in the paper we use z_t instead of x_t but during the forward pass they are equivalent)
        """Samples from the distributions q(x_t | x_0) at the given time steps."""
        with torch.enable_grad():  # Need gradient to compute loss even when evaluating
            gamma_t = self.gamma(times) 

        gamma_t_padded = unsqueeze_right(gamma_t, x.ndim - gamma_t.ndim)
        mean = x * sqrt(sigmoid(-gamma_t_padded))  # x * alpha
        scale = sqrt(sigmoid(gamma_t_padded))
        if noise is None:
            noise = torch.randn_like(x)
        return mean + noise * scale, gamma_t
    
    #This function sample_times is responsible for sampling the time steps t at which you will compute the forward diffusion samples (x_t or z_t).

    def sample_times(self, batch_size):
        if self.cfg.antithetic_time_sampling:
            t0 = np.random.uniform(0, 1 / batch_size)
            times = torch.arange(t0, 1.0, 1.0 / batch_size, device=self.device)
        else:
            times = torch.rand(batch_size, device=self.device)
        return times
    
    def kl_std_normal(self, mean_squared, var):
        return 0.5 * (var + mean_squared - torch.log(var.clamp(min=1e-15)) - 1.0)
    

    def log_probs_x_z0(self, x=None, z_0=None):
        """Computes log p(x | z_0) for all possible values of x.

        Compute p(x_i | z_0i), with i = pixel index, for all possible values of x_i in
        the vocabulary. We approximate this with q(z_0i | x_i). Unnormalized logits are:
            -1/2 SNR_0 (z_0 / alpha_0 - k)^2
        where k takes all possible x_i values. Logits are then normalized to logprobs.

        The method returns a tensor of shape (B, C, H, W, vocab_size) containing, for
        each pixel, the log probabilities for all `vocab_size` possible values of that
        pixel. The output sums to 1 over the last dimension.

        The method accepts either `x` or `z_0` as input. If `z_0` is given, it is used
        directly. If `x` is given, a sample z_0 is drawn from q(z_0 | x). It's more
        efficient to pass `x` directly, if available.

        Args:
            x: Input image, shape (B, C, H, W).
            z_0: z_0 to be decoded, shape (B, C, H, W).

        Returns:
            log_probs: Log probabilities of shape (B, C, H, W, vocab_size).
        """
        gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))
        if x is None and z_0 is not None:
            z_0_rescaled = z_0 / sqrt(sigmoid(-gamma_0))  # z_0 / alpha_0
        elif z_0 is None and x is not None:
            # Equal to z_0/alpha_0 with z_0 sampled from q(z_0 | x)
            z_0_rescaled = x + exp(0.5 * gamma_0) * torch.randn_like(x)  # (B, C, H, W)
        else:
            raise ValueError("Must provide either x or z_0, not both.")
        z_0_rescaled = z_0_rescaled.unsqueeze(-1)  # (B, C, H, W, 1)
        x_lim = 1 - 1 / self.vocab_size
        x_values = linspace(-x_lim, x_lim, self.vocab_size, device=self.device)
        logits = -0.5 * exp(-gamma_0) * (z_0_rescaled - x_values) ** 2  # broadcast x
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, C, H, W, vocab_size)
        return log_probs
    
    def forward(self, batch, *, noise=None):
        x, label = maybe_unpack_batch(batch)
        assert x.shape[1:] == self.image_shape
        assert 0.0 <= x.min() and x.max() <= 1.0

        bpd_factor = 1 / (np.prod(x.shape[1:]) * np.log(2)) #converts the loss into bits-per-dimension.This is standard in generative modeling to report likelihood per pixel in bits (Appendix C in VDM paper).

        # Convert image to integers in range [0, vocab_size - 1].
        img_int = torch.round(x * (self.vocab_size - 1)).long()
        print("The shape after the transfer to image_int:", img_int.shape)
        #print("x after vocab size scaling: \n", img_int)
        assert (img_int >= 0).all() and (img_int <= self.vocab_size - 1).all()
        # Check that the image was discrete with vocab_size values.
        assert allclose(img_int / (self.vocab_size - 1), x, atol=1e-2) #here we are adding a tolerance. example below why we need it

        """
        x[0,0,0,0] = 0.12345
        img_int[0,0,0,0] = round(0.12345 * 255) = 31
        img_int[0,0,0,0] / 255 = 0.12157 != 0.12345

        """

        # Rescale integer image to [-1 + 1/vocab_size, 1 - 1/vocab_size]
        x = 2 * ((img_int + 0.5) / self.vocab_size) - 1 #in the forward diffusion step we want the range from -1 to 1, the original input is from 0,1 so we transfer from image into to -1 to 1 range



        # Sample from q(x_t | x_0) with random t.
        times = self.sample_times(x.shape[0]).requires_grad_(True)
        if noise is None:
            noise = torch.randn_like(x)
        x_t, gamma_t = self.sample_q_t_0(x=x, times=times, noise=noise) #eq 1,3,4 in the paper. still have a question regarding th SNR #this line is sampling from q(x_t|x_0) which is the encoder distribution
        
        # Forward through model
        #this is calling the model that we pass into the vdm for noise prediction
        #typically a cnn or a u-net
        #what it does is it takes into the noisy input x_t and the noise level gamma_t and comes up with a noise estimate
        model_out = self.model(x_t, gamma_t) #The model is trained to predict the noise so that reverse diffusion can reconstruct the original image 
        #this is the reverse process so the decoder part (calling the model)

        # *** Diffusion loss (bpd)
        gamma_grad = autograd.grad(  # gamma_grad shape: (B, ) #computes the gradient of the function gamma Appendix H
            gamma_t,  # (B, )
            times,  # (B, )
            grad_outputs=torch.ones_like(gamma_t),
            create_graph=True,
            retain_graph=True,
        )[0] #computes the differential of gamma_t wrt t. equivalent to exp(gamma_s-gamma_t) which is exp(SNR_s - SNR_t) in eq 13 and eq14
        pred_loss = ((model_out - noise) ** 2).sum((1, 2, 3))  # (B, ) #difference between real noise and the predicted one #sums over all chnaels for each batch element #the ||eps-eps_hat||^2 in the eq 14
        diffusion_loss = 0.5 * pred_loss * gamma_grad * bpd_factor

        # *** Latent loss (bpd): KL divergence from N(0, 1) to q(z_1 | x)
        #Implements the KL divergence between q(z_1 | x) and standard normal p(z_1): or ??
        gamma_1 = self.gamma(torch.tensor([1.0], device=self.device))
        sigma_1_sq = sigmoid(gamma_1)
        mean_sq = (1 - sigma_1_sq) * x**2  # (alpha_1 * x)**2
        latent_loss =self.kl_std_normal(mean_sq, sigma_1_sq).sum((1, 2, 3)) * bpd_factor #this id the KL term (prior loss) in the VLB frmula in equation 11

        # *** Reconstruction loss (bpd): - E_{q(z_0 | x)} [log p(x | z_0)]. equation 11
        # Compute log p(x | z_0) for all possible values of each pixel in x.


        """
        When the decoder tries to reconstruct an image, it does not output a single value for each pixel like “the pixel should be 0.73.”
        Instead, it outputs a probability distribution over all possible pixel values.
        Let say you have grayscale pixels that can take integer values from 0 to 255.
        That means there are 256 possible values, so vocab_size = 256.
        for each pixel the ourput of the decode is a vector which contains 256 probabilities of how much the decoder believes that the pixel holds a particular value true
        The model says:

        “I believe there is a 10% chance the pixel intensity is 254,
        and a 2% chance it is 255, etc.”
        this is shown in the line below, which is the final stage of the decoder. it evaluates the likelihood of the reconstructed data
        """


        log_probs = self.log_probs_x_z0(x)  # (B, C, H, W, vocab_size)
        #So for every image (B), every color channel (C), and every pixel position (H, W),
        #you get a vector of length vocab_size (e.g., 256) that holds the log-probabilities
        #of all possible discrete pixel values.


        """
        earlier we represented the values of each pixel in the image as integers
        so now we use that representation to create a one hot encoding for the real value of each pixel where we have 1 for the actual value and 0 for all the rest
        """

        # One-hot representation of original image. Shape: (B, C, H, W, vocab_size).
        x_one_hot = torch.zeros((*x.shape, self.vocab_size), device=self.device)
        x_one_hot.scatter_(4, img_int.unsqueeze(-1), 1)  # one-hot over last dim
        # Select the correct log probabilities.
        log_probs = (x_one_hot * log_probs).sum(-1)  # (B, C, H, W)
        # Overall logprob for each image in batch.
        recons_loss = -log_probs.sum((1, 2, 3)) * bpd_factor

        # *** Overall loss in bpd. Shape (B, ).
        loss = diffusion_loss + latent_loss + recons_loss #this is the VLB the latent loss is the prior loss

        with torch.no_grad():
            gamma_0 = self.gamma(torch.tensor([0.0], device=self.device))
        metrics = {
            "bpd": loss.mean(),
            "diff_loss": diffusion_loss.mean(),
            "latent_loss": latent_loss.mean(),
            "loss_recon": recons_loss.mean(),
            "gamma_0": gamma_0.item(),
            "gamma_1": gamma_1.item(),
        }
        return loss.mean(), metrics


    