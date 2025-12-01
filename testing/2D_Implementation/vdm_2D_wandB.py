import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import wandb

# ---------------------------
# Initialize WandB
# ---------------------------
wandb.init(
    project="VDM_2D",
    entity="s215114-danmarks-tekniske-universitet-dtu",
    config={
        "N": 1024,
        "learning_rate": 1e-3,
        "hidden_units": 512,
        "num_steps": 20000
    }
)
config = wandb.config


# ----------------------------------------------------
# (YOUR CODE) — Everything from here is unchanged
# ----------------------------------------------------
N = config.N
init_gamma_0 = -13.3
init_gamma_1 = 5.
hidden_units = config.hidden_units
T_train = 0
vocab_size = 256
learning_rate = config.learning_rate
num_train_steps = config.num_steps
device = torch.device('cpu')


class Base2FourierFeatures(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        freqs = 2.0 ** torch.arange(num_frequencies)
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        B, D = x.shape
        w = self.freqs[None, :, None] * 2 * np.pi
        h = x.unsqueeze(1) * w
        sin_feat = torch.sin(h)
        cos_feat = torch.cos(h)
        h = torch.cat([sin_feat, cos_feat], dim=1)
        return h.flatten(1)


class ScoreNetwork(nn.Module):
    def __init__(self, input_dim, hidden_units, num_frequencies=16):
        super().__init__()
        ff_out_dim = input_dim * 2 * num_frequencies
        self.ff = Base2FourierFeatures(num_frequencies)
        self.dense1 = nn.Linear(input_dim + 1 + ff_out_dim, hidden_units)
        self.dense2 = nn.Linear(hidden_units, hidden_units)
        self.dense3 = nn.Linear(hidden_units, input_dim)
        self.act = nn.SiLU()

    def forward(self, z, gamma_t):
        B, D = z.shape
        gamma_t_norm = ((gamma_t - init_gamma_0) / (init_gamma_1 - init_gamma_0)) * 2 - 1
        h_ff = self.ff(z)
        h = torch.cat([z, gamma_t_norm.unsqueeze(1), h_ff], dim=1)
        h = self.act(self.dense1(h))
        h = self.act(self.dense2(h))
        h = self.dense3(h)
        return h


class NoiseSchedule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.full((1,), init_gamma_1 - init_gamma_0))
        self.b = nn.Parameter(torch.full((1,), init_gamma_0))

    def forward(self, t):
        return torch.abs(self.w) * t + self.b


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
# Data encode/decode
# ---------------------------
def data_encode(x):
    x = x.round()
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0) + 1e-6
    return (x - x_mean) / x_std


def data_decode(z_0_rescaled, gamma_0):
    B, D = z_0_rescaled.shape
    x_vals = torch.arange(vocab_size, device=z_0_rescaled.device).float()[:, None]
    x_vals = x_vals.repeat(1, D)
    x_vals = data_encode(x_vals).T.unsqueeze(0)
    inv_stdev = torch.exp(-0.5 * gamma_0[..., None])
    logits = -0.5 * ((z_0_rescaled[..., None] - x_vals) * inv_stdev) ** 2
    return F.log_softmax(logits, dim=-1)


def data_logprob(x, z_0_rescaled, gamma_0):
    x = x.round().long()
    x_onehot = F.one_hot(x, num_classes=vocab_size).float()
    logprobs = data_decode(z_0_rescaled, gamma_0)
    return torch.sum(x_onehot * logprobs, dim=(1,2))


def data_generate_x(z_0, gamma_0):
    var_0 = torch.sigmoid(gamma_0)
    z_0_rescaled = z_0 / torch.sqrt(1. - var_0).unsqueeze(1)
    logits = data_decode(z_0_rescaled, gamma_0)
    logits = logits.permute(0,2,1)
    dist = torch.distributions.Categorical(logits=logits)
    samples = dist.sample().T.float()
    return samples


# ---------------------------
# Loss
# ---------------------------
def loss_fn(model, x, T_train=0):
    B, D = x.shape
    gamma_0 = model.gamma(torch.tensor(0., device=x.device))
    gamma_1 = model.gamma(torch.tensor(1., device=x.device))
    var_0 = torch.sigmoid(gamma_0)
    var_1 = torch.sigmoid(gamma_1)

    f = data_encode(x)

    eps_0 = torch.randn_like(f)
    z_0_rescaled = f + torch.exp(0.5 * gamma_0) * eps_0
    loss_recon = -data_logprob(x, z_0_rescaled, gamma_0)

    mean1_sqr = (1. - var_1) * f**2
    loss_klz = 0.5 * torch.sum(mean1_sqr + var_1 - torch.log(var_1) - 1., dim=1)

    t = torch.rand(B, device=x.device)
    gamma_t = model.gamma(t)
    var_t = torch.sigmoid(gamma_t).unsqueeze(1)
    eps = torch.randn_like(f)
    z_t = torch.sqrt(1. - var_t) * f + torch.sqrt(var_t) * eps
    eps_hat = model.score(z_t, gamma_t)
    loss_diff_mse = torch.sum((eps - eps_hat)**2, dim=1)

    t.requires_grad_(True)
    grad = torch.autograd.grad(
        outputs=model.gamma(t).sum(),
        inputs=t,
        create_graph=True
    )[0]
    loss_diff = 0.5 * grad * loss_diff_mse

    rescale = 1. / (D * np.log(2.))
    bpd_latent = torch.mean(loss_klz) * rescale
    bpd_recon = torch.mean(loss_recon) * rescale
    bpd_diff = torch.mean(loss_diff) * rescale
    bpd = bpd_latent + bpd_recon + bpd_diff

    return bpd, [bpd_latent, bpd_recon, bpd_diff]


# ---------------------------
# Dataset: 2D swirl
# ---------------------------
theta = np.sqrt(np.random.rand(N)) * 3 * np.pi
r_a = 2 * theta + np.pi
x_np = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
x_np = 4 * (x_np + .25*np.random.randn(N,2) + 30)
x = torch.tensor(x_np, dtype=torch.float32, device=device)


# ---------------------------
# Training
# ---------------------------
model = Model(input_dim=2, hidden_units=hidden_units).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(num_train_steps):
    model.train()
    optimizer.zero_grad()

    loss, (b_latent, b_recon, b_diff) = loss_fn(model, x)
    loss.backward()
    optimizer.step()

    wandb.log({
        "total_bpd": loss.item(),
        "latent_bpd": b_latent.item(),
        "recon_bpd": b_recon.item(),
        "diff_bpd": b_diff.item(),
        "gamma_w": model.noise_schedule.w.item(),
        "gamma_b": model.noise_schedule.b.item()
    })

    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

    #     # visualize γ(t)
    #     t_vis = torch.linspace(0,1,200)
    #     gamma_vis = model.gamma(t_vis).detach().cpu().numpy()
    #     wandb.log({"gamma_curve": wandb.Image(plt.plot(t_vis, gamma_vis))})
    #     plt.clf()

    #     # visualize raw data
    #     plt.scatter(x_np[:,0], x_np[:,1], s=2)
    #     wandb.log({"dataset": wandb.Image(plt)})
    #     plt.clf()

print("Training complete!")
wandb.finish()
