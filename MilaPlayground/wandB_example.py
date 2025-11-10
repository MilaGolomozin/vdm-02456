import torch
import torch.nn as nn
import random
import wandb
from VDM_Breakdown import VDM, FixedLinearSchedule, LearnedLinearSchedule

# ============ 1️⃣ Start W&B Run ============
run = wandb.init(
    project="vdm-experiment",
    entity="s215114-danmarks-tekniske-universitet-dtu",  # <- your team or user name
    config={
        "architecture": "DummyDenoiser",
        "dataset": "toy",
        "epochs": 10,
        "learning_rate": 0.001,
        "batch_size": 2,
        "image_size": 8,
        "noise_schedule": "fixed_linear",
    }
)
cfg = wandb.config  # optional convenience handle

# ============ 2️⃣ Define Dummy Model ============
class DummyDenoiser(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, in_channels, 3, padding=1)
        )

    def forward(self, x, gamma_t):
        gamma_t = gamma_t.view(-1, 1, 1, 1)
        return self.net(x) * torch.tanh(gamma_t)

# ============ 3️⃣ Define Config and Create VDM ============
class Config:
    noise_schedule = cfg.noise_schedule
    gamma_min = -5.0
    gamma_max = 5.0
    antithetic_time_sampling = False

image_shape = (3, cfg.image_size, cfg.image_size)
model = DummyDenoiser(in_channels=3)
vdm = VDM(model=model, cfg=Config(), image_shape=image_shape)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

# ============ 4️⃣ Toy Dataset ============
def get_fake_batch(batch_size=cfg.batch_size, shape=image_shape):
    x = torch.rand(batch_size, *shape)
    return (x, None)

# ============ 5️⃣ Training Loop ============
for epoch in range(cfg.epochs):
    batch = get_fake_batch()
    loss, metrics = vdm.forward(batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log all metrics to W&B
    wandb.log({
        "epoch": epoch,
        "loss": loss.item(),
        **{f"metric/{k}": v for k, v in metrics.items()}
    })

    print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

# ============ 6️⃣ Sampling ============
samples = vdm.sample(batch_size=2, n_sample_steps=5, clip_samples=True)
wandb.log({"samples": [wandb.Image(samples, caption="Generated Samples")]})

# ============ 7️⃣ Finish Run ============
run.finish()
