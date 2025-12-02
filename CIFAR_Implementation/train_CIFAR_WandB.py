import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from ema_pytorch import EMA

# -----------------------------------------
# Import your modules
# -----------------------------------------
from VDM_our_implementation import VDM
from UNetModel import UNet


# ---------------------------------------------------------
# W&B Config (you can edit this or override from CLI)
# ---------------------------------------------------------
config_defaults = dict(
    batch_size=64,
    lr=2e-4,
    weight_decay=1e-3,
    beta1=0.9,
    beta2=0.99,
    ema_beta=0.9999,
    num_epochs=100,
    num_workers=4,
    gamma_min=-13.3,
    gamma_max=5.0,
)


def evaluate_vdm(vdm, dataloader, device):
    """Compute negative ELBO only."""
    vdm.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.enable_grad():
        for x, _ in dataloader:
            x = x.to(device)
            loss, metrics = vdm(x)
            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches


def main():
    # -----------------------------------------
    # Init W&B
    # -----------------------------------------
    wandb.init(project="VDM-training", config=config_defaults)
    cfg = wandb.config

    # -----------------------------------------
    # Device
    # -----------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------
    # Datasets + loaders
    # -----------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    val_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # -----------------------------------------
    # Model + VDM
    # -----------------------------------------
    image_shape = (3, 32, 32)
    model = UNet(in_channels=3).to(device)

    vdm = VDM(
        model=model,
        image_shape=image_shape,
        gamma_min=cfg.gamma_min,
        gamma_max=cfg.gamma_max,
    ).to(device)

    # -----------------------------------------
    # Optimizer
    # -----------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
        eps=1e-8,
    )

    # EMA wrapper
    ema = EMA(model, beta=cfg.ema_beta)

    # -----------------------------------------
    # Training Loop
    # -----------------------------------------
    for epoch in range(cfg.num_epochs):
        vdm.train()
        running_loss = 0.0

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            loss, metrics = vdm(x)
            loss.backward()
            optimizer.step()

            # update EMA
            ema.update()

            running_loss += loss.item()

            

        # mean train loss
        avg_loss = running_loss / len(train_loader)

        # Log per-batch metrics
        wandb.log({
            "train/avg_loss": avg_loss,
            **{f"train/{k}": v for k, v in metrics.items()},
            })

        # -----------------------------------------
        # Validation (EMA model)
        # -----------------------------------------
        ema_model = ema.ema_model
        ema_model.eval()
        vdm_ema = VDM(
            model=ema_model,
            image_shape=image_shape,
            gamma_min=cfg.gamma_min,
            gamma_max=cfg.gamma_max,
        ).to(device)

        val_elbo = evaluate_vdm(vdm_ema, val_loader, device)

        # -----------------------------------------
        # Logging
        # -----------------------------------------
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_avg_loss": avg_loss,
            "val/elbo": val_elbo,
        })

        print(
            f"Epoch {epoch+1}/{cfg.num_epochs} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val ELBO: {val_elbo:.4f}"
        )

    # Save final model to W&B
    torch.save(model.state_dict(), "vdm_final.pth")
    wandb.save("vdm_final.pth")

    wandb.finish()


if __name__ == "__main__":
    main()
