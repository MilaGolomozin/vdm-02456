"""
This was AI generated code for testing if the fourier features in UNetModel work as intended.
"""

import argparse
import importlib.util
from pathlib import Path

import torch

# Load UNetFourier directly from file to avoid package/path issues
THIS_DIR = Path(__file__).parent
MODEL_PATH = THIS_DIR / "UNetModelFourier.py"
spec = importlib.util.spec_from_file_location("unet_fourier", str(MODEL_PATH))
mod = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None, "Failed to load UNetModelFourier.py"
spec.loader.exec_module(mod)
UNetFourier = mod.UNetFourier


def main():
    parser = argparse.ArgumentParser(description="UNetFourier quick shape check", add_help=True)
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--channels", type=int, default=3, help="Input/output channels")
    parser.add_argument("--height", type=int, default=32, help="Image height")
    parser.add_argument("--width", type=int, default=32, help="Image width")
    parser.add_argument("--emb", type=int, default=128, help="Time embedding dim")
    parser.add_argument("--base", type=int, default=64, help="Base channels")
    # Be tolerant to unknown args (e.g., when run via certain VS Code run modes)
    args, _unknown = parser.parse_known_args()

    B, C, H, W = args.batch, args.channels, args.height, args.width

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetFourier(
        in_channels=C,
        out_channels=C,
        time_emb_dim=args.emb,
        base_channels=args.base,
    ).to(device)
    model.eval()

    x = torch.randn(B, C, H, W, device=device)
    # gamma_t can be shape [B] or [B,1,1,1]; we use [B]
    gamma_t = torch.rand(B, device=device)

    with torch.no_grad():
        y = model(x, gamma_t)

    print(f"Input:  {tuple(x.shape)} on {device}")
    print(f"Output: {tuple(y.shape)}")
    if y.shape == x.shape:
        print("Result: OK — output matches input shape.")
    else:
        print("Result: MISMATCH — check channels/heights/widths.")


if __name__ == "__main__":
    main()
