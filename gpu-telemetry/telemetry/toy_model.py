"""A tiny stand-in for dots.tts: an Euler flow-matching loop with fixed work per step."""

import torch

BATCH, DIM = 256, 1024
SEED = 0


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def run_toy_model(steps: int) -> float:
    """Integrate x' = tanh(x @ W) from t=0 to 1 in `steps` Euler steps.

    `steps` plays the role of NFE: each step is one "function evaluation"
    of the same cost, so total work scales linearly with it.
    Returns a checksum of the final state, to confirm the run happened.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    device = get_device()
    # Generate on CPU with a fixed seed, then move: identical inputs on any device.
    gen = torch.Generator().manual_seed(SEED)
    x = torch.randn(BATCH, DIM, generator=gen).to(device)
    w = (torch.randn(DIM, DIM, generator=gen) / DIM**0.5).to(device)

    dt = 1.0 / steps
    for _ in range(steps):
        v = torch.tanh(x @ w)
        x = x + dt * v

    # .item() copies to the host, which also waits for any queued GPU work.
    return x.sum().item()


if __name__ == "__main__":
    print(f"device: {get_device()}")
    for steps in (8, 32):
        checksum = run_toy_model(steps)
        print(f"steps={steps:>3}  completed  checksum={checksum:.4f}")
