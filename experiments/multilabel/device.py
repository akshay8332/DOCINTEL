"""Choose CPU or CUDA without hiding a missing GPU."""

from __future__ import annotations

import torch


def describe_device(require_gpu: bool = False) -> str:
    """Print the PyTorch and CUDA versions, then return ``cuda`` or ``cpu``.

    When ``require_gpu`` is set and CUDA is missing, training stops.
    """
    cuda = bool(torch.cuda.is_available())
    cuda_version = getattr(torch.version, "cuda", None)
    print(f"PyTorch {torch.__version__}", flush=True)
    print(f"CUDA version: {cuda_version}", flush=True)
    if cuda:
        name = torch.cuda.get_device_name(0)
        print(f"GPU: {name}", flush=True)
        return "cuda"
    print("GPU: none", flush=True)
    if require_gpu:
        raise RuntimeError(
            "GPU execution was required, but CUDA is not available. Training stopped."
        )
    return "cpu"
