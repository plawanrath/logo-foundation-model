# @plawanrath

import torch

def check_mps_support():
    """
    VErifies PyTorch MPS backend support
    """
    print(f"PyTorch version: {torch.__version__}")

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("NPS not available because the current PyTorch install was not built with MPS enabled.")
        else:
            print("NPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS enabled device on this machine.")
    else:
        print("MPS is available!")
        try:
            mps_device = torch.device("mps")
            x = torch.ones(5, device=mps_device)
            y = x * 2
            print("Successfully created a tensor on the MPS device and performed an operation.")
            print(f"Result tensor: {y}")
            print(f"Tensor device: {y.device}")
        except Exception as e:
            print(f"AN error occured while testing the MPS device: {e}")


if __name__ == "__main__":
    check_mps_support()
