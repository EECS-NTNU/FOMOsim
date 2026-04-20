import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("Success: MPS device found.")
else:
    print("MPS device not found. Check macOS version (needs 12.3+).")