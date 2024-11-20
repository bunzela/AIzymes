import torch
import os

def setup_torch():
    reset_cuda()
    torch.manual_seed(29)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    SMOKE_TEST = os.environ.get("SMOKE_TEST")

    return device, dtype, SMOKE_TEST

def reset_cuda():
    torch.cuda.empty_cache() # Clear cache
    for obj in list(locals().values()): # Optionally, you can reset all GPU tensors
        if torch.is_tensor(obj):
            del obj
    torch.cuda.empty_cache()