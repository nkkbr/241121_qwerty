# deepspeed test_deepspeed.py

import deepspeed
import torch

# 简单的张量加法
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    x = torch.rand(1000, 1000).to(device)
    y = torch.rand(1000, 1000).to(device)
    z = x + y
    print("Result:", z.mean().item())

if __name__ == "__main__":
    run()