# deepspeed test_deepspeed_1.py

import torch
import torch.distributed as dist
import os

# 初始化分布式
dist.init_process_group(backend='nccl')
local_rank = int(os.getenv('LOCAL_RANK', 0))
torch.cuda.set_device(local_rank)

# 张量计算
device = torch.device(f"cuda:{local_rank}")
x = torch.rand(1000, 1000).to(device)
y = torch.rand(1000, 1000).to(device)
z = x + y

# 汇总结果
z_mean = z.mean()
z_mean_tensor = torch.tensor(z_mean.item(), device=device)
dist.all_reduce(z_mean_tensor, op=dist.ReduceOp.SUM)
z_mean_tensor /= dist.get_world_size()

# 输出汇总后的结果（仅 rank 0 输出）
if local_rank == 0:
    print(f"Summed Result: {z_mean_tensor.item()}")
