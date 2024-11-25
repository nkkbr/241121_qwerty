from qwerty_qwen2 import QwertyQwen2ForCausalLM
import torch

model = QwertyQwen2ForCausalLM.from_pretrained(
    "/data/uchiha_ssd2/fengqi/241121_qwerty/231124_012656/checkpoint-4651",
    torch_dtype=torch.bfloat16,
    device_map = "cuda:0",
    )