from qwerty_qwen2 import QwertyQwen2Model
from transformers import AutoTokenizer

model = QwertyQwen2Model.from_pretrained("/data/uchiha_ssd2/fengqi/241121_qwerty/231124_012656/checkpoint-4651")
tokenizer = AutoTokenizer.from_pretrained("/data/uchiha_ssd2/fengqi/241121_qwerty/231124_012656/checkpoint-4651")

print(model.state_dict())