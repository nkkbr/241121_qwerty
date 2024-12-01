import torch
from qwerty_qwen2_update import QwertyQwen2ForCausalLM
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer, CLIPImageProcessor, Trainer
from typing import Dict, Sequence
import os
import conversation 
from PIL import Image

device = 'cuda:2'
model_name_or_path: str = "/data/uchiha_ssd2/fengqi/241121_qwerty/251124_230127/"
vision_tower_name_or_path: str = "openai/clip-vit-large-patch14-336"

model = QwertyQwen2ForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    )
model.to(device)
model.config.use_cache = True
tokenizer = Qwen2Tokenizer.from_pretrained(model_name_or_path)
image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name_or_path)

from safetensors import safe_open
folder_path = model_name_or_path
merged_weights: Dict[str, torch.Tensor] = {}
safetensors_files = [
    f for f in os.listdir(folder_path) 
    if f.endswith('.safetensors')
]

for file_name in safetensors_files:
    file_path = os.path.join(folder_path, file_name)
    with safe_open(file_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        
        for key in keys:
            if key in merged_weights:
                print(f"警告: 键 {key} 在多个文件中出现,将使用文件 {file_name} 中的值")
            tensor = f.get_tensor(key)
            merged_weights[key] = tensor

for key in model.state_dict().keys():
    if key.startswith('vision_model'):
        print(key)
        assert model.state_dict()[key].shape == merged_weights['model.' + key].shape, "未能正确加载模型参数"
        model.load_state_dict({
            **model.state_dict(),
            **{key: merged_weights['model.' + key] for key in merged_weights}
        }, strict=False)
        #model.state_dict()[key] = merged_weights['model.' + key] 据说，直接修改 state_dict() 中的值通常是不被推荐的操作，因为 state_dict() 是一个浅拷贝，而不是模型参数的直接映射。这种操作可能会导致模型参数与优化器不匹配。
    if key.startswith('mm_projector'):
        print(key)
        assert model.state_dict()[key].shape == merged_weights['model.' + key].shape, "未能正确加载模型参数"
        model.load_state_dict({
            **model.state_dict(),
            **{key: merged_weights['model.' + key] for key in merged_weights}
        }, strict=False)
        
image_path:str = "test_images/1.T.jpg"
#image_path:str = "test_images/2.G.jpg"
prompt :str = "<image>\nIs there a flag in this picture?"

"""
image_path = '/data/uchiha_ssd2/fengqi/llava_dataset/COCO/train2017/000000353197.jpg'
prompt = 'What do you see happening in this image?\n<image>'
"""

cur_conv = conversation.conv_qwen2_5.copy()
cur_image = Image.open(image_path)
image = image_processor(cur_image, return_tensors='pt')['pixel_values']
cur_conv.append_message(['USER',(prompt,cur_image)])
text = cur_conv.get_prompt()
input_ids = tokenizer(text,return_tensors="pt",add_special_tokens=False)['input_ids'][0]
input_ids = input_ids.unsqueeze(0)
attention_mask = torch.ones_like(input_ids)
labels = torch.arange(input_ids.shape[-1]).unsqueeze(0)

image = image.to(device)
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
labels = labels.to(device)
output_ids = model.generate(
    inputs=input_ids,       # 输入 tokens
    max_length=2048,                      
    num_return_sequences=1,             # 返回生成的序列数
    temperature=0.7,                    # 控制生成的多样性
    top_k=50,                           # 限制最高概率的 K 个标记
    top_p=0.95,                         # 过滤累积概率小于 P 的标记
    do_sample=True,                     # 使用采样生成（而非贪心算法）
    images=image,
    attention_mask=attention_mask,
    labels=labels,
    use_cache=True
)

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"生成的文本: {generated_text}")

###失败了，没有能够成功运行