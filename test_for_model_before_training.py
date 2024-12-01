import torch
from qwerty_qwen2_update import QwertyQwen2ForCausalLM
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer, CLIPImageProcessor, Trainer
from typing import Dict, Sequence
import os
import conversation 
from PIL import Image

device = 'cuda:2'
model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
vision_tower_name_or_path: str = "openai/clip-vit-large-patch14-336"

model = QwertyQwen2ForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    )
model.to(device)
model.config.use_cache = True
tokenizer = Qwen2Tokenizer.from_pretrained(model_name_or_path)
image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name_or_path)

special_tokens = {
    "additional_special_tokens": ["<image>"]
}
tokenizer.add_special_tokens(special_tokens)

# 同时，我们也需要修改embedding的初始值，在"Qwen/Qwen2.5-7B-Instruct"中，我们可以看到，新增加的<image>的token id应该是 151665， 
# 但是其input_embeddings 和 output_embeddings 分别是类似于 [ 1.1755e-37,  1.1755e-37, -1.1755e-37,  1.1755e-37,  1.1755e-37, ...] 和 [ 0.0018,  0.0081,  0.0044, -0.0008,  0.0001 ] 的值
# 如果没有正常初始化，会造成模型无法或难以学习到这个新的token应该有的embedings，因为 1.1755e-37 几乎就是 0
# "Qwen/Qwen2.5-7B-Instruct" 的embedding是有些奇特的，它有效的embedding的数量仅有 tokenizer.vocab_size = 151643，外加少量特殊的token。其余的embedding仅仅是填充在那里，等着备用。这样有新的special token的话也不需要再resize embedding的矩阵了。

input_embeddings = model.get_input_embeddings().weight.data
output_embeddings= model.get_output_embeddings().weight.data

hidden_size = model.get_input_embeddings().weight.data.shape[-1]
effective_token_number = tokenizer.vocab_size # 151643
image_token_id = tokenizer.encode("<image>")[0] # 151665

input_embeddings_avg = input_embeddings[:effective_token_number].mean(dim=0, keepdim=True)
output_embeddings_avg = output_embeddings[:effective_token_number].mean(dim=0, keepdim=True)

input_embeddings[image_token_id] = input_embeddings_avg
output_embeddings[image_token_id] = output_embeddings_avg
        
image_path:str = "test_images/1.T.jpg"
#image_path:str = "test_images/2.G.jpg"
prompt :str = "<image>\nPlease describe this picture in detal."

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