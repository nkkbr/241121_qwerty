# python test_for_prepare_inputs_labels_for_multimodal.py >> test_for_prepare_inputs_labels_for_multimodal.txt 2>&1

import torch

# 设置打印选项，不省略任何元素
torch.set_printoptions(profile="default")  # 等效于 profile=None
torch.set_printoptions(threshold=float("inf"))  # 设置 threshold 为无限大

from train import SupervisedDataset,LazySupervisedDataset,DataCollatorForSupervisedDataset
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, Qwen2Tokenizer
from qwerty_qwen2 import QwertyQwen2ForCausalLM

model = QwertyQwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

special_tokens = {
    "additional_special_tokens": ["<image>"]
}
tokenizer.add_special_tokens(special_tokens)

lazydataset = LazySupervisedDataset(
    image_folder = "/data/uchiha_ssd2/fengqi/llava_dataset/CC3M_Pretrain_595K/images/",
    data_path = "/data/uchiha_ssd2/fengqi/llava_dataset/CC3M_Pretrain_595K/chat.json" ,
    image_processor = image_processor,
    tokenizer = tokenizer
)

collator = DataCollatorForSupervisedDataset(
    tokenizer=tokenizer
)

dataloader = DataLoader(lazydataset, batch_size=4, collate_fn=collator, shuffle= True)

for batch in dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    images = batch['images']

    new_position_ids, new_attention_mask, new_inputs_embeds,new_labels = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=attention_mask,
        labels=labels,
        images=images
    )
    print('=======================================================================================================')
    print("input_ids")
    print(input_ids)
    print()
    print("attention_mask")
    print(attention_mask)
    print()
    print("labels")
    print(labels)
    print()
    print('=======================================================================================================')
    print("new_position_ids:")
    print(new_position_ids)
    print()
    print('new_attention_mask:')
    print(new_attention_mask)
    print()
    print("new_labels")
    print(new_labels)
    print()
    print("new_inputs_embeds.shape")
    print(new_inputs_embeds.shape)
    break