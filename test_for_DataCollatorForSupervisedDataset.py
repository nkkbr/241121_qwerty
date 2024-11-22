from train import SupervisedDataset,LazySupervisedDataset,DataCollatorForSupervisedDataset
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, Qwen2Tokenizer

tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

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

print('测试DataCollatorForSupervisedDataset，Batch_size = 4')
print('数据集被打乱了，首先测试第一个batch的内容')
print(f"pad_token_id = {tokenizer.pad_token_id}")
print()
for batch in dataloader:
    print('input_ids:')
    print(batch['input_ids'])
    print()
    print(batch['input_ids'].shape)
    print()
    print('attention_mask:')
    print(batch['attention_mask'])
    print()
    print(batch['attention_mask'].shape)
    print()
    print('labels:')
    print(batch['labels'])
    print()
    print(batch['labels'].shape)
    print()
    print('images:')
    print(batch['images'])
    print()
    print(batch['images'].shape)
    break