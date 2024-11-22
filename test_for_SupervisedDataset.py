from transformers import CLIPImageProcessor, Qwen2Tokenizer
from train import SupervisedDataset, LazySupervisedDataset

lazydataset = LazySupervisedDataset(
    image_folder = "/data/uchiha_ssd2/fengqi/llava_dataset/CC3M_Pretrain_595K/images/",
    data_path = "/data/uchiha_ssd2/fengqi/llava_dataset/CC3M_Pretrain_595K/chat.json" ,
    image_processor= CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336"),
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
)

print("测试LazySupervisedDataset")
print(f"数据集长度  {len(lazydataset)}")
print()
first_sample = lazydataset[0]
print("第一个样本的input_ids：")
print(first_sample['input_ids'])
print()
print("第一个样本的labels：")
print(first_sample['labels'])
print()
print("第一个样本的image：")
print(first_sample['image'])
print(first_sample['image'].shape)


"""
# 下面这个似乎要加载40min
dataset = SupervisedDataset(
    image_folder = "/data/uchiha_ssd2/fengqi/llava_dataset/CC3M_Pretrain_595K/images/",
    data_path = "/data/uchiha_ssd2/fengqi/llava_dataset/CC3M_Pretrain_595K/chat.json" ,
    image_processor= CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336"),
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
)

print("测试SupervisedDataset")
print(f"数据集长度  {len(dataset)}")
print()
first_sample = lazydataset[0]
print("第一个样本的input_ids：")
print(first_sample['input_ids'])
print()
print("第一个样本的labels：")
print(first_sample['labels'])
print()
print("第一个样本的image：")
print(first_sample['image'])
print(first_sample['image'].shape)
"""