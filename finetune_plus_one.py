from dataclasses import dataclass,field
import transformers
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer, CLIPImageProcessor, Trainer
from torch.utils.data import Dataset
import json
import conversation 
import torch
import os
from PIL import Image
from tqdm.auto import tqdm
from typing import Dict, Sequence
from datetime import datetime
from qwerty_qwen2 import QwertyQwen2ForCausalLM

current_time = datetime.now().strftime("%d%m%y_%H%M%S")
output_dir = os.path.join(os.getcwd(), current_time) 
logging_dir = os.path.join(output_dir, 'logs')
#tokenizer_dir = os.path.join(output_dir, 'tokenizer')
os.makedirs(output_dir, exist_ok=True) 
os.makedirs(logging_dir, exist_ok=True) 
#os.makedirs(tokenizer_dir, exist_ok=True) 

local_rank = None # 其实我也还是不太知道这个具体是如何产生作用的

@dataclass
class ModelArguments:
    model_name_or_path: str = "/data/uchiha_ssd2/fengqi/241121_qwerty/251124_230127/" # 第二阶段训练的结果保存在这里
    vision_tower_name_or_path: str = "openai/clip-vit-large-patch14-336"

@dataclass
class DataArguments:
    image_folder: str = "/data/uchiha_ssd2/fengqi/llava_dataset/COCO/train2017"
    data_path: str = "/data/uchiha_ssd2/fengqi/llava_dataset/COCO/llava_instruct_150k.json" 


@dataclass
class TrainingArguments(transformers.TrainingArguments):  
    output_dir: str = output_dir
    logging_dir: str = logging_dir
    logging_steps: int = 1
    log_level: str = "debug"

"""
SupervisedDataset 和 LazySupervisedDataset 选用一个就可以
"""
class SupervisedDataset(Dataset):

    def __init__(
            self,
            data_path: str,
            image_folder: str,
            tokenizer: transformers.PreTrainedTokenizer,
            image_processor: transformers.CLIPImageProcessor
            ):
        self.input_ids = []
        self.labels = []
        self.images = []

        with open(data_path,'r') as f:
            data_list = json.load(f)
        for cur_data in tqdm(data_list, desc="Loading data", ncols=80):
            cur_conv = conversation.conv_qwen2_5.copy()

            cur_image = os.path.join(image_folder,cur_data['image'])
            cur_image = Image.open(cur_image)
            image = image_processor(cur_image, return_tensors='pt')['pixel_values']
            self.images.append(image)

            for idx, conv in enumerate(cur_data['conversations']):
                if idx % 2 == 0:
                    if '<image>' in conv['value']:
                        cur_conv.append_message(['USER',(conv['value'],cur_image)])
                    else:
                        cur_conv.append_message(['USER',conv['value']])
                else:
                    cur_conv.append_message(['ASSISTANT',conv['value']])
            text = cur_conv.get_prompt()
            input_ids = tokenizer(text,return_tensors="pt",add_special_tokens=False)['input_ids'][0]
            self.input_ids.append(input_ids)

            labels = convert_input_ids_to_labels_for_qwen2(input_ids)
            self.labels.append(labels)

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            images = self.images[i]
            )
        
class LazySupervisedDataset(Dataset):

    def __init__(
            self,
            data_path: str,
            image_folder: str,
            tokenizer: transformers.PreTrainedTokenizer,
            image_processor: transformers.CLIPImageProcessor
            ):
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        with open(data_path,'r') as f:
            self.data_list = json.load(f)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, i):
        cur_data = self.data_list[i]

        cur_conv = conversation.conv_qwen2_5.copy()

        cur_image = os.path.join(self.image_folder,cur_data['image'])
        cur_image = Image.open(cur_image)
        image = self.image_processor(cur_image, return_tensors='pt')['pixel_values']

        for idx, conv in enumerate(cur_data['conversations']):
            if idx % 2 == 0:
                if '<image>' in conv['value']:
                    cur_conv.append_message(['USER',(conv['value'],cur_image)])
                else:
                    cur_conv.append_message(['USER',conv['value']])
            else:
                cur_conv.append_message(['ASSISTANT',conv['value']])
        text = cur_conv.get_prompt()
        # print(repr(text)) # 测试用
        input_ids = self.tokenizer(text,return_tensors="pt",add_special_tokens=False)['input_ids'][0]

        labels = convert_input_ids_to_labels_for_qwen2(input_ids)

        return dict(
            input_ids=input_ids,
            labels=labels,
            images=image
            )
        

class DataCollatorForSupervisedDataset:

    def __init__(
            self,
            tokenizer: transformers.PreTrainedTokenizer
        ):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(
            self,
            batch:Sequence[Dict]
        ):
        input_ids, labels, images = tuple([item[key] for item in batch] for key in ['input_ids', 'labels', 'images'])
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        images = torch.cat(images,dim=0)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.pad_token_id),
            images=images
        )


def convert_input_ids_to_labels_for_qwen2(input_tensor):
    """
    这个函数的输入是一个input_ids,它是用Qwen2.5的模版包装好了的对话，再tokenize后的结果，例如：

    tensor([151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
            151645,    198, 151644,    872,    198, 151665,   9612,  21927,     13,
            151645,    198, 151644,  77091,    198,  25699,     32,     13, 151645,
            198, 151644,    872,    198,  73442,     33,     13, 151645,    198,
            151644,  77091,    198,   1359,     68,     13, 151645])

    它的各项是：

          151644     '<|im_start|>'
            8948     'system'
             198     '\n'
            2610     'You'
             525     ' are'
             264     ' a'
           10950     ' helpful'
           17847     ' assistant'
              13     '.'
          151645     '<|im_end|>'
             198     '\n'
          151644     '<|im_start|>'
             872     'user'
             198     '\n'
          151665     '<image>'
            9612     '/n'
           21927     ' Hello'
              13     '.'
          151645     '<|im_end|>'
             198     '\n'
          151644     '<|im_start|>'
           77091     'assistant'
             198     '\n'
           25699     'AAAA'
              32     'A'
              13     '.'
          151645     '<|im_end|>'
             198     '\n'
          151644     '<|im_start|>'
             872     'user'
             198     '\n'
           73442     'BBBB'
              33     'B'
              13     '.'
          151645     '<|im_end|>'
             198     '\n'
          151644     '<|im_start|>'
           77091     'assistant'
             198     '\n'
            1359     'By'
              68     'e'
              13     '.'
          151645     '<|im_end|>'

        我们微调的逻辑是，assistant的回答，我们要计算损失，其余的都不计算损失。assistant的回答后面的 '<|im_end|>' 也应该计算损失，这样模型可以学习到应该在什么位置停止继续生成token。
        assistant 的话都以

          151644     '<|im_start|>'
           77091     'assistant'
             198     '\n'

        开始，并以

          151645     '<|im_end|>'

        结束。我们要做的就是将这两者之间的，包含151645，保留，其余的都设置成 -100。例如，上面这个tensor，被处理后应该成为：

    tensor([  -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            -100,   -100,   -100,   -100,   -100,  25699,     32,     13, 151645,
            -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,   -100,
            -100,   -100,   -100,   1359,     68,     13, 151645])

    这就是这个函数的功能。
    """
    result = torch.full_like(input_tensor, -100)

    pattern = torch.tensor([151644, 77091, 198])
    pattern_len = len(pattern)
    
    matches = (input_tensor.unfold(0, pattern_len, 1) == pattern).all(dim=1).nonzero(as_tuple=True)[0]
    
    for start_idx in matches:
        start_idx += pattern_len
        try:
            end_idx = (input_tensor[start_idx:] == 151645).nonzero(as_tuple=True)[0][0] + start_idx
            result[start_idx:end_idx + 1] = input_tensor[start_idx:end_idx + 1]
        except IndexError:
            continue

    return result
       

def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("1现在载入model")
    model = QwertyQwen2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
    model.config.use_cache = False # 训练不需要在前向传播时缓存计算的中间激活值（hidden states）
    print("2现在载入tokenizer")
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_name_or_path)
    print("3现在载入image_processor")
    image_processor = CLIPImageProcessor.from_pretrained(model_args.vision_tower_name_or_path)
    print("3.5现在载入预训练的权重")
    """
    以下的这段代码，加载第一阶段预训练好的模型，到指定的地方。
    """
    from safetensors import safe_open
    folder_path = model_args.model_name_or_path
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
            #model.state_dict()[key] = merged_weights['model.' + key] 据说，直接修改 state_dict() 中的值通常是不被推荐的操作，因为 state_dict() 是一个浅拷贝，而不是模型参数的直接映射。这种操作可能会导致模型参数与优化器不匹配。
    print("3.75预训练的权重加载完毕")
    #print("4现在开始修改tokenizer")
    """
    # 如果加载的是"Qwen/Qwen2.5-7B-Instruct"的分词器，那么需要把<image>加入到特殊token中
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

    # 以上添加特殊token以及修改特殊embedding的值的步骤，仅在加载 "Qwen/Qwen2.5-7B-Instruct" 的分词器和模型权重，进行预训练的时候需要。
    # 之后的微调与推理，就不需要了
    """
    print("5现在开始建立Dataset")
    train_dataset = LazySupervisedDataset(
        data_path = data_args.data_path,
        image_folder = data_args.image_folder,
        tokenizer = tokenizer,
        image_processor = image_processor
    )
    print("6现在开始建立data_collator")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # 第二阶段的微调，只冻结vision tower
    for name, param in model.named_parameters():
        if 'vision_model' in name:
            #print(f"参数名: {name}, 可训练: {param.requires_grad}")
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
    print("6终于可以开始train了")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params}")
    #for name, param in model.named_parameters():
    #    print(name)
    #    print(f"          {param.dtype}")
    trainer.train()
    print("7train完了，保存权重与分词器")
    # 保存训练状态与权重，保存修改过的分词器
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)  # 它会保存分词器，不需要显式的调用 tokenizer.save_pretrained(save_directory=training_args.output_dir)
    print("8顺利退出")

if __name__ == "__main__":
    train()