from dataclasses import dataclass,field
import transformers
from transformers import Qwen2MoeForCausalLM, Qwen2Tokenizer
from torch.utils.data import Dataset
import json
import conversation 
import torch

@dataclass
class ModelArguments:
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class DataArguments:
    image_folder: str
    data_path: str = "/data/uchiha_ssd2/fengqi/llava_dataset/CC3M_Pretrain_595K/chat.json" 


@dataclass
class TrainingArguments(transformers.TrainingArguments):  
    pass


class SupervisedDataset(Dataset):

    def __init__(
            self,
            data_path: str,
            tokenizer: transformers.PreTrainedTokenizer,
            image_processor: transformers.CLIPImageProcessor
            ):
        self.input_ids = []
        self.labels = []
        self.images = []

        with open(data_path,'r') as f:
            data_list = json.load(f)
        for data in data_list:
            cur_conv = conversation.conv_qwen2_5.copy()
            




def get_labels(input_tensor):
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
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = Qwen2MoeForCausalLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_name_or_path)

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


if __name__ == "__main__":
    train()