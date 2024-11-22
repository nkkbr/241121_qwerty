from dataclasses import dataclass, field
from enum import auto, Enum
from typing import List, Tuple, Union
from PIL import Image

class TemplateStyle(Enum):
    """
    不同的语言模型有不同的template，这个枚举类用于指定是哪一种template
    """
    QWEN2_5 = auto()
    LLAMA3_1 = auto()


@dataclass
class Conversation:
    """
    用于保存所有的对话，并返回用于模型处理的文本串。
    在这个版本中，我们假定，只有USER的文本中有图像，ASSISTANT的回答是纯文本。但并不限于第一句话可以有图片。
    我们假定，一句话中，只能有一张图片（今后可以扩展）
    所有的图片，都会被移到文本的前面，在训练和推理时都这样。
    
    message: 对话的角色，对话的内容及图片
            messages=[
                ["USER", ("<image> A picture of a cat.", image_data(是一个Image.Image对象))]
                ["ASSISTANT", "Looks like a cat."]
            ],
        也要支持没有图片的情况：
            messages=[
                ["USER", "<image> A picture of a cat.")]
                ["ASSISTANT", "Looks like a cat."]
            ],
        有图片时，“USER”之后的数据类型是一个tuple，否则，是一个str。

            "USER" 和 "ASSISTANT" 必须交替出现。USER必须在第一个出现。
    <image> 来指代图片是被硬编码的，制作的数据中会使用这个特殊的字符串。
    """
    messages: List[List[Union[str, Tuple[str, Image.Image]]]] = field(default_factory=list)
    offset:int = 0
    template_style: TemplateStyle = TemplateStyle.QWEN2_5

    def get_prompt(self):
        """
        根据template_style和messages返回文本串。
        """
        # 下面这一段代码，是将<image>移到文本的前面。我们的训练数据，<image>在文本和开头和末尾的情况都是有的。这一版的训练，我们假设图像随机地可能在开头或者末尾。
        # messages = self.messages.copy()
        # for message in messages:
        #     if message[0] != "USER" or type(message[1]) is not tuple:
        #         continue
        #     msg = message[1][0].replace("<image>","").strip()
        #     message[1][0] = "<image>\n" + msg
        
        if self.template_style == TemplateStyle.QWEN2_5:
            ret = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            for idx, message in enumerate(self.messages):
                if idx % 2 == 0:
                    if type(message[1]) is tuple:
                        msg = message[1][0]
                    else:
                        msg = message[1]
                    msg = "<|im_start|>user\n" + msg + "<|im_end|>\n"
                else:
                    msg = message[1]
                    msg = "<|im_start|>assistant\n" + msg + "<|im_end|>\n"
                ret += msg
            if ret.endswith("\n"):
                ret = ret[:-1]  # 去掉最后一个'\n'
        elif self.template_style == TemplateStyle.LLAMA3_1:
            from datetime import datetime
            current_date = datetime.now()
            formatted_date = current_date.strftime("%d %b %Y")

            ret = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: {formatted_date}\n\nYou are a helpful assistant.<|eot_id|>"
            for idx, message in enumerate(self.messages):
                if idx % 2 == 0:
                    if type(message[1]) is tuple:
                        msg = message[1][0]
                    else:
                        msg = message[1]
                    msg = "<|start_header_id|>user<|end_header_id|>\n\n" + msg + "<|eot_id|>"
                else:
                    msg = message[1]
                    msg = "<|start_header_id|>assistant<|end_header_id|>\n\n" + msg + "<|eot_id|>"
                ret += msg
        else:
            raise ValueError(f"Invalid style: {self.template_style}")
        
        return ret
    
    def append_message(self, 
                       role_message_image_list: List
        ):
        self.messages.append(role_message_image_list)

    def copy(self):
        return Conversation(
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            template_style = self.template_style
            )

conv_qwen2_5 = Conversation(template_style=TemplateStyle.QWEN2_5)
conv_llama3_1 = Conversation(template_style=TemplateStyle.LLAMA3_1)

if __name__ == "__main__":

    # 简单地测试
    print("conv_qwen2_5 初始化后的内容：")
    print(conv_qwen2_5.get_prompt())
    conv_qwen2_5.append_message(['USER',("<image>/n Hello.",Image.Image())])
    conv_qwen2_5.append_message(['ASSISTANT',"AAAAA."])
    print("添加一段对话后：")
    print(conv_qwen2_5.get_prompt())
    conv_qwen2_5.append_message(['USER',"BBBBB."])
    conv_qwen2_5.append_message(['ASSISTANT',"Bye."])
    print("添加另一段对话后：")
    print(conv_qwen2_5.get_prompt())
    print()

    print("conv_llama3_1 初始化后的内容：")
    print(conv_llama3_1.get_prompt())
    conv_llama3_1.append_message(['USER',("<image>/n Hello.",Image.Image())])
    conv_llama3_1.append_message(['ASSISTANT',"AAAAA."])
    print("添加一段对话后：")
    print(conv_llama3_1.get_prompt())
    conv_llama3_1.append_message(['USER',"BBBBB."])
    conv_llama3_1.append_message(['ASSISTANT',"Bye."])
    print("添加另一段对话后：")
    print(conv_llama3_1.get_prompt())
