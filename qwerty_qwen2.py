import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM, AutoConfig, AutoModelForCausalLM, CLIPImageProcessor,CLIPVisionModel
from typing import Optional, Union, Tuple, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336"


class QwertyConfig(Qwen2Config):
    model_type = 'qwerty_qwen2'


class QwertyQwen2Model(Qwen2Model):
    config_class = QwertyConfig

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.image_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.vision_model = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME)
        self.mm_projector = nn.Sequential(
            nn.Linear(in_features=self.vision_model.config.hidden_size, out_features=config.hidden_size *2),
            nn.GELU(),
            nn.Linear(in_features=config.hidden_size *2, out_features=config.hidden_size)
        )


class QwertyQwen2ForCausalLM(Qwen2ForCausalLM):
    config_class = QwertyConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = QwertyQwen2Model(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        """
        images 是已经被image_processor处理过的图像，shape是(batch_size,3, 336,336)
        """
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                labels,
                images,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                images,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        return inputs
    
    def prepare_inputs_labels_for_multimodal(self,
                                             input_ids,
                                             position_ids,
                                             attention_mask,
                                             labels,
                                             images):
        
        image_features = self.model.vision_model(images,output_hidden_states=True).hidden_states[-2]   # 选取了倒数第二层，形状是(batch_size, 577, 1024) 577是一个CLS + (336/14)**2。 也可以不选CLS，只使用(batch_size, 576, 1024)
        image_features = self.model.mm_projector(image_features)                                       # (batch_size, 577, 3584)
        
        batch_size = input_ids.shape[0]
        new_sequence_length = input_ids.shape[-1] + 577 - 1                                      # 如果你不选CLS，那么这里就是576

        image_token_id = 151665                                                                  # <image>的token_id

        new_position_ids = torch.arange(new_sequence_length)
        new_position_ids = new_position_ids.repeat(batch_size,1)
        new_attention_mask = torch.ones((batch_size,new_sequence_length))
        new_labels = torch.full((batch_size,new_sequence_length),-100)                           # 
        new_inputs_embeds = torch.zeros((batch_size,new_sequence_length,3584))                   # Qwen2.5 的嵌入维度是3584

        for i in range(batch_size):
            cur_input_ids = input_ids[i]
            image_location = torch.where(cur_input_ids==image_token_id)[0].item()

            new_inputs_embeds[i,:image_location] = self.model.embed_tokens(cur_input_ids[:image_location])
            new_inputs_embeds[i,image_location:image_location+577] = image_features[i]
            new_inputs_embeds[i,image_location+577:] = self.model.embed_tokens(cur_input_ids[image_location+1:])

            number_of_zeros = torch.sum(~attention_mask[i])
            if number_of_zeros > 0:
                new_attention_mask[i][-number_of_zeros:]=0
                new_position_ids[i][-number_of_zeros:]=0

        new_labels[:,577-1:]=labels                                                            # 反正前面是一堆-100，图像总是在前面，无论插在哪里，都是把原来非-100的labels往后面移动

        return None, new_position_ids, new_attention_mask.bool(), new_inputs_embeds,new_labels


AutoConfig.register('qwerty_qwen2', QwertyConfig)
AutoModelForCausalLM.register(QwertyConfig, QwertyQwen2ForCausalLM)