from qwerty_qwen2 import QwertyQwen2ForCausalLM
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = QwertyQwen2ForCausalLM.from_pretrained(model_name)

for name, param in model.named_parameters():
    if name.startswith("model.vision_model") or name.startswith("model.mm_projector"):
        continue
    param.requires_grad_(False)
        