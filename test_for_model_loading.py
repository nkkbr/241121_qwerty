from qwerty_qwen2 import QwertyQwen2ForCausalLM, CLIPVisionModel

model = QwertyQwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

param1 = model.state_dict()['model.vision_model.vision_model.encoder.layers.9.self_attn.q_proj.weight'][0]

vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")

param2 = vision_tower.state_dict()['vision_model.encoder.layers.9.self_attn.q_proj.weight'][0]

print("param1:")
print(param1)
print()
print('parma2:')
print(param2)
print()
print("param1 == param2 ?")
print(all(param1==param2))