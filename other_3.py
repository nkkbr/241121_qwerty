from transformers import Qwen2Tokenizer

tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

special_tokens = {
    "additional_special_tokens": ["<image>"]
}
tokenizer.add_special_tokens(special_tokens)

tokenizer.save_pretrained(save_directory="./test/")