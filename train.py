def train():

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

