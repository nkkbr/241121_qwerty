export NCCL_SOCKET_IFNAME=eno1np0
export NCCL_SOCKET_FAMILY=IPv4
export GLOO_SOCKET_FAMILY=IPv4
export CUDA_VISIBLE_DEVICES=2,3
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=192.168.100.31 --master_port=29500 finetune.py \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 10 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
    --tf32 True