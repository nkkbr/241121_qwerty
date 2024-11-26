export CUDA_VISIBLE_DEVICES=0,1,2,3
echo $CUDA_VISIBLE_DEVICES
export TRITON_CACHE_DIR=/tmp/triton_autotune
#export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL

torchrun --nproc_per_node=4 --master_port=29500 finetune.py \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --tf32 True \
    --gradient_checkpointing True \
    --deepspeed "./script/default_offload_opt_param.json"
    #>> log_241122.txt 2>&1

# 确保有这个缓存的路径，先运行：
# 
# mkdir -p /tmp/triton_autotune
# chmod 700 /tmp/triton_autotune


# deepspeed "./script/default_offload_opt_param.json" 里关于scheduler的设置，似乎与这里的consine冲突了，可能应该删去