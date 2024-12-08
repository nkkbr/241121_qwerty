export CUDA_VISIBLE_DEVICES=0,1,2
echo $CUDA_VISIBLE_DEVICES
export TRITON_CACHE_DIR=/tmp/triton_autotune

torchrun --nproc_per_node=3 --master_port=29500 finetune.py \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 5 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed "./script/default_offload_opt_param.json"
    #>> log_241122.txt 2>&1

# 确保有这个缓存的路径，先运行：
# 
# mkdir -p /tmp/triton_autotune
# chmod 700 /tmp/triton_autotune