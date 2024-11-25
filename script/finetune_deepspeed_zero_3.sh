export CUDA_VISIBLE_DEVICES=0,1,2
echo $CUDA_VISIBLE_DEVICES


torchrun --nproc_per_node=3 --master_port=29500 finetune.py \
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
    --tf32 True \
    --deepspeed "./script/default_offload_opt_param.json"
    #>> log_241122.txt 2>&1