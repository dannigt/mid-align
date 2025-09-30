#!/bin/bash
export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=#eno2np1 #eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=29501}"

module load compiler/gnu/12
module load devel/cuda/12.0

HOST_NUM=1
INDEX=0

model_path="meta-llama/Meta-Llama-3-8B-Instruct"
train_files="./data_example/train_baseline.json" # replace by actual training data
valid_files="./data_example/train_baseline.json" # replace by actual validation data
train_bsz=32
eval_bsz=32
gradient_accumulation_steps=4
lora_config="./config/lora_config.json"
LR="5e-4"
OUTDIR="./test_run_outputs"
nproc_per_node=1 # number of GPUs used in training
loss_layer=16
loss_temperature=0.1
loss_distance_type="cosine"


torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node $nproc_per_node \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ./scripts/run_clm_lora.py \
    --deepspeed ./config/deepspeed_config.json \
    --bf16 True \
    --bf16_full_eval True \
    --model_name_or_path ${model_path} \
    --train_file $train_files \
    --validation_file $valid_files \
    --use_lora True \
    --lora_config ./config/lora_config.json \
    --torch_dtype bfloat16 \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 1 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size $train_bsz \
    --per_device_eval_batch_size $eval_bsz \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs 10 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "inverse_sqrt" \
    --logging_steps 10 \
    --block_size 2048 \
    --do_train \
    --eval_strategy "steps" \
    --eval_steps 200 \
    --streaming \
    --ddp_timeout 3600 \
    --seed 1 \
    --gradient_checkpointing True \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --patience 5 \
    --output_dir $OUTDIR \
    --contrastive_data_mode 2 \
    --additional_loss_layer $loss_layer \
    --contrastive_loss_temperature $loss_temperature \
    --distance_function $loss_distance_type \
    --alternate_training \
    --disable_tqdm True --overwrite_output_dir 2>&1  | tee -a $OUTDIR/train.log
