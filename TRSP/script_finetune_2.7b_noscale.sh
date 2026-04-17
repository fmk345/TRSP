export CUDA_VISIBLE_DEVICES=1
MODEL_SIZE=2.7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=16
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

CUDA_VISIBLE_DEVICES=1 accelerate launch \
    --main_process_port 8889 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file /home/fmk/self-rag-main/retrieval_lm/stage3_no_offloading_accelerate.conf \
    /home/fmk/retrieval_lm/finetune.py \
    --model_name_or_path /raid2/DATA/llm_model/opt-2.7b \
    --use_flash_attn \
    --dataset_name c4 \
    --num_layers 32 \
    --lambda_reg 0.01 \
    --scale False \
    --tokenizer_name /raid2/DATA/llm_model/opt-2.7b \
    --use_slow_tokenizer \
    --train_file /home/fmk/dataset/wikitext/wikitext-2-raw-v1 \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir output/self_rag_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_special_tokens