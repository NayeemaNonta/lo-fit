#!/bin/bash
## Uncomment the following to run experiments for: llama2-llama2_13B-base
# model_name="llama2-llama2_13B-base";
# task="clutrr";
# seed=42;
# CUDA_VISIBLE_DEVICES=0 python3 lofit_trainer.py \
#     --task $task \
#     --base_model_name $model_name \
#     --apply_chat_template False \
#     --ft_method lofit \
#     --lofit_component A \
#     --use_topk_heads 160 \
#     --lr 1e-3 \
#     --train_batch 8 \
#     --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}"\
#     --run_mode train \
#     --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}"\
#     --applied_module attention \
#     --save_strategy no \
#     --l1_lambda 1e-3 \
#     --eval_batch 32 \
#     --train_size 300 \
#     --seed $seed;
# CUDA_VISIBLE_DEVICES=0 python3 lofit_trainer.py \
#     --task $task  \
#     --base_model_name $model_name \
#     --apply_chat_template False \
#     --ft_method lofit \
#     --lofit_component v \
#     --use_topk_heads 48 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy"\
#     --lr 1e-2 \
#     --train_batch 8 \
#     --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_lofit_seed${seed}"\
#     --run_mode train \
#     --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_lofit_seed${seed}"\
#     --applied_module attention \
#     --save_strategy no \
#     --l1_lambda 0 \
#     --eval_batch 32 \
#     --train_size 300 \
#     --seed $seed;
## Uncomment the following to run experiments for: llama2_7B
set -euo pipefail
model_name="llma3.2_1B";
task="clutrr";
seed=42;
CUDA_VISIBLE_DEVICES=0 python3 lofit_trainer.py \
    --task $task \
    --base_model_name $model_name \
    --apply_chat_template False \
    --ft_method lofit \
    --lofit_component A \
    --use_topk_heads 160 \
    --lr 5e-4 \
    --train_batch 8 \
    --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}"\
    --run_mode train_wandb \
    --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}"\
    --applied_module attention \
    --save_strategy no \
    --l1_lambda 5e-3 \
    --eval_batch 32 \
    --train_size 300 \
    --seed $seed;
CUDA_VISIBLE_DEVICES=0 python3 lofit_trainer.py \
    --task $task  \
    --base_model_name $model_name \
    --apply_chat_template False \
    --ft_method lofit \
    --lofit_component v \
    --use_topk_heads 32 \
    --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy"\
    --lr 1e-2 \
    --train_batch 8 \
    --num_epoch 5 \
    --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_lofit_seed${seed}"\
    --run_mode train_wandb \
    --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_lofit_seed${seed}"\
    --applied_module attention \
    --save_strategy no \
    --l1_lambda 0 \
    --eval_batch 32 \
    --train_size 300 \
    --seed $seed;
## Uncomment the following to run experiments for: gemma_7b
# model_name="gemma_7b";
# task="clutrr";
# seed=42;
# CUDA_VISIBLE_DEVICES=0 python3 lofit_trainer.py \
#     --task $task \
#     --base_model_name $model_name \
#     --apply_chat_template False \
#     --ft_method lofit \
#     --lofit_component A \
#     --use_topk_heads 160 \
#     --lr 5e-4 \
#     --train_batch 8 \
#     --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_Aonly_seed${seed}"\
#     --run_mode train \
#     --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_Aonly_seed${seed}"\
#     --applied_module attention \
#     --save_strategy no \
#     --l1_lambda 5e-3 \
#     --eval_batch 32 \
#     --train_size 300 \
#     --seed $seed;
# CUDA_VISIBLE_DEVICES=5 python3 lofit_trainer.py \
#     --task $task  \
#     --base_model_name $model_name \
#     --apply_chat_template False \
#     --ft_method lofit \
#     --lofit_component v \
#     --use_topk_heads 16 \
#     --lofit_heads "./top_heads/${model_name}_${task}_Aonly_top160heads_${seed}.npy"\
#     --lr 1e-2 \
#     --train_batch 8 \
#     --num_epoch 5 \
#     --output_dir "./finetuned_checkpoints/${task}/${model_name}_${task}_lofit_seed${seed}"\
#     --run_mode train \
#     --output_file_name "./finetuned_outputs/${task}/${model_name}_${task}_lofit_seed${seed}"\
#     --applied_module attention \
#     --save_strategy no \
#     --l1_lambda 0 \
#     --eval_batch 32 \
#     --train_size 300 \
#     --seed $seed;