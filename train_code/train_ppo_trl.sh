# accelerate launch --config_file /home/wxt/huatong/TRL_FT/config_file/deepspeed_zero3.yaml \
# 	/home/wxt/huatong/TRL_FT/train_code/original_ppov2.py \
#     --output_dir /home/wxt/huatong/huggingface/hub/llama2_ppo_online \
#     --num_ppo_epochs 1 \
#     --num_mini_batches 1 \
#     --learning_rate 3e-6 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --total_episodes 10000 \
#     --response_length 100 \
#     --model_name_or_path daryl149/llama-2-7b-hf \
#     --sft_model_path daryl149/llama-2-7b-hf \
#     --reward_model_path "EleutherAI/pythia-1b-deduped"  \
#     --local_rollout_forward_batch_size 1 \
#     --stop_token eos \
#     --stop_token_id 2 \
#     --non_eos_penalty 2>&1 | tee /home/wxt/huatong/TRL_FT/train_code/llama2_ppo_logs_sft.txt 

# --world_size: int = 4 \
# accelerate launch --config_file=/home/wxt/huatong/TRL_FT/config_file/deepspeed_zero3.yaml  --main_process_port 8888\ 
#     --num_processes 4\
#     /home/wxt/huatong/TRL_FT/train_code/ppov2_train.py \    
#     --output_dir  /home/wxt/huatong/huggingface/hub/llama2_ppo_online \
#     --num_ppo_epochs 1 \
#     --num_mini_batches 1 \
#     --learning_rate 3e-6 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --total_episodes 10000 \
#     --model_name_or_path daryl149/llama-2-7b-hf \
#     --sft_model_path daryl149/llama-2-7b-hf \
#     --reward_model_path OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 \
#     --local_rollout_forward_batch_size 1 \
#     --deepspeed3 \
#     --non_eos_penalty  2>&1 | tee /home/wxt/huatong/TRL_FT/train_code/llama2_ppo_logs.txt 

#     # -reward_model_path OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 \
#     #EleutherAI/pythia-1b-deduped
# #cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr

# python /home/wxt/huatong/TRL_FT/train_code/ppo_train.py \
#     --log_with=wandb \
#     --reward_model="OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5" \
#     --query_dataset="trl-internal-testing/hh-rlhf-trl-style" \
#     --model_name="/home/wxt/huggingface/hub/llama2_sft_mirror/"  2>&1 | tee /home/wxt/huatong/TRL_FT/train_code/ppov1_logs_sft.txt 

accelerate launch --config_file=/home/wxt/huatong/TRL_FT/config_file/deepspeed_zero3.yaml  --main_process_port 8888\ 
    /home/wxt/huatong/TRL_FT/train_code/ppo_train.py \
    --output_dir  /home/wxt/huatong/huggingface/hub/llama2_ppo_online_v1 \
    --num_mini_batches 32 \
    --learning_rate 3e-6 \
    --num_ppo_epochs 3 \
    --log_with=wandb \
    --reward_model="OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5" \
    --query_dataset="trl-internal-testing/hh-rlhf-trl-style" \
    --deepspeed3 \
    --model_name="/home/wxt/huggingface/hub/llama2_sft_mirror/"  2>&1 | tee /home/wxt/huatong/TRL_FT/train_code/ppov1_logs_sft.txt 

