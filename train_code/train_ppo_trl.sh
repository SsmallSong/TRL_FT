accelerate launch --config_file /home/wxt/huatong/TRL_FT/config_file/deepspeed_zero3.yaml \
	/home/wxt/huatong/TRL_FT/train_code/ppov2_train.py \
    --output_dir /home/wxt/huatong/huggingface/hub/llama2_ppo_online \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path daryl149/llama-2-7b-hf \
    --sft_model_path daryl149/llama-2-7b-hf \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --non_eos_penalty 2>&1 | tee /home/wxt/huatong/TRL_FT/train_code/llama2_ppo_logs.txt 


    # -reward_model_path OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 \
