# trl dpo --model_name_or_path /data2/huatong/model/gpt2 --dataset_name /data2/huatong/dataset/hh-rlhf-trl-style --output_dir gpt2-dpo 
# trl sft --model_name_or_path /data2/huatong/model/gpt2 --dataset_name /data2/huatong/dataset/imdb --output_dir gpt2-sft 



# # nohup bash /data2/huatong/trl.bash > /data2/huatong/trl.txt 2>&1 &

#load again try
#这个可以push吗，如果可以那就是服务器不能同时push咯
# python /data2/huatong/train/sft_train.py \
#     --model_name_or_path="gpt2" \
#     --learning_rate=1.41e-5 \
#     --per_device_train_batch_size=64 \
#     --gradient_accumulation_steps=16 \
#     --output_dir="/home/wxt/huatong/model/gpt2-sft" \
#     --logging_steps=1 \
#     --num_train_epochs=1 \
#     --max_steps=-1 \
#     --gradient_checkpointing \

# export CUDA_VISIBLE_DEVICES=5

# python /data2/huatong/train/dpo_train.py \
#     --dataset_name='/data2/huatong/dataset/hh-rlhf-trl-style' \
#     --model_name_or_path='/data2/huatong/model/gpt2' \
#     --per_device_train_batch_size 4 \
#     --learning_rate 1e-3 \
#     --gradient_accumulation_steps 1 \
#     --logging_steps 10 \
#     --eval_steps 500 \
#     --output_dir="/data2/huatong/model/dpo_gpt_hh" \
#     --warmup_steps 150 \
#     --logging_first_step \
#     --no_remove_unused_columns

# python /data2/huatong/train/ppo_train.py \
#     --dataset_name='/data2/huatong/dataset/hh-rlhf-trl-style' \
#     --model_name_or_path='/data2/huatong/model/gpt2' \
#     --per_device_train_batch_size 4 \
#     --learning_rate 1e-3 \
#     --gradient_accumulation_steps 1 \
#     --logging_steps 10 \
#     --eval_steps 500 \
#     --output_dir="/data2/huatong/model/dpo_gpt_hh" \
#     --warmup_steps 150 \
#     --logging_first_step \
#     --no_remove_unused_columns

# nohup bash trl.sh > /data2/huatong/output_dpo.txt 2>&1 &
# ps aux | grep huatong

# python /data2/huatong/train/ppo.py

# huggingface-cli download --token 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 --resume-download mistralai/Mistral-7B-Instruct-v0.2 --local-dir /data2/huatong/model/Mistral-7B-Instruct-v0.2

# 这个地方的cuda:0 实际上并不是0号GPU，他取决于CUDA_VISIBLE_DEVICES
# 然后逻辑GPU和物理GPU有一个对应关系
# 如果CUDA_VISIBLE_DEVICES为2,1,3
# 那么CUDA:0就是2号GPU， CUDA:1 就是1号GPU CUDA:3 就是3号GPU
# return torch.device('cuda:0' if cuda else 'cpu')
 
python /home/wxt/huatong/TRL_FT/dpo_train.py \
    --dataset_name = snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset \
    --model_name_or_path = mistralai/Mistral-7B-Instruct-v0.2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir='/home/wxt/hautong/huggingface/hub/mistral_7b_instruct_dpo' \
    --warmup_steps 150 \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

#nohup bash trl.sh > /home/wxt/huatong/TRL_FT/output_dpo.txt 2>&1 &