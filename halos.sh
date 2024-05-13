
python train.py loss=sft model=llama_2_7b datasets=[hh] exp_name=llama2_7b_sft_halos_2_2 mode=train \
    ++cache_dir=/home/wxt/.cache/huggingface/hub ++trainer=SFTTrainer ++seed 2\
    ++model.load_from=/home/wxt/.cache/huggingface/hub/llama2_7b_sft_halos_2/LATEST/policy.pt\
    ++use_fsdp=true  2>&1 | tee /home/wxt/huatong/TRL_FT/halos_train_sft_log_new_2.txt

python train.py loss=sft model=llama_2_7b datasets=[hh] exp_name=llama2_7b_sft_halos_2_3 mode=train \
    ++cache_dir=/home/wxt/.cache/huggingface/hub ++trainer=SFTTrainer ++seed 3\
    ++model.load_from=/home/wxt/.cache/huggingface/hub/llama2_7b_sft_halos_2_2/LATEST/policy.pt\
    ++use_fsdp=true  2>&1 | tee /home/wxt/huatong/TRL_FT/halos_train_sft_log_new_3.txt