export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=16

index="llama2_dpo_handbook_2048"
log_file="/home/wxt/huatong/TRL_FT/test_code/PRO/eval_hh/logs_new/generate_infer_main_$index.log"
model_ckpt="non"

accelerate launch --config_file dp_config.yaml infer_and_eval_main_generate.py \
    --index $index \
    --model_ckpt $model_ckpt \
    --stage 1 
     
    
accelerate launch --config_file dp_config.yaml infer_and_eval_main_reward.py \
    --index $index \
    --stage 1 

python -u infer_and_eval_main_score.py \
    --index $index \
    --stage 1 > $log_file 2>&1 
    
# index="7b_llama_dpo_beta001_lr2e6"
# log_file="/home/wxt/huatong/TRL_FT/test_code/PRO/eval_hh/logs_new/generate_infer_main_$index.log"
# model_ckpt="llama2_7b_dpo_halos_beta001_lr2e6"

# accelerate launch --config_file dp_config.yaml infer_and_eval_main_generate.py \
#     --index $index \
#     --model_ckpt $model_ckpt \
#     --stage 1 
     
    
# accelerate launch --config_file dp_config.yaml infer_and_eval_main_reward.py \
#     --index $index \
#     --stage 1 

# python -u infer_and_eval_main_score.py \
#     --index $index \
#     --stage 1 > $log_file 2>&1 


# index="llama2_7b_dpo_halos_beta01_frontenter"
# log_file="/home/wxt/huatong/TRL_FT/test_code/PRO/eval_hh/logs_new/generate_infer_main_llama2_{index}.log"
# model_ckpt="llama2_7b_dpo_halos_beta01"

# accelerate launch --config_file dp_config.yaml infer_and_eval_main_generate.py \
#     --index $index  \
#     --model_ckpt $model_ckpt
#     --stage 1 > $log_file 2>&1 

# # accelerate launch --config_file dp_config.yaml infer_and_eval_main_reward.py \
# #     --index $index \
# #     --stage 1 > $log_file 2>&1

# # python -u infer_and_eval_main_score.py \
# #     --index $index \
# #     --stage 1 > $log_file 2>&1

