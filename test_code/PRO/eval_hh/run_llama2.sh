export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=16

index="llama2_7b_sft_halos_2_3"
log_file="/home/wxt/huatong/TRL_FT/test_code/PRO/eval_hh/logs/generate_infer_main_${index}.log"

# 执行命令并重定向输出到动态生成的文件名

accelerate launch --config_file dp_config.yaml infer_and_eval_main_generate.py \
    --index $index  \
    --stage 1 > $log_file 2>&1 
    

accelerate launch --config_file dp_config.yaml infer_and_eval_main_reward.py \
    --index $index \
    --stage 1 > $log_file 2>&1

python -u infer_and_eval_main_score.py \
    --index $index \
    --stage 1 > $log_file 2>&1



#accelerate launch --config_file dp_config.yaml infer_and_eval_main_generate.py \
 #   --index dpo_beta03  \
  #  --stage 2 > /home/wxt/huatong/TRL_FT/test_code/PRO/eval_hh/logs/generate_infer_main_dpo_beta03.log 2>&1 
    
#accelerate launch --config_file dp_config.yaml infer_and_eval_main_reward.py \
#    --index dpo_beta03 \
#    --stage 2 > /home/wxt/huatong/TRL_FT/test_code/PRO/eval_hh/logs/reward_infer_main_dpo_beta03.log 2>&1

#python -u infer_and_eval_main_score.py \
#    --index dpo_beta03 \
#    --stage 2 > /home/wxt/huatong/TRL_FT/test_code/PRO/eval_hh/logs/score_infer_main_dpo_beta03.log 2>&1
