export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=16

id=$1
ranking_len=$2
accelerate launch --config_file dp_config.yaml infer_and_eval_main_generate.py \
    --index no1  \
    --stage 2 > /home/wxt/huatong/TRL_FT/test_code/PRO/eval_hh/logs/generate_infer_main_2.log 2>&1 
    

accelerate launch --config_file dp_config.yaml infer_and_eval_main_reward.py \
    --index no1 \
    --stage 2 > /home/wxt/huatong/TRL_FT/test_code/PRO/eval_hh/logs/reward_infer_main_2.log 2>&1

python -u infer_and_eval_main_score.py \
    --index no1 \
    --stage 2 > /home/wxt/huatong/TRL_FT/test_code/PRO/eval_hh/logs/score_infer_main_2.log 2>&1


# accelerate launch --config_file dp_config.yaml infer_and_eval_main_generate.py \
# 	    --index dpo  \
# 	    --stage 2 \
# 	    --directory  /home/wxt/huatong/huggingface/hub/llama-7b-hh-sft > /home/wxt/huatong/TRL_FT/test_code/PRO/eval_hh/logs/generate_infer_main_dpo.log 2>&1 

# accelerate launch --config_file dp_config.yaml infer_and_eval_main_reward.py \
# 	    --index dpo \
# 	    --stage 2 \
# 		--directory  /home/wxt/huatong/huggingface/hub/llama-7b-hh-sft > /home/wxt/huatong/TRL_FT/test_code/PRO/eval_hh/logs/reward_infer_main_dpo.log 2>&1

# python -u infer_and_eval_main_score.py \
# 	    --index dpo \
# 	    --stage 2 \
# 		--directory  /home/wxt/huatong/huggingface/hub/llama-7b-hh-sft > /home/wxt/huatong/TRL_FT/test_code/PRO/eval_hh/logs/score_infer_main_dpo.log 2>&1
