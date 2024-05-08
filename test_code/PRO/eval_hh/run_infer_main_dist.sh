export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=16

id=$1
ranking_len=$2
accelerate launch --config_file dp_config.yaml infer_and_eval_main_generate.py \
    --index $id \
    --stage $ranking_len \
    --directory  /home/wxt/huatong/huggingface/hub/llama-7b-hh-sft > logs/generate_infer_main_${id}_${ranking_len}.log 2>&1 \
    

accelerate launch --config_file dp_config.yaml infer_and_eval_main_reward.py \
    --index $id \
    --stage $ranking_len \
    --directory  /home/wxt/huatong/huggingface/hub/llama-7b-hh-sft > logs/reward_infer_main_${id}_${ranking_len}.log 2>&1

python -u infer_and_eval_main_score.py \
    --index $id \
    --stage $ranking_len \
    --directory  /home/wxt/huatong/huggingface/hub/llama-7b-hh-sft > logs/score_infer_main_${id}_${ranking_len}.log 2>&1