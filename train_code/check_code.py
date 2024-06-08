import shutil

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig
from trl.trainer.ppov2_trainer import PPOv2Config, PPOv2Trainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE


"""
python -i examples/scripts/ppo/ppo.py \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path daryl149/llama-2-7b-hf \
    --non_eos_penalty \

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/ppo/ppo.py \
    --output_dir models/minimal/ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path daryl149/llama-2-7b-hf \
    --sft_model_path daryl149/llama-2-7b-hf \
    --reward_model_path OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 \
    --local_rollout_forward_batch_size 1 \
    --deepspeed3 \
    --non_eos_penalty \
"""


if __name__ == "__main__":
    # parser = HfArgumentParser((PPOv2Config, ModelConfig))
    # config, model_config = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    # shutil.rmtree(config.output_dir, ignore_errors=True)

    raw_datasets = load_dataset("trl-internal-testing/hh-rlhf-trl-style",split='train')
    print(raw_datasets)
    # eval_samples = 20
    # train_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples))
    # eval_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples, len(raw_datasets)))
    dataset_text_field = "prompt"

    
