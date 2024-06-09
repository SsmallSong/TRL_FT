#import shutil
import model_training.models.reward_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification,AutoTokenizer,HfArgumentParser,AutoConfig

from trl import ModelConfig
from trl.trainer.ppov2_trainer import PPOv2Config, PPOv2Trainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
import torch
import os
import gc
from typing import Dict, Union, Type, List
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
def delete_dict(d: Dict):
    """Delete all items inside the dict."""
    for k in list(d.keys()):
        del d[k]
print("+"*20)
print("come on!")
print("+"*20)
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
    --model_name_or_path daryl149/llama-2-7b-hf\
    --sft_model_path daryl149/llama-2-7b-hf \
    --reward_model_path OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 \
    --local_rollout_forward_batch_size 1 \
    --deepspeed3 \
    --non_eos_penalty \
"""


if __name__ == "__main__":
    model_name_or_path ='/home/wxt/huggingface/hub/llama2_sft_mirror/'

    parser = HfArgumentParser((PPOv2Config, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    # # remove output_dir if exists
    #   shutil.rmtree(config.output_dir, ignore_errors=True)
    
    ################
    # Model & Tokenizer
    ################
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    model_config_2 = AutoConfig.from_pretrained(model_name_or_path)
    ref_policy = AutoModelForCausalLM.from_pretrained(model_name_or_path,config=model_config_2)
    policy = AutoModelForCausalLM.from_pretrained(model_name_or_path,config=model_config_2)
    

    value_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1)
    reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1)
    ################
    # Dataset
    ################
    # raw_datasets = load_dataset("trl-internal-testing/descriptiveness-sentiment-trl-style", split="descriptiveness")
    # print(raw_datasets)
    train_dataset = load_dataset("trl-internal-testing/hh-rlhf-trl-style", split="train")
    eval_dataset = load_dataset("trl-internal-testing/hh-rlhf-trl-style", split="test")
    dataset_text_field = "prompt"

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            batched=True,
            num_proc=4,  # multiprocessing.cpu_count(),
            load_from_cache_file=False,
        )

    ################
    # Training
    ################
    trainer = PPOv2Trainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=prepare_dataset(train_dataset, tokenizer),
        eval_dataset=prepare_dataset(eval_dataset, tokenizer),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    trainer.push_to_hub()
    trainer.generate_completions()
