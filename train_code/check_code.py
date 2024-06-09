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

print("+"*20)
print("come on!")
print("+"*20)


if __name__ == "__main__":
    model_name_or_path ='daryl149/llama-2-7b-hf'

    train_dataset = load_dataset("trl-internal-testing/hh-rlhf-trl-style", split="train")
    eval_dataset = load_dataset("trl-internal-testing/hh-rlhf-trl-style", split="test")
    dataset_text_field = "prompt"
    prompt_origin=train_dataset[dataset_text_field]
    print(prompt_origin[0:10])
    
