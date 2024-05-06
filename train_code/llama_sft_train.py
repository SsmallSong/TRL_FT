# # imports
# from datasets import load_dataset
# from trl import SFTTrainer

# # get dataset
# dataset = load_dataset("/data2/huatong/dataset/imdb", split="train")

# # get trainer
# trainer = SFTTrainer(
#     "/data2/huatong/model/gpt2",
#     train_dataset=dataset,
#     dataset_text_field="text",
#     max_seq_length=512,
# )

# # train
# trainer.train()

# huggingface-cli download --resume-download teknium/OpenHermes-2.5-Mistral-7B --local-dir /data2/huatong/model/teknium/OpenHermes-2.5-Mistral-7B
# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""
import logging
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SftScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset,Dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, TrainingArguments

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((SftScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # print("==========================")
    # raw_datasets = load_dataset('stanfordnlp/imdb')
    # train_dataset = raw_datasets["train"]
    # eval_dataset = raw_datasets["test"]
    # print(raw_datasets)
    # print(train_dataset)
    # print("==========================")

    print("==========================")
    raw_datasets = load_dataset('Anthropic/hh-rlhf')
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]
    hh_train=train_dataset['chosen']
    text_list=[]
    label_list=[]
    # hh_train=hh_train[0:3]
    for item in hh_train:
        split_index = item.rfind('Assistant')
        if split_index != -1:
            text_list.append(item[:split_index])
            label_list.append(item[split_index:])
        else:
            # Handle cases where 'Assistant' substring is not found
            text_list.append(item)
            label_list.append("")  # Appending an empty string to label_list

    train_dataset={}
    train_dataset['text']=text_list
    train_dataset['label']=label_list
    train_dataset = Dataset.from_dict(train_dataset)
    hh_test=eval_dataset['chosen']
    text_list=[]
    label_list=[]
    # hh_test=hh_test[0:3]
    for item in hh_test:
        split_index = item.rfind('Assistant')

        if split_index != -1:
            text_list.append(item[:split_index])
            label_list.append(item[split_index:])
        else:
            # Handle cases where 'Assistant' substring is not found
            text_list.append(item)
            label_list.append("")  # Appending an empty string to label_list

    test_dataset={}
    test_dataset['text']=text_list
    test_dataset['label']=label_list
    eval_dataset = Dataset.from_dict(test_dataset)
    print(train_dataset)
    print("==========================")
    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            model=model_config.model_name_or_path,
            model_init_kwargs=model_kwargs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field=args.dataset_text_field,
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
            packing=args.packing,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )
    # assert 0==1
    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)