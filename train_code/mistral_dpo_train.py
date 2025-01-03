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
print("I change little herr,can you find it?")
# regular:
python examples/scripts/dpo.py \
    --dataset_name =snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset \
    --model_name_or_path =mistralai/Mistral-7B-Instruct-v0.2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="mistral_7b_instruct_dpo" \
    --warmup_steps 150 \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

# peft:
python examples/scripts/dpo.py \
    --dataset_name=s"norkelai/Snorkel-Mistral-PairRM-DPO-Dataset" \
    --model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2" \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="mistral_7b_instruct_dpo_peft" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""
import logging
import multiprocessing
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import DpoScriptArguments, init_zero_verbose, TrlParser
from datasets import Dataset, Features

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from trl import (
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
#from trl import DDDD


if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((DpoScriptArguments, TrainingArguments, ModelConfig))
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
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )
    print('max_length:',args.max_length)
    
    ################
    # Dataset
    ################

    #snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset 
    ds = load_dataset(args.dataset_name)
    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(50))
    # print(ds)
    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    ds = ds.map(
       process,
       num_proc=multiprocessing.cpu_count(),
       load_from_cache_file=False,
   )
    train_dataset = ds["train_iteration_3"][11:12]
    eval_dataset = ds["test_iteration_3"][0:10]
    eval_dataset = Dataset.from_dict(eval_dataset)
    train_dataset = Dataset.from_dict(train_dataset)
    #hh-rlhf
#     ds = load_dataset(args.dataset_name)
#     if args.sanity_check:
#         for key in ds:
#             ds[key] = ds[key].select(range(50))
#     # print(ds)
#     def process(row):
#         row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
#         row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
#         return row

#     train_dataset = ds["train"][5:6]
#     eval_dataset = ds["test"][0:5]

#     # train_chosen=train_dataset_origin['chosen']
#     # train_rejected=train_dataset_origin['rejected']
#     # eval_chosen=eval_dataset_origin['chosen']
#     # eval_rejected=eval_dataset_origin['rejected']
#     # train_prompt=train_dataset_origin['prompt']
#     # eval_prompt=eval_dataset_origin['prompt']

#     # pop_list_train=[]
#     # pop_list_eval=[]
#     # train_chosen_list=[]
#     # train_rejected_list=[]
#     # eval_chosen_list=[]
#     # eval_rejected_list=[]
#     # for i in range(len(train_chosen)):
#     #     conversation_list=train_chosen[i]   
#     #     roles_str = ' '.join([item['role'] for item in conversation_list])
#     #     train_chosen_list.append(roles_str)

#     # for i in range(len(train_rejected)):
#     #     conversation_list=train_rejected[i]   
#     #     roles_str = ' '.join([item['role'] for item in conversation_list])
#     #     train_rejected_list.append(roles_str)

#     # for i in range(len(eval_chosen)):
#     #     conversation_list=eval_chosen[i]   
#     #     roles_str = ' '.join([item['role'] for item in conversation_list])
#     #     eval_chosen_list.append(roles_str)

#     # for i in range(len(eval_rejected)):
#     #     conversation_list=eval_rejected[i]   
#     #     roles_str = ' '.join([item['role'] for item in conversation_list])
#     #     eval_rejected_list.append(roles_str)

#     # for i in range(len(train_chosen)):
#     #     if 'user user' in train_chosen_list[i] or 'assistant assistant' in train_chosen_list[i] or 'user user' in train_rejected_list[i] or 'assistant assistant' in train_rejected_list[i]:
#     #         pop_list_train.append(i)

#     # for i in range(len(eval_chosen)):
#     #     if 'user user' in eval_chosen_list[i] or 'assistant assistant' in eval_chosen_list[i] or 'user user' in eval_rejected_list[i] or 'assistant assistant' in eval_rejected_list[i]:
#     #         pop_list_eval.append(i)
#     # train_chosen = [value for index, value in enumerate(train_chosen) if index not in pop_list_train]
#     # train_rejected = [value for index, value in enumerate(train_rejected) if index not in pop_list_train]
#     # eval_chosen = [value for index, value in enumerate(eval_chosen) if index not in pop_list_eval]
#     # eval_rejected = [value for index, value in enumerate(eval_rejected) if index not in pop_list_eval]
#     # train_prompt = [value for index, value in enumerate(train_prompt) if index not in pop_list_train]
#     # eval_prompt = [value for index, value in enumerate(eval_prompt) if index not in pop_list_eval]

#     # train_dataset={}
#     # eval_dataset={}
#     # train_dataset['chosen']=train_chosen
#     # train_dataset['rejected']=train_rejected
#     # eval_dataset['chosen']=eval_chosen
#     # eval_dataset['rejected']=eval_rejected
#     # train_dataset['prompt']=train_prompt
#     # eval_dataset['prompt']=eval_prompt

#     train_dataset = Dataset.from_dict(train_dataset)
#     eval_dataset = Dataset.from_dict(eval_dataset)

#     train_dataset = train_dataset.map(
#        process,
#        num_proc=multiprocessing.cpu_count(),
#        load_from_cache_file=False,
#    )
#     eval_dataset = eval_dataset.map(
#        process,
#        num_proc=multiprocessing.cpu_count(),
#        load_from_cache_file=False,)
    ################
    # Training
    ################
    with init_context:
        trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            beta=args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_target_length=args.max_target_length,
            max_prompt_length=args.max_prompt_length,
            generate_during_eval=args.generate_during_eval,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
            trainer.save_model(training_args.output_dir)
