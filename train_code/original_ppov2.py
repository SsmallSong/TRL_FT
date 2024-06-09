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

if __name__ == "__main__":
    model_name_or_path ='/home/wxt/huggingface/hub/llama2_sft_mirror/'

    parser = HfArgumentParser((PPOv2Config, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    import model_training.models.reward_model

    
    # # # remove output_dir if exists
    # #   shutil.rmtree(config.output_dir, ignore_errors=True)
    
    # ################
    # # Model & Tokenizer
    # ################
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    # model_config_2 = AutoConfig.from_pretrained(model_name_or_path)
    ref_policy = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    policy = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    
    print(config.reward_model_path)
    print("+"*30)
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
            # print(element[dataset_text_field])
            element_temp=["<|user|>\n"+ele+"\n<|assistant|>" for ele in element[dataset_text_field]]
            # print(element_temp)
            outputs = tokenizer(
                element_temp,
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
