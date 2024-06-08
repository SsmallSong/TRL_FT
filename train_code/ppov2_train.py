#import shutil

from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification,AutoTokenizer,HfArgumentParser

from trl import ModelConfig
from trl.trainer.ppov2_trainer import PPOv2Config, PPOv2Trainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE

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
    --model_name_or_path /home/wxt/huatong/huggingface/hub/models--daryl149--llama-2-7b-hf/snapshots/142d0a5354ab12acdfff745a4d5c2ced307970dd \
    --sft_model_path /home/wxt/huatong/huggingface/hub/models--daryl149--llama-2-7b-hf/snapshots/142d0a5354ab12acdfff745a4d5c2ced307970dd \
    --reward_model_path OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 \
    --local_rollout_forward_batch_size 1 \
    --deepspeed3 \
    --non_eos_penalty \
"""


if __name__ == "__main__":
    parser = HfArgumentParser((PPOv2Config, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
 #   shutil.rmtree(config.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    
    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)

    print('begin loading pre-trained weights')
    ckpt_path = f"/home/wxt/.cache/huggingface/hub/llama2_7b_sft_halos_2_3/LATEST/policy.pt"
    state_dict = torch.load(ckpt_path, map_location='cpu')
    ref_policy.load_state_dict(state_dict['state'])
    policy.load_state_dict(state_dict['state'])
    delete_dict(state_dict)
    gc.collect()
    torch.cuda.empty_cache()
    print('loaded pre-trained weights')

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
