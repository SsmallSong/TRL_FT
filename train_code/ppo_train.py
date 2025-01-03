# !export CUDA_VISIBLE_DEVICES="0,1,2,3'
"""
python examples/scripts/ppo.py \
    --log_with=wandb
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3"
print("okokokoko")
from dataclasses import dataclass, field
from typing import Optional
import model_training.models.reward_model  # noqa: F401 (registers reward model for AutoModel loading)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available

tqdm.pandas()
print("okokok")


@dataclass
class ScriptArguments:
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})

    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})


parser = HfArgumentParser((ScriptArguments, PPOConfig))
args, ppo_config = parser.parse_args_into_dataclasses()

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead
model_name_or_path ='/home/wxt/huggingface/hub/llama2_sft_mirror/'

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, query_dataset):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(query_dataset, split="train")
    # ds = ds.select(range(10))
    # print(ds[:3])

    def tokenize(sample):
        element_temp="\n<|user|>\n"+sample['prompt']+"\n<|assistant|>\n" 
        sample["input_ids"] = tokenizer.encode(element_temp, padding=True, truncation=True,max_length=128)
        sample["query"] = element_temp
        return sample
    
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


print("00000000000")

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(ppo_config, "trl-internal-testing/hh-rlhf-trl-style")
print(dataset[:3])
# print(dataset)
print('111111111111')

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# set seed before initializing value head for deterministic eval
set_seed(ppo_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(model_name_or_path, trust_remote_code=args.trust_remote_code)
    device_map = None
    peft_config = None
else:
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

model = trl_model_class.from_pretrained(
    model_name_or_path,
    trust_remote_code=args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.eos_token_id

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
print("22222222222")
# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    if is_xpu_available():
        device = "xpu:0"
    elif is_npu_available():
        device = "npu:0"
    else:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
        
reward_model_name="OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5"
rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
rm= AutoModelForSequenceClassification.from_pretrained(reward_model_name)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}
print("333333333333")
print(ppo_trainer.dataloader)
total_batches = len(ppo_trainer.dataloader)
print("total_batches: ",total_batches)
save_path="/home/wxt/huatong/huggingface/hub/llama2_ppov1_online/"
for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    print("The epoch is: ",_epoch)
    # print(batch)
    query_tensors = batch["input_ids"]

    # Get response from gpt2
    print("444444444444")
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
    )
    print("55555555555555")
    batch["response"] = tokenizer.batch_decode(response_tensors)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)
    print("6666666666666")
    # Compute reward score
    
    texts = [q.replace("\n<|user|>\n","<|prompter|>").replace("\n<|assistant|>\n","<|endoftext|><|assistant|>") + r + "<|endoftext|>" for q, r in zip(batch["query"], batch["response"])]
    # print("====================text==========================")
    # for i in range(10):
    #     print(texts[i])
    # print("====================text==========================")
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,max_length=256)
    # print("The inputs shape: ",inputs.shape)
    rewards_tensor = rm(**inputs).logits#.cpu().detach()
    rewards = [row for row in rewards_tensor]
    # print("====================reward==========================")
    # print(rewards)
    # print("====================reward==========================")
    
    ref_texts = [q.replace("\n<|user|>\n","<|prompter|>").replace("\n<|assistant|>\n","<|endoftext|><|assistant|>") + r + "<|endoftext|>" for q, r in zip(batch["query"], batch["ref_response"])]
    ref_inputs = tokenizer(ref_texts, return_tensors="pt", padding=True, truncation=True,max_length=256)
    ref_rewards_tensor = rm(**ref_inputs).logits#.cpu().detach()
    ref_rewards = [row for row in ref_rewards_tensor]

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])
    print('777777777777777')
    # model.to("cpu")
    torch.save(ppo_trainer.model.state_dict(), save_path)

torch.save(ppo_trainer.model.state_dict(), save_path)
