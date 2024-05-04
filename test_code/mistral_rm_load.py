# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# import llm_blender
# blender = llm_blender.Blender()
# blender.loadranker("llm-blender/PairRM") # load PairRM

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback")
# model = AutoModelForSequenceClassification.from_pretrained("Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback")
# print("111111111")
# # Load model directly

from transformers import AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets
import json
import os
import numpy as np
import random
import torch
import transformers
from vllm import LLM, SamplingParams
import pickle as pkl
import model_training.models.reward_model  # noqa: F401 (registers reward model for AutoModel loading)

# tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5")
# model = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5")

# print("2222222")

# # Load model directly

# from transformers import AutoModel
# model = AutoModel.from_pretrained("OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5")

# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("openbmb/Eurus-RM-7b", trust_remote_code=True)
# model = AutoModel.from_pretrained("openbmb/Eurus-RM-7b", trust_remote_code=True)
# print('333333333')
# from datasets import load_dataset

# dataset = load_dataset("tatsu-lab/alpaca_eval")
# print("4444444")


#model_id = '/home/wxt/huatong/huggingface/hub/mistral_7b_instruct_dpo'

#mistral_temp = False 
#res = []
#tokenizer = transformers.AutoTokenizer.from_pretrained(model_id,legacy=False)
#print('model ok')
#eval_set = datasets.load_dataset("mt-bench/alpaca_eval", "alpaca_eval")["eval"]

#eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval",trust_remote_code=True)["eval"]

#print('dataset ok')
#eval_set = datasets.load_dataset("alpaca_eval", "alpaca_eval")["eval"]
from datasets import load_dataset

dataset = load_dataset("trl-internal-testing/hh-rlhf-trl-style")
print("hh-rlhf loading finished")


#load model directly

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("daryl149/llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("daryl149/llama-2-7b-hf")
print("llama-2-7b loading finished")
