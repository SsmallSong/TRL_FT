# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# import llm_blender
# blender = llm_blender.Blender()
# blender.loadranker("llm-blender/PairRM") # load PairRM

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback")
model = AutoModelForSequenceClassification.from_pretrained("Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback")

# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5")

# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("openbmb/Eurus-RM-7b", trust_remote_code=True)
model = AutoModel.from_pretrained("openbmb/Eurus-RM-7b", trust_remote_code=True)

from datasets import load_dataset

<<<<<<< HEAD
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
import llm_blender
blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM") # load PairRM
from datasets import load_dataset

dataset = load_dataset("snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset")
=======
dataset = load_dataset("tatsu-lab/alpaca_eval")
>>>>>>> cfc8fcb576b31bdc3f5c085b2edd055359407dc8
