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
print("111111111")
# Load model directly
<<<<<<< HEAD
from transformers import AutoModel
model = AutoModel.from_pretrained("OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5")
print('2222222222')
=======
# from transformers import AutoModel
# model = AutoModel.from_pretrained("OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5")

>>>>>>> 8c16173bdfb45a441508be46f2626c58d146252e
# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("openbmb/Eurus-RM-7b", trust_remote_code=True)
model = AutoModel.from_pretrained("openbmb/Eurus-RM-7b", trust_remote_code=True)
print('333333333')
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca_eval")
