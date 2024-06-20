# install open assistant model_training module (e.g. run `pip install -e .` in `model/` directory of open-assistant repository)
import model_training.models.reward_model  # noqa: F401 (registers reward model for AutoModel loading)
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
model_name="OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
rm = AutoModelForSequenceClassification.from_pretrained(model_name)
input_text = "\n<|user|>\nHi how are you?\n<|assistant|>\nHi, I am Open-Assistant a large open-source language model trained by LAION AI. How can I help you today?"
input_text_origin = "<|prompter|>Hi how are you?<|endoftext|><|assistant|>Hi, I am Open-Assistant a large open-source language model trained by LAION AI. How can I help you today?<|endoftext|>"

#inputs = tokenizer(input_text, return_tensors="pt")
#output=rm(**inputs)
#score = output.logits[0].cpu().detach()
#print("change prompt score: ",score)

inputs = tokenizer(input_text_origin, return_tensors="pt")
output=rm(**inputs)
print(output)
score = output.logits[0].cpu().detach()
print("origin prompt score: ",score)

print("="*60)

batch_input=([input_text,input_text,input_text,input_text,input_text])
# print(batch_input)
inputs = tokenizer(batch_input, return_tensors="pt")
# print(inputs)
output=rm(**inputs)
print(output)
score = output.logits.cpu().detach()
print("batch score: ",score)
