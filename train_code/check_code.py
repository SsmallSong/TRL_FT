# install open assistant model_training module (e.g. run `pip install -e .` in `model/` directory of open-assistant repository)
import model_training.models.reward_model  # noqa: F401 (registers reward model for AutoModel loading)
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

model_name="OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
rm = AutoModelForSequenceClassification.from_pretrained(model_name)
input_text = "<|prompter|>Hi how are you?<|endoftext|><|assistant|>Hi, I am Open-Assistant a large open-source language model trained by LAION AI. How can I help you today?<|endoftext|>"
inputs = tokenizer(input_text, return_tensors="pt")
output=rm(**inputs)
print(type(output))
print(output.keys())
score = output.logits[0].cpu().detach()
print(score)
