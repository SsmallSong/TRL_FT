# install open assistant model_training module (e.g. run `pip install -e .` in `model/` directory of open-assistant repository)
import model_training.models.reward_model  # noqa: F401 (registers reward model for AutoModel loading)

tokenizer = AutoTokenizer.from_pretrained(model_name)
rm = AutoModelForSequenceClassification.from_pretrained(model_name)
input_text = "<|prompter|>Hi how are you?<|endoftext|><|assistant|>Hi, I am Open-Assistant a large open-source language model trained by LAION AI. How can I help you today?<|endoftext|>"
inputs = tokenizer(input_text, return_tensors="pt")
score = rm(**inputs).logits[0].cpu().detach()
print(score)
