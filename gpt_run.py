# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, set_seed
from datasets import load_dataset
from trl import SFTTrainer
device = "cuda" # the device to load the model onto

tokenizer = AutoTokenizer.from_pretrained("/data2/huatong/model/gpt2-sft")
model = AutoModelForCausalLM.from_pretrained("/data2/huatong/model/gpt2-sft")

#Text Generated:
prompt_text = "Tell me a cute story"
input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

output_ids = model.generate(input_ids, max_length=100, num_return_sequences=3, 
                            pad_token_id=tokenizer.eos_token_id,
                            do_sample=True,  # 使用采样而不是贪心搜索
                            top_k=50,         # 控制采样时考虑的最大词汇数量
                            top_p=0.95,       # 控制采样时的累积概率阈值
                            temperature=0.7,  # 控制模型生成的多样性，值越高生成的结果越随机
                            no_repeat_ngram_size=2  # 防止生成的文本中出现重复的n-gram
                            )
generated_text=[]
for i in range(len(output_ids)):
    generated_text.append(tokenizer.decode(output_ids[i], skip_special_tokens=True))
print("Generated Text:")

for i in range(len(generated_text)):
    print(generated_text[i])
    print('')


# #Text Generated in other way:
# generator = pipeline('text-generation', model='gpt2')
# set_seed(42)
# generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)


# #Get the features of given text:
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)