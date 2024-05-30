# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-1.5-34B-Chat")
model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-1.5-34B-Chat")

kill
# Load model directly
from transformers import AutoModelForCausalLM,AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-13B-Chat", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-13B-Chat", use_fast=False, trust_remote_code=True)
query="hello"
from transformers.generation.utils import GenerationConfig
import torch 
model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan-13B-Chat")
messages = []
messages.append({"role": "user", "content": "世界上第二高的山峰是哪座"})

messages=tokenizer.apply_chat_template(
            messages,
                tokenize=False,
                    add_generation_prompt=True
                    )
query=tokenizer.apply_chat_template(
            query,
                tokenize=False,
                    add_generation_prompt=True
                    
                    )
print(query)
print("====================================")
print(messages)
kill

import json
import os
import gzip

# 读取JSON文件
input_file = '/home/wxt/huatong/TRL_FT/rag/article.json.gz'
output_dir = '/home/wxt/huatong/renmin_docs'

# # 如果输出目录不存在，则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取JSON文件内容到一个列表
with gzip.open(input_file, 'rt', encoding='utf-8') as f:
    data = json.load(f)


# 将每个JSON对象写入单独的TXT文件
for i, article in enumerate(data, start=1):
    file_name = os.path.join(output_dir, f"{i}.txt")
    with open(file_name, 'w', encoding='utf-8') as f:
        title = article['title'].replace('\r\n', ' ')
        content = article['content'].replace('\r\n', ' ').replace('             ', ' ')
        
        # 写入URL
        f.write(f"URL: {article['url']}\n")
        # 写入标题
        f.write(f"Title: {title}\n")
        # 写入内容
        f.write(f"Content:\n{content}\n")

print("所有文章已成功保存为TXT文件。")
