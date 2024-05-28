import json
import jieba
import re
import nltk
from rank_bm25 import BM25Okapi
import pickle
import gzip
import time
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.device_count())

#加载mdoel
model_path = "itpossible/Chinese-Mistral-7B-Instruct-v0.1"
model_path = "baichuan-inc/Baichuan2-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-7B-Chat")
model.to(device)

#编写retriever
chinese_stopwords = set(stopwords.words('chinese'))

with gzip.open('/home/wxt/huatong/TRL_FT/rag/article.json.gz', 'rt', encoding='utf-8') as f:
    data = json.load(f)

url_list = []
text_list = []

for item in data:
    url_list.append(item['url'])
    text_list.append((item['title'] + " " + item['content']).replace("\n", "").replace("\r", ""))
def remove_punctuation(words):
    # 定义中文标点符号
    punctuation =  {"，", "。", "！", "？", "、", "；", "：", "（", "）", "《", "》", "“", "”", "‘", "’","【","】"," ","\r\n","\n","\xa0"} | chinese_stopwords #|set(string.punctuation) 
    # 去除标点符号
    cleaned_words = [word for word in words if word not in punctuation]
    return cleaned_words
    
with open('/home/wxt/huatong/renmin_docs/bm25_model/rmrb_text_cut.pkl', 'rb') as f:
    text_seg_list = pickle.load(f)

with open('/home/wxt/huatong/renmin_docs/bm25_model/bm25_model.pkl', 'rb') as f:
    bm25 = pickle.load(f)

top_k=3
query_list=[
"谁主持了国务院第七次专题学习？",
"重庆市潼南区文化和旅游发展委员会党组书记、主任是谁？",
"元古堆村村委会主任是谁？",
"《中华人民共和国国务院组织法》什么时候公布？",
"陕西省延安市安塞区高桥镇南沟村党支书是谁？",
"首艘长江支线换电电池动力集装箱班轮是什么？",
"国家数据局挂牌时间是什么时候？",
"北京长峰医院发生重大火灾事故造成多少人死亡？",
"2023上半年机械工业增加值同比增长多少？",
"中国艺术体操队的首个世界冠军是在哪个城市取得的？",
"长江生态环境保护民主监督启动于什么时候？",
"联合国教科文组织在促进女童和妇女教育领域的唯一奖项是什么？",
"2023年是纪念中美“乒乓外交”多少周年？",
"哈尔滨亚冬会包含几个大项？",
"全国人大常委会副委员长、中华全国总工会主席是谁",
]
tok_k_docs_index = []
for query in query_list:
    seg_list = jieba.lcut(query)
    tokenized_query = remove_punctuation(seg_list)
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    prompt_now='''你是一个问答机器人，结合提供的参考资料，回答我的问题。\n\n参考资料如下：\n'''
    print("Top-{} 最相关的文档索引: {}".format(top_k, top_k_indices))
    print("Top-{} 最相关的文档url: {}".format(top_k, [url_list[i] for i in top_k_indices]))
    for i in top_k_indices:
        prompt_now=prompt_now+text_list[i]+'\n'
    prompt_now=prompt_now+"\n问题如下：\n"+query
    messages=[{"role": "user", "content": prompt_now}]
    messages=messages.to(device)
    response = model.chat(tokenizer, messages)
    print(query)
    print(response)
    # print(prompt_now)

# 输出结果
# print("Top-{} 最相关的文档索引: {}".format(top_k, top_k_indices))

# print(text_seg_list[top_k_indices[0]])
