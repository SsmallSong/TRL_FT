from FlagEmbedding import FlagModel
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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
"2023年广西植树造林面积大约多少亩？",
"在第二十八个世界读书日，构建海洋命运共同体理念已经提出几周年了？",
"中央外事工作会议指出，新时代新征程中国特色大国外交方针原则有“四个坚持”，是哪四个",
"第十一届茅盾文学奖获奖作品有哪些?",
"第四届“光影中国”荣誉盛典获“荣誉推编剧有谁？"
]

# sentences_1 = ["样例数据-1", "样例数据-2"]
# sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

import gzip
import json
import pickle
with gzip.open('/data2/huatong/rag/article.json.gz', 'rt', encoding='utf-8') as f:
    data = json.load(f)
# # data=data[0:5]
print("begin get url and text")
url_list = []
text_list = []
for item in data:
    url_list.append(item['url'])
    text_list.append((item['title'] + " " + item['content']).replace("\n", "").replace("\r", ""))
# embeddings = model.encode(text_list)
# with open("/data2/huatong/rag/rmrb_bge.pkl", 'wb') as file:
#     pickle.dump(embeddings, file)

with open("/data2/huatong/rag/rmrb_bge_chunk.pkl", 'rb') as file:
    docs_embeddings=pickle.load(file)
# print(np.array(docs_embeddings).shape)

query_embeddings=model.encode_queries(query_list)
# scores_all=query_embeddings @ docs_embeddings.T
print("begin score")
scores_all = np.zeros((len(query_embeddings), len(docs_embeddings)))
for i in range(len(query_embeddings)):
    for j in range(len(docs_embeddings)):
        chunks = docs_embeddings[j]
        chunk_scores = [query_embeddings[i] @ chunk.T for chunk in chunks]
        scores_all[i, j] = max(chunk_scores)
print("finish score")

# #加载mdoel
model_path = "/data2/huatong/model/01ai"
model_path="01-ai/Yi-1.5-9B-Chat"

print("begin load model")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16).eval()
#model.to(device)
top_k=1
tok_k_docs_index = []
print("begin get response")
for i in range(len(query_list)):
    scores= scores_all[i]
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    prompt_now='''你是一个问答机器人，结合提供的参考资料，回答我的问题。\n\n参考资料如下：\n'''
    print("Top-{} 最相关的文档索引: {}".format(top_k, top_k_indices))
    print("Top-{} 最相关的文档url: {}".format(top_k, [url_list[i] for i in top_k_indices]))
    for i in top_k_indices:
        prompt_now=prompt_now+text_list[i]+'\n'
    prompt_now=prompt_now+"\n问题如下：\n"+query_list[i]
    messages=[{"role": "user", "content": prompt_now}]
    #messages=torch.tensor(messages).to(device)
    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    print(query_list[i])
    print(response)