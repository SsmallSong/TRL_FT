
from FlagEmbedding import FlagModel
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from FlagEmbedding import FlagReranker
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

query_list=['简述总书记的“青年观”。', '“两个结合”的重大意义是什么？', '入伏时间如何确定？', '简述巴中经济走廊建设取得的成就？', '如何把红色基因传承下去？', '简述新时代对台工作的根本遵循和行动指南？', 
 '简述如何增强土地要素对优势地区高质量发展保障能力?', '结合真实脱贫案例，简述我国可以采取怎样的措施来帮助贫困人口脱贫？', '结合近年来的标志事件，简述我国该怎样更好的参与国际合作？', 
 '针对近期频发的食品安全事件，结合相关法案，简述如何让人民群众吃得放心', '党政机关干部应该梳理怎样的政绩观？结合相关思想简述一下。','简述我们应当如何从马克思主义层面提升自己的思维能力？',
   '结合相关报道，简述应该如何提升我们的文化自信？', '结合相关报道，简述应该如何化解城市公交经营困境？', '简述如何做好干部教育培训', '简述党的自我革命', '如何促进股权投资行业高质量发展', 
   '结合具体事例，介绍我国冰雪经济发展情况', '根据2024年已经发生的事情，写一段2025年新年贺词', '结合具体事例，讲解“江山就是人民”的深刻内涵']


# sentences_1 = ["样例数据-1", "样例数据-2"]
# sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

import gzip
import json
import pickle
import jieba
import re
import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
chinese_stopwords = set(stopwords.words('chinese'))
with gzip.open('/home/wxt/huatong/TRL_FT/rag/article.json.gz', 'rt', encoding='utf-8') as f:
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

with open("/home/wxt/huatong/rmrb_bge_chunk1000.pkl", 'rb') as file:
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


# #加载mdoel
# model_path = "/data2/huatong/model/01ai"
model_path="01-ai/Yi-1.5-34B-Chat"

print("begin load model")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16).eval()
#model.to(device)

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

top_k_recall=15
top_k_final=1
print("begin get response")
response_list=[]
for i in range(len(query_list)):
    
    prompt_now='''你是一个问答机器人，结合提供的参考资料回答问题。问题是简答题，回答要完整。\n下面是若干项相关的参考资料，请根据这些资料作答：\n\n'''
    print("===============================================================")
    bge_scores= scores_all[i]

    top_k_indices_recall_bge = sorted(range(len(bge_scores)), key=lambda i: bge_scores[i], reverse=True)[:top_k_recall]
    rerank_list_bge=[]
    for g in top_k_indices_recall_bge:
        rerank_list_bge.append([query_list[i],text_list[g]])
    rerank_score_bge = reranker.compute_score(rerank_list_bge, normalize=True)
    top_k_indices_in_rerankls_bge = sorted(range(len(rerank_score_bge)), key=lambda i: rerank_score_bge[i], reverse=True)[:top_k_final]
    top_k_indices_bge =[top_k_indices_recall_bge[ele] for ele in top_k_indices_in_rerankls_bge]
 #   print("Bge-Recall-{} 相关的文档url: {}".format(top_k_recall, [url_list[i] for i in top_k_indices_recall_bge]))
    # top_k_indices_bge = sorted(range(len(bge_scores)), key=lambda i: bge_scores[i], reverse=True)[:top_k]
    # print("Bge-Top-{} 最相关的文档索引: {}".format(top_k, top_k_indices))
    print("Bge-Top-{} 最相关的文档url: {}".format(top_k_final, [url_list[i] for i in top_k_indices_bge]))
    for j in top_k_indices_bge:
        prompt_now=prompt_now+text_list[j]+'\n'

    seg_list = jieba.lcut(query_list[i])
    tokenized_query = remove_punctuation(seg_list)
    bm25_scores = bm25.get_scores(tokenized_query)

    top_k_indices_recall_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k_recall]
    rerank_list_bm25=[]
    for g in top_k_indices_recall_bm25:
        rerank_list_bm25.append([query_list[i],text_list[g]])
    rerank_score_bm25 = reranker.compute_score(rerank_list_bm25, normalize=True)
    top_k_indices_in_rerankls_bm25 = sorted(range(len(rerank_score_bm25)), key=lambda i: rerank_score_bm25[i], reverse=True)[:top_k_final]
    top_k_indices_bm25 =[top_k_indices_recall_bm25[ele] for ele in top_k_indices_in_rerankls_bm25]
    # top_k_indices_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    # print("BM25-Top-{} 最相关的文档索引: {}".format(top_k, top_k_indices))
    print("Bm25-Top-{} 最相关的文档url: {}".format(top_k_final, [url_list[i] for i in top_k_indices_bm25]))
    for j in top_k_indices_bm25:
        if j not in top_k_indices_bge:
            prompt_now=prompt_now+text_list[j]+'\n'

    prompt_now=prompt_now+"\n根据上面提供的资料，回答如下问题：\n查询："+query_list[i]+"\n请用如下格式回答：\n答案：{$ans}"
    messages=[{"role": "user", "content": prompt_now}]
    #messages=torch.tensor(messages).to(device)
    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id, max_new_tokens=100)
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    print(query_list[i])
    print(response)
    response_list.append(response)
print(response_list)
res_list=[ele[ele.index("：")+1:] for ele in response_list]
print(res_list)
