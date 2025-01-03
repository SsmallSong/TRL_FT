from FlagEmbedding import FlagModel
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import gzip
import json
import pickle
import jieba
import re
import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
def evaluate(queries: list):
    """
    queries: List[str] 输入查询列表
    Return: List[str] 输出答案列表
    """
    ans_list=[]
    model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
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

    with open("/home/wxt/huatong/rmrb_bge_chunk500.pkl", 'rb') as file:
        docs_embeddings=pickle.load(file)
    # print(np.array(docs_embeddings).shape)

    query_embeddings=model.encode_queries(queries)
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
    model_path="01-ai/Yi-1.5-9B-Chat"

    print("begin load model")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16).eval()
    #model.to(device)
    top_k=1
    tok_k_docs_index = []
    print("begin get response")
    for i in range(len(queries)):
        prompt_now='''你是一个问答机器人，结合提供的参考资料，回答我的问题，问题是填空题，答案要简洁明了。\n\n下面是一个例子：\n问题：谁主持了国务院第七次专题学习？\n答案：李强\n\n参考资料如下：\n'''
        print("===============================================================")
        bge_scores= scores_all[i]
        top_k_indices_bge = sorted(range(len(bge_scores)), key=lambda i: bge_scores[i], reverse=True)[:top_k]
        # print("Bge-Top-{} 最相关的文档索引: {}".format(top_k, top_k_indices))
        print("Bge-Top-{} 最相关的文档url: {}".format(top_k, [url_list[i] for i in top_k_indices_bge]))
        for j in top_k_indices_bge:
            prompt_now=prompt_now+text_list[j]+'\n'

        seg_list = jieba.lcut(query_list[i])
        tokenized_query = remove_punctuation(seg_list)
        bm25_scores = bm25.get_scores(tokenized_query)
        top_k_indices_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        # print("BM25-Top-{} 最相关的文档索引: {}".format(top_k, top_k_indices))
        print("Bm25-Top-{} 最相关的文档url: {}".format(top_k, [url_list[i] for i in top_k_indices_bm25]))
        for j in top_k_indices_bm25:
            if j not in top_k_indices_bge:
                prompt_now=prompt_now+text_list[j]+'\n'

        prompt_now=prompt_now+"\n问题："+queries[i]
        messages=[{"role": "user", "content": prompt_now}]
        #messages=torch.tensor(messages).to(device)
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id,max_new_tokens=100)
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        ans_list.append(response)
        print(queries[i])
        print(response)
    return ans_list
