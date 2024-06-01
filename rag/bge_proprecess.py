from FlagEmbedding import FlagModel
import numpy as np
import os
import torch
# from sklearn.metrics.pairwise import cosine_similarity

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

# model.model.to(device)
import gzip
import json
import pickle
with gzip.open('~/huatong/TRL_FT/rag/article.json.gz', 'rt', encoding='utf-8') as f:
    data = json.load(f)
# # data=data[0:5]
print("begin get url and text")
url_list = []
text_list = []
for item in data:
    url_list.append(item['url'])
    text_list.append((item['title'] + " " + item['content']).replace("\n", "").replace("\r", ""))

def split_into_chunks(text, chunk_size=500):
    # return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    chunks = []
    while len(text) > chunk_size:
        split_index = chunk_size
        while split_index < len(text) and text[split_index] not in '。！；？':
            split_index += 1
        split_index += 1  # 包含句号
        chunks.append(text[:split_index])
        text = text[split_index:]
    chunks.append(text)
    return chunks


split_list = [split_into_chunks(s) for s in text_list]
# split_list=split_list[0:3]
embeddings_list=[]
for i, text_list in enumerate(split_list):
    embeddings_now = model.encode(text_list)
    embeddings_list.append(embeddings_now)
    
    # 每处理500个元素输出一个日志
    if (i + 1) % 500 == 0:
        print("Finish docs:",i+1)

# dict_len={}
# for sp in split_list:
#     len_now=len(sp)
#     if len_now in dict_len:
#         dict_len[len_now]+=1
#     else:
#         dict_len[len_now]=1
# dict_len=sorted(dict_len.items(), key=lambda x: x[0], reverse=False)
# print(dict_len)

# embeddings = model.encode(split_list)

with open("~/huatong/rmrb_bge_chunk500.pkl", 'wb') as file:
    pickle.dump(embeddings_list, file)

# with open("/data2/huatong/rag/rmrb_bge_chunk.pkl", 'rb') as file:
#     docs_embeddings=pickle.load(file)

# query_embeddings=model.encode_queries(query_list)
# # scores_all=query_embeddings @ docs_embeddings.T
# # scores_all = cosine_similarity(query_embeddings , docs_embeddings)
# #print(np.array(scores_all).shape)

