import json
import jieba
import re
import nltk
from rank_bm25 import BM25Okapi
import pickle
import gzip
# 确保已下载nltk的中文停用词表
nltk.download('stopwords')
import time
from nltk.corpus import stopwords

# 获取中文停用词列表
chinese_stopwords = set(stopwords.words('chinese'))

# 假设你的JSON文件名为'articles.json'
with gzip.open('/home/wxt/huatong/TRL_FT/rag/article.json.gz', 'rt', encoding='utf-8') as f:
    data = json.load(f)

url_list = []
text_list = []

for item in data:
    url_list.append(item['url'])
    text_list.append(item['title'] + " " + item['content'])

#print("URL列表:", url_list)
#print("文本列表:", text_list)

def remove_punctuation(words):
    # 定义中文标点符号
    punctuation =  {"，", "。", "！", "？", "、", "；", "：", "（", "）", "《", "》", "“", "”", "‘", "’","【","】"," ","\r\n","\n","\xa0"} | chinese_stopwords #|set(string.punctuation) 
    # 去除标点符号
    cleaned_words = [word for word in words if word not in punctuation]
    return cleaned_words

text_seg_list = []
for text in text_list:
    seg_list = jieba.lcut(text)
    clean_seg_list = remove_punctuation(seg_list)
    text_seg_list.append(clean_seg_list)

with open('/home/wxt/huatong/renmin_docs/bm25_model/rmrb_text_cut.pkl', 'wb') as f:
    pickle.dump(text_seg_list, f)


#print("文本切词列表:", text_seg_list)

#for text in text_seg_list:
 #   print(len(text))

bm25 = BM25Okapi(text_seg_list)
with open('/home/wxt/huatong/renmin_docs/bm25_model/bm25_model.pkl', 'wb') as f:
    pickle.dump(bm25, f)
    
# with open('/Users/songhuatong/Desktop/py-大三下/自然语言处理/NLP2024期末作业/rmrb_text_cut.pkl', 'rb') as f:
#     text_seg_list = pickle.load(f)

# with open('/Users/songhuatong/Desktop/py-大三下/自然语言处理/NLP2024期末作业/bm25_model.pkl', 'rb') as f:
#     bm25 = pickle.load(f)

top_k=5
query_list=["学校重视劳动教育课吗？","学校重视劳动教育课吗？","学校重视劳动教育课吗？","学校重视劳动教育课吗？","学校重视劳动教育课吗？","学校重视劳动教育课吗？"]
query=query_list[0]
seg_list = jieba.lcut(query)
tokenized_query = remove_punctuation(seg_list)
scores = bm25.get_scores(tokenized_query)
top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

# 输出结果
print("Top-{} 最相关的文档索引: {}".format(top_k, top_k_indices))

# print(text_seg_list[top_k_indices[0]])
