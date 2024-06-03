from FlagEmbedding import FlagModel
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from FlagEmbedding import FlagReranker
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
query_list=['全球首个“双奥之城”是哪个城市？', '中国第一大贸易伙伴是？', '《宝水》的作者是谁？', '俄罗斯国立图书馆是什么电影的取景地？', '摩中战略伙伴关系是哪一年建立的？', '“2023·中国西藏发展论坛”的主题是什么？', '“澳门科学一号”卫星由哪所高校研制？', 'G60高速公路的起点是什么？', '河西走廊的“母亲河”是哪条河？', '第三届中非经贸博览会在哪里举行？', '2023年是中国和所罗门群岛建交多少年？', '中国海铁联运业务量第二大港是？', '中国首个出舱活动的航天飞行工程师是？', '“中华诗城”是哪里？', '第三十一届世界大学生夏季运动会开幕式在哪里举行？', '《全球发展报告2023》的主题是什么？', '第十九届亚运会中国体育代表团年龄最小的运动员是？', '杭州亚运会组委会主席是谁？', '《共建“一带一路”：构建人类命运共同体的重大实践》白皮书什么时候发布？', '神舟十七号载人飞船是长征系列运载火箭的第几次飞行？', '2024年4月1日报纸的11版面版式设计是由谁负责的？', '哪位教授的团队首次观察到了引力子的投影？', '哪位福建省习近平新时代中国特色社会主义思想研究中心特约研究员供稿了2024年4月8日第9版？', '2024年4月9日，哪位泰国公主访华？', '国家体育总局经济司负责人是谁？', '两岸企业家峰会10周年年会的主题是什么？', '谁在第十八次全国代表大会上代表武警部队发言致辞？', '虞城县荠菜协会会长是谁？', '第十届世界互联网大会乌镇峰会的主题是什么？', '2023年全国政协新年茶话会是什么时候举行的？', '我国国防部部长是何时被任命的？', '2023年前三季度中国柜内生产总值同比增长多少？', '谁主持了甘苦同志诞辰100周年座谈会？', '第十四届全国冬季运动会的主题MV叫什么？', '国务院第三次全体会议主题是什么？', '嘉汇汉唐书城科技生活区区域经理是谁？', '福建龙岩漳平市永福镇西山村党总支书记是谁？', '全球首个达到700万辆新能源汽车销售量的品牌是什么？', '从今年起，报考高水平运动队需要获得什么称号？', '雪车世界杯中谁夺得了女子钢架雪车项目银牌？', '《食品经营许可和备案管理办法》从何日开始施行', '2023年上半年，我国快递业务量同比增长超多少？', '2023年亚洲田径锦标赛女子4×100米接力决赛中，中国队有哪些运动员出场？', '2023年是中国和毛里塔尼亚建交多少年？', '2023年上半年机械工业战略性新兴产业相关行业累计实现营业收入多少万亿元？', '2021年美国有多少25岁以下的工人死于工作？', '2023年，塔吉克斯坦已建成多少个充电桩设备', '杭州亚运会火种采集仪式在何时举行？', '2023年中国国际服务贸易交易会何日闭幕？', '谁陪同李强总理在北京市调研专精特新企业发展情况', '第十一次全国归侨侨眷代表大会上，谁发表了致词', '2023年上半年，我国公共充电桩增量多少万台？', '第二届U15世界中学生夏季运动会中国队派出了多少名运动员？', '乌兹别克斯坦司法部长是谁？', '2023年，红其拉甫海关被评为什么？', '谁是杭州亚运会女子200米个人混合泳金牌？', '中华海外联谊会会长是谁？', '日内瓦国际车展始于哪年？', '中国历史研究院院长是谁？', '“雪龙”号大洋队队长是谁？', '获得第二十七届“中国青年五四奖章”的女性有谁？', '澜湄六国是哪六个国家？', '“四下基层”指的是什么？', '2023中关村论坛的主办方有几个？', '游泳世锦赛中国跳水队获得多少枚奖牌？', '第三届雄安·雄州文化艺术节共举行几天？', '《习近平新时代中国特色社会主义思想专题摘编》民族文字版共有几个出版社参与发行？', '2023年9月25日亚运会开幕报道第一版由几位记者联合报道？', '中科大先进技术创业服务中心截止2023年累计孵化企业多少家？', '2024年2月中国制造业采购经理指数大概多少？', '今年是钱塘潮涌潮观测站启用第几年？', '我国医保码哪一年正式上线的？', '根据国务院发展研究中心金融研究所副所长的论点，可以从哪几个方面建设强大的中央银行？', '“联合国教科文组织世界地质公园”标识创建至今多少年了？', '2023年亚洲田径锦标赛女子标枪中，中国队有哪些运动员获得名次？', '新中国成立以来第一次全国双拥工作会议上，习近平同志赋诗的具体内容是？', '湖南郴州苏仙区的钟家村涌水澜组的村民罗心树今年多少岁？', '十四届全国人大常委会第五次会议联组会议，记录了哪些人的发言？', '1984年，习近平同志任什么职位（不含地名）？', '杭州亚运会火炬在几个城市传递？']
query_list_1=[
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
    prompt_now='''你是一个问答机器人，结合提供的参考资料，按照示例的格式回答问题。问题都是填空题，所以你的答案要简洁。下面是一些问答示例：\n查询：谁主持了国务院第七次专题学习？\n答案：李强\n\n查询：第十一届茅盾文学奖获奖作品有哪些？\n答案：《雪山大地》；《宝水》；《本巴》；《千里江山图》；《回响》\n\n查询：中国艺术体操队的首个世界冠军是在哪个城市取得的？\n答案：西班牙瓦伦西亚\n\n下面是若干项相关的参考资料，请根据这些资料作答：\n\n'''
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
