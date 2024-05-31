
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
print(torch.cuda.device_count())
import chromadb
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import PromptTemplate
from huggingface_hub import snapshot_download
from FlagEmbedding import FlagReranker
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.llms import ChatMessage, MessageRole
documents= SimpleDirectoryReader('/home/wxt/huatong/renmin_docs').load_data()
#print(type(documents))
#print(len(documents))
#print(documents[0])
#documents=documents[0:20]
system_prompt="""
你是一个问答助手。你的目标是根据提供的指令和上下文尽可能准确地回答问题。
你的所有回答都应该是中文的。
"""



modelid='mistralai/Mistral-7B-Instruct-v0.2'
modelid="itpossible/Chinese-Mistral-7B-Instruct-v0.1"
modelid="baichuan-inc/Baichuan2-7B-Chat"
modelid="01-ai/Yi-1.5-34B-Chat"

# modelid="baichuan-inc/Baichuan2-13B-Chat"
llm = HuggingFaceLLM(
    context_window=3072,
    max_new_tokens=1024,
    generate_kwargs={"pad_token_id": 2,
            "temperature": 0.2, "do_sample": True},
 #   system_prompt=system_prompt,
    tokenizer_name=modelid,
    model_name=modelid,
    device_map="auto",
    model_kwargs={"trust_remote_code":True,"torch_dtype": torch.float16},
    tokenizer_kwargs={"trust_remote_code":True}

)


def Mistral_instruct_query(questionText):
    questionText = "<s>[INST] " + questionText + " [/INST]"
    return questionText

def Baichuan_instruct_query(questionText):
    questionText="<reserved_106>"+questionText #+ "<reserved_107>"
    return questionText

def Origin_instruct_query(questionText):
    return questionText

def Chat_instruct_query(questionText,modelid,use_chat=True):
    if use_chat==False:
        questionText=Origin_instruct_query(questionText)
    else:
        if modelid.split('/')[0]=="baichuan-inc":
            questionText=Baichuan_instruct_query(questionText)
        elif modelid.split('/')[0]=="mistralai" or modelid.split('/')[0]=="itpossible":
            questionText=Mistral_instruct_query(questionText)
        else:
            questionText=Origin_instruct_query(questionText)
    return questionText

# embed_model=LangchainEmbedding(
#     HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

embed_model=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5"))

Settings.embed_model = embed_model
Settings.llm = llm

#service_context=ServiceContext.from_defaults(
 #   chunk_size=1024,
 #   llm=llm,
 #   embed_model=embed_model
#)

print("Begin Index")

db = chromadb.PersistentClient(path="/home/wxt/huatong/rmrb_chroma_db_zh_base")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#index=VectorStoreIndex.from_documents(documents,storage_context=storage_context)
index = VectorStoreIndex.from_vector_store( vector_store, storage_context=storage_context)
print(index)
print("Finish Index")

rerank_llm_name = "BAAI/bge-reranker-v2-m3"
#downloaded_rerank_model = snapshot_download(rerank_llm_name)
reranker= SentenceTransformerRerank(model=rerank_llm_name, top_n=10)
#reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
print("rerank-llm finished")
# query_engine = CitationQueryEngine.from_args(
#             index, 
#             similarity_top_k=3, 
#             citation_chunk_size=256,
#                     )
from llama_index.core import ChatPromptTemplate

new_summary_tmpl_str_zh=(
"根据提供的内容来回答问题\n"
"---------------------\n"
"{context_str}\n"
"---------------------\n"
"仅仅根据上面提供的知识，不要考虑任何先验知识，回答下面的问题\n"
"{query_str}"
        )
chat_template = ChatPromptTemplate(message_templates=[
        ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=new_summary_tmpl_str_zh)
            ])


query_engine=index.as_query_engine(similarity_top_k=3,
                                    node_postprocessors=[reranker],
                                    )

query_engine.update_prompts(
            prompts_dict={"response_synthesizer:text_qa_template": chat_template}
            )
#query_engine=index.as_query_engine()
print("query_engine finished")
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

    # "2024年是中国红十字会成立多少周年?",
    # "《中华人民共和国爱国主义教育法》什么时候实施？",
    # "2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？",
    # "2024年我国文化和旅游部部长是谁？",
    # "2023—2024赛季国际滑联短道速滑世界杯北京站比赛中，刘少昂参与获得几枚奖牌？",
    # "福建自贸试验区在自贸建设十年中主要从哪几个方面推动改革创新？"
]


print("begin gen-answer")
for i in range(len(query_list)):
    #response=query_engine.query(Chat_instruct_query(query_list[i],modelid,use_chat=False))
    response=query_engine.query(query_list[i])
    print(response)
