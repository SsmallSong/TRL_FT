
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="1"
print(torch.cuda.device_count())
import chromadb
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.core.query_engine import CitationQueryEngine

documents= SimpleDirectoryReader('/home/wxt/huatong/renmin_docs').load_data()
#print(type(documents))
#print(len(documents))
#print(documents[0])
#documents=documents[0:20]
system_prompt="""
你是一个问答助手。你的目标是根据提供的指令和上下文尽可能准确地回答问题。
你的所有回答除了给定格式外都应该是中文的。
知识库中每篇文章都提供了url，每回答一个问题，都要在后面同时给出相关文档的url。
答案格式如下:
"answer:{$answer}\nrelated-urls:{$related-urls}\n "
"""


query_wrapper_prompt= SimpleInputPrompt("<USER|>{query_string}<|ASSISTANT>")

modelid='mistralai/Mistral-7B-Instruct-v0.2'
modelid="itpossible/Chinese-Mistral-7B-Instruct-v0.1"
modelid="baichuan-inc/Baichuan2-7B-Chat"
llm = HuggingFaceLLM(
    context_window=1024,
    max_new_tokens=512,
 #   trust_remote_code=True,
    generate_kwargs={"pad_token_id": 2,
            "temperature": 0.2, "do_sample": True},
#    system_prompt=system_prompt,
#    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=modelid,
    model_name=modelid,
    device_map="auto",
    model_kwargs={"trust_remote_code":True,"torch_dtype": torch.float16},
    tokenizer_kwargs={"trust_remote_code":True}
    #model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)


def Mistral_instruct_query(questionText):
#    questionText = "<s>[INST] " + questionText + " [/INST]"
    return questionText

# embed_model=LangchainEmbedding(
#     HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

embed_model=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5"))
Settings.embed_model = embed_model
Settings.llm = llm

#service_context=ServiceContext.from_defaults(
 #   chunk_size=1024,
 #   llm=llm,
 #   embed_model=embed_model
#)

print("Begin Index")

db = chromadb.PersistentClient(path="/home/wxt/huatong/rmrb_chroma_db_zh")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#index=VectorStoreIndex.from_documents(documents,storage_context=storage_context)
index = VectorStoreIndex.from_vector_store( vector_store, storage_context=storage_context)
print("Finish Index")
# query_engine = CitationQueryEngine.from_args(
#             index, 
#             similarity_top_k=3, 
#             citation_chunk_size=256,
#                     )
query_engine=index.as_query_engine()

response=query_engine.query(Mistral_instruct_query("2024年是中国红十字会成立多少周年?"))
print(response)

response=query_engine.query(Mistral_instruct_query("《中华人民共和国爱国主义教育法》什么时候实施？"))
print(response)

response=query_engine.query(Mistral_instruct_query("2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？"))
print(response)

response=query_engine.query(Mistral_instruct_query("2024年我国文化和旅游部部长是谁？"))
print(response)

response=query_engine.query(Mistral_instruct_query("2023—2024赛季国际滑联短道速滑世界杯北京站比赛中，刘少昂参与获得几枚奖牌？"))
print(response)

response=query_engine.query(Mistral_instruct_query("福建自贸试验区在自贸建设十年中主要从哪几个方面推动改革创新？"))
print(response)

