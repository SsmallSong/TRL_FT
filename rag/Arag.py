# # -*- coding: utf-8 -*-
"""RAG System Using Llama2 With Hugging Face.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HsKNhtqnoH3G2wdgr8-iN0Snj7kbCJKt
"""

# !pip install pypdf

# !pip install transformers einops accelerate langchain bitsandbytes

# ## Embeddings
# !pip install sentence_transformers

# !pip install llama_index

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt

documents= SimpleDirectoryReader('/home/wxt/huatong/renmin_docs').load_data()
print(type(documents))
print(len(documents))
print(documents[0])

system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
ccurately as possible based on the instructions and context provided.
"""

## Default Format Supportable By Llama2
query_wrapper_prompt= SimpleInputPrompt("<USER|>{query_string}<|ASSISTANT>")

query_wrapper_prompt

# !huggingface-cli login

import torch
model_id='mistralai/Mistral-7B-Instruct-v0.2'
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=model_id,
    model_name=model_id,
    device_map="auto",
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext
from llama_index.embeddings import LangchainEmbedding

embed_model=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

service_context=ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

index=VectorStoreIndex.from_documents(documents,service_context=service_context)

query_engine=index.as_query_engine()

response=query_engine.query("what is attention is all you need?")

print(response)

response=query_engine.query("what is YOLO?")

print(response)
