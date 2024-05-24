from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer
import torch

documents = SimpleDirectoryReader("data").load_data()

# nomic embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# ollama
# Settings.llm = Ollama(model="llama3", request_timeout=360.0)
model_id='mistralai/Mistral-7B-Instruct-v0.2'
Settings.llm = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16, device_map=device)
print("Model Loading Finished!")

# index = VectorStoreIndex.from_documents(
#     documents,
# )