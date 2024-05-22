from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth", None)
import datasets

ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")

print(ds)

from langchain.docstore.document import Document as LangchainDocument

RAW_KNOWLEDGE_BASE = [
            LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
            ]

from langchain.text_splitter import RecursiveCharacterTextSplitter

# We use a hierarchical list of separators specifically tailored for splitting Markdown documents
# This list is taken from LangChain's MarkdownTextSplitter class.
MARKDOWN_SEPARATORS = [
            "\n#{1,6} ",
                "```\n",
                    "\n\\*\\*\\*+\n",
                        "\n---+\n",
                            "\n___+\n",
                                "\n\n",
                                    "\n",
                                        " ",
                                            "",
                                            ]

text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # the maximum number of characters in a chunk: we selected this value arbitrarily
                chunk_overlap=100,  # the number of characters to overlap between chunks
                    add_start_index=True,  # If `True`, includes chunk's start index in metadata
                        strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
                            separators=MARKDOWN_SEPARATORS,
                            )

docs_processed = []
for doc in RAW_KNOWLEDGE_BASE:
        docs_processed += text_splitter.split_documents([doc])

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
                multi_process=True,
                    model_kwargs={"device": "cuda"},
                        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
                        )

KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
            docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
            )

user_query = "How to create a pipeline object?"
query_vector = embedding_model.embed_query(user_query)

import pacmap
import numpy as np
import plotly.express as px

embedding_projector = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1)

embeddings_2d = [
            list(KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n(idx, 1)[0]) for idx in range(len(docs_processed))
            ] + [query_vector]

# fit the data (The index of transformed data corresponds to the index of the original data)
documents_projected = embedding_projector.fit_transform(np.array(embeddings_2d), init="pca")

df = pd.DataFrame.from_dict(
            [
                        {
                                        "x": documents_projected[i, 0],
                                                    "y": documents_projected[i, 1],
                                                                "source": docs_processed[i].metadata["source"].split("/")[1],
                                                                            "extract": docs_processed[i].page_content[:100] + "...",
                                                                                        "symbol": "circle",
                                                                                                    "size_col": 4,
                                                                                                            }
                                for i in range(len(docs_processed))
                                    ]
                + [
                            {
                                            "x": documents_projected[-1, 0],
                                                        "y": documents_projected[-1, 1],
                                                                    "source": "User query",
                                                                                "extract": user_query,
                                                                                            "size_col": 100,
                                                                                                        "symbol": "star",
                                                                                                                }
                                ]
                )

# visualize the embedding
fig = px.scatter(
            df,
                x="x",
                    y="y",
                        color="source",
                            hover_data="extract",
                                size="size_col",
                                    symbol="symbol",
                                        color_discrete_map={"User query": "black"},
                                            width=1000,
                                                height=700,
                                                )
fig.update_traces(marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")), selector=dict(mode="markers"))
fig.update_layout(
            legend_title_text="<b>Chunk source</b>",
                title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>",
                )
fig.show()

