#%%

import pinecone
import os
from dotenv import load_dotenv
import os
import urllib.request
from pathlib import Path
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index.text_splitter import SentenceSplitter
from llama_index.schema import TextNode
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
)
from llama_index.vector_stores import PineconeVectorStore
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
import openai
from llama_index import VectorStoreIndex
from llama_index.storage import StorageContext

load_dotenv()

if not os.path.exists('data'):
    os.makedirs('data')

pinecone_api_key = os.environ["PINECONE_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]

pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
pinecone.list_indexes()

#%%

# dimensions are for text-embedding-ada-002
#pinecone.create_index("quickstart", dimension=1536, metric="euclidean", pod_type="p1")
pinecone_index = pinecone.Index("quickstart")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

#%%

url = 'https://arxiv.org/pdf/2307.09288.pdf'
filename = 'data/llama2.pdf'
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'

req = urllib.request.Request(url, headers={'User-Agent': user_agent})
with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
    data = response.read()
    out_file.write(data)

#%%

loader = PyMuPDFReader()
documents = loader.load(file_path="./data/llama2.pdf")

#%%

text_splitter = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
)

text_chunks = []
# maintain relationship with source doc index, to help inject doc metadata in (3)
doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_text_chunks = text_splitter.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

print(len(documents))
print(len(text_chunks))
print(len(doc_idxs))

#%%

nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)
    
print(len(nodes))

# print a sample node
print(nodes[0].get_content(metadata_mode="all"))

#%%

llm = OpenAI(model="gpt-3.5-turbo")

metadata_extractor = MetadataExtractor(
    extractors=[
        TitleExtractor(nodes=5, llm=llm),
        QuestionsAnsweredExtractor(questions=3, llm=llm),
    ],
    in_place=False,
)

#%%

nodes = metadata_extractor.process_nodes(nodes)
print(nodes[0])

#%%

embed_model = OpenAIEmbedding()

for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

#%%

print(len(nodes[0].embedding))

#%%

vector_store.add(nodes)

#%%

index = VectorStoreIndex.from_vector_store(vector_store)

#%%

query_engine = index.as_query_engine()

#%%

query_str = "Can you tell me about the key concepts for safety finetuning"
response = query_engine.query(query_str)
print(str(response))

#%%

#%%

#%%

#%%