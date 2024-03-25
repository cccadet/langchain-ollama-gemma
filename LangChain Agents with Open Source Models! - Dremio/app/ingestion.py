from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DocusaurusLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader

embeddings = OllamaEmbeddings(model='nomic-embed-text:v1.5')
vectorestore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

# Docsaurus 
#loader = DocusaurusLoader(
#    "https://docs.dremio.com/",
#    filter_urls=[
#        "https://docs.dremio.com/current/get-started/cluster-deployments/customizing-configuration/dremio-conf/"
#    ],
#)

loader = WebBaseLoader("https://docs.dremio.com/current/get-started/cluster-deployments/architecture/distributed-storage")


documents = loader.load()

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100
)

chunked_docs = text_spliter.split_documents(documents)

vectorestore.add_documents(chunked_docs)