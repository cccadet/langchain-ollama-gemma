from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DocusaurusLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorestore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

# Docsaurus 
loader = DocusaurusLoader(
    "https://python.langchain.com",
    filter_urls=[
        "https://python.langchain.com/docs/expression_language/"
    ],
)

documents = loader.load()

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100
)

chunked_docs = text_spliter.split_documents(documents)

vectorestore.add_documents(chunked_docs)