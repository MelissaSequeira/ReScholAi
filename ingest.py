from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load=PyPDFLoader("repaper.pdf")
docs=load.load()

splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splitdoc=splitter.split_documents(docs)

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectodb=FAISS.from_documents(splitdoc, embedding)
vectodb.save_local("faiss_index")

print("ingestion complete")