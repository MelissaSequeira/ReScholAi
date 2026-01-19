from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

pdfpath=input("enter pdf path:")

loader=PyPDFLoader(pdfpath)
docs=loader.load()

splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
splitdoc=splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="tinyllama")

vectordb=FAISS.from_documents(splitdoc, embeddings)

llm = Ollama(model="tinyllama")

qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

while True:
    q = input("Ask: ")
    if q.lower() == "exit":
        print("Bye 👋")
        break

    answer = qa.run(q)
    print("Answer:", answer, "\n")