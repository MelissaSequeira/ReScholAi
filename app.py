from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/bge-small-en"
)

# Load FAISS
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# LLM
llm = Ollama(model="mistral")

# Strong grounding prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a strict question-answering system.
Answer ONLY using the context below.
If the answer is not present in the context, say exactly:
"Not found in the document."

Context:
{context}

Question:
{question}

Answer:
"""
)

# Build QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False
)

print("Welcome to the PDF QA system. Type 'exit' to quit.")

while True:
    query = input("Enter your question: ")
    if query.lower() == "exit":
        print("Goodbye!")
        break

    response = qa.run(query)
    print("Answer:", response)
