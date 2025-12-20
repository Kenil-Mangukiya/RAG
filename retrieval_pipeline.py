from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory = persistent_directory,
    embedding_function = embedding_model,
    collection_metadata = {"hnsw:space": "cosine"}
)

query = "What was the vision behind founding SpaceX?"

retriever = db.as_retriever(search_kwargs = {"k": 3})

relevant_docs = retriever.invoke(query)
print(f"User query is : {query}")

print("---Context---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i} : \n{doc.page_content}")

prompt = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in  relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents.if you can't find the answer in the in the documents, say "I don't have enough information to answer that question based on the provided documents
"""

model = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=prompt)
]

result = model.invoke(messages)

print("Generated response : ")
print(f"result is : {result}")
print("Content only : ")
print(result.content)