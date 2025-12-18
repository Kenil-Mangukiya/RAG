import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(doc_path):
    """Load all file from the docs directory"""
    print(f"Loading documents from {doc_path}...")

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"The directory {doc_path} does not exist")
    
    loader = DirectoryLoader(
        path= doc_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True}
    )

    documents = loader.load()
    print(f"Documents are : {documents}")

    if len(documents) ==0:
        raise FileNotFoundError(f"No .txt files found in {doc_path}")
    
    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Spliting documents into chunk...")

    text_spliter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )  

    chunks = text_spliter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk{i+1} ---")
            print(f"Source : {chunk.metadata["source"]}")
            print(f"Length : {len(chunk.page_content)} characters")
            print(f"Content : ")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5 } more chunks")
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vectore store"""
    print("Creating embedding and storing in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    print(f"Embedding model is : {embedding_model}")

    print("---Creating vector store---")

    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = persist_directory,
        collection_metadata = {"hnsw:space" : "cosine"}
    )
    print("---Finished creating vector store---")
    print(f"Vector store created and saved to {persist_directory}")

    return vector_store


def main():
    print("main function")
    documents = load_documents(doc_path="docs")
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)

if __name__ == "__main__":
    main()