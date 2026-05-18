import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


DOCUMENTS_DIR = "data/documents"
VECTOR_DB_DIR = "data/vector_db"
COLLECTION_NAME = "investment_research_notes"


def load_documents():
    documents = []

    for file_path in Path(DOCUMENTS_DIR).glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        documents.append(
            {
                "id": file_path.stem,
                "text": text,
                "source": file_path.name,
            }
        )

    return documents


def create_vector_store():
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    documents = load_documents()

    for doc in documents:
        existing = collection.get(ids=[doc["id"]])

        if len(existing["ids"]) == 0:
            collection.add(
                ids=[doc["id"]],
                documents=[doc["text"]],
                metadatas=[{"source": doc["source"]}],
            )

    print(f"Vector store created with {len(documents)} documents.")


def query_research_notes(question, n_results=3):
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )

    return results


if __name__ == "__main__":
    create_vector_store()

    sample_question = "What are the key risks for NVIDIA?"
    results = query_research_notes(sample_question)

    print("\nSample Question:")
    print(sample_question)

    print("\nRetrieved Context:")
    for doc in results["documents"][0]:
        print(doc[:500])
        print("-" * 50)