
from langchain_community.vectorstores import FAISS

from src.embeddings import get_embeddings
from src.document_loader import load_pdf
from src.text_splitter import split_documents


def create_vector_store(chunks):

    # Step 1: Get embedding model
    embeddings = get_embeddings()

    # Step 2: Create FAISS vector database
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Step 3: Save database locally
    vector_store.save_local("faiss_index")

    return vector_store


if __name__ == "__main__":

    # Load PDF
    documents = load_pdf("documents/Python_Notes.pdf")

    # Split PDF
    chunks = split_documents(documents)

    print("Total Chunks:", len(chunks))

    # Create vector database
    vector_store = create_vector_store(chunks)

    print("\nFAISS Vector Database Created Successfully!")

    # Search test
    results = vector_store.similarity_search(
        "What is Python?",
        k=2
    )

    print("\nSearch Results:\n")

    for result in results:

        print("Content:")
        print(result.page_content)

        print("\nMetadata:")
        print(result.metadata)

        print("-" * 50)
