
from langchain_community.vectorstores import FAISS

from src.embeddings import get_embeddings


def load_vector_store():

    # Load embedding model
    embeddings = get_embeddings()

    # Load existing FAISS database
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


def get_retriever():

    # Load vector database
    vector_store = load_vector_store()

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    return retriever


if __name__ == "__main__":

    retriever = get_retriever()

    question = "What is Python?"

    # Retrieve relevant documents
    results = retriever.invoke(question)

    print("Question:", question)

    print("\nRetrieved Documents:\n")

    for i, doc in enumerate(results):

        print(f"Document {i + 1}:")

        print(doc.page_content)

        print("\nSource:", doc.metadata.get("source"))

        print("Page:", doc.metadata.get("page"))

        print("-" * 50)
