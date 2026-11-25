
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings


if __name__ == "__main__":

    embeddings = get_embeddings()

    text = "Python is a programming language"

    # Generate embedding
    vector = embeddings.embed_query(text)

    print("Embedding Generated Successfully!")

    print("\nOriginal Text:")
    print(text)

    print("\nEmbedding Vector:")
    print(vector)

    print("\nVector Dimensions:")
    print(len(vector))
