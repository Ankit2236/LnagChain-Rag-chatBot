
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path):
    """
    Load PDF document using LangChain.
    """

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents


if __name__ == "__main__":

    file_path = "documents/Python_Notes.pdf"

    documents = load_pdf(file_path)

    print("PDF Loaded Successfully!")

    print("Total Pages:", len(documents))

    print("\nFirst Page Content:\n")

    print(documents[0].page_content[:1000])

    print("\nMetadata:\n")

    print(documents[0].metadata)
