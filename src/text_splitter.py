
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    from src.document_loader import load_pdf

    file_path = "documents/Python_Notes.pdf"

    # Step 1: Load PDF
    documents = load_pdf(file_path)

    print("PDF Loaded Successfully!")

    # Step 2: Split PDF into chunks
    chunks = split_documents(documents)

    print("\nText Splitting Completed!")

    print("Total Pages:", len(documents))

    print("Total Chunks:", len(chunks))

    print("\nFirst Chunk Content:\n")

    print(chunks[0].page_content)

    print("\nFirst Chunk Metadata:\n")

    print(chunks[0].metadata)
