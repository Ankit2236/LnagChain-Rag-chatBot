
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.retriever import get_retriever
from src.llm import get_llm


def format_docs(docs):
    formatted_docs = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get(
            "page_label",
            doc.metadata.get("page", "Unknown")
        )

        formatted_docs.append(
            f"Source: {source}, Page: {page}\n{doc.page_content}"
        )

    return "\n\n".join(formatted_docs)


def create_rag_chain():

    retriever = get_retriever()
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI document assistant.

    Answer the user's question in a detailed, clear,
    and beginner-friendly way.

    Instructions:
    1. Explain the topic in detail using simple language.
    2. Provide definitions and explanations.
    3. Include important points and examples from the context.
    4. Use headings and bullet points.
    5. Give a detailed answer, not just 2-3 lines.
    6. Use ONLY the provided context.
    7. Do not invent information.
    8. If information is unavailable, clearly say so.
    9. Mention source PDF name and page number.

    Context:
    {context}

    Question:
    {question}

    Detailed Answer:
    """)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":

    rag_chain = create_rag_chain()

    question = "What is Python used for?"

    response = rag_chain.invoke(question)

    print("\nQuestion:")
    print(question)

    print("\nRAG Answer:")
    print(response)
