
import streamlit as st
from src.rag_chain import create_rag_chain

st.set_page_config(
    page_title="LangChain Document Chatbot",
    page_icon="📚",
    layout="centered"
)

st.title("📚 LangChain Document Chatbot")

st.write(
    "Ask questions about your uploaded PDF knowledge base."
)

# Create RAG chain once and reuse it
@st.cache_resource
def load_rag_chain():
    return create_rag_chain()


rag_chain = load_rag_chain()

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
question = st.chat_input("Ask a question about your PDF...")

if question:
    # Display user question
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                response = rag_chain.invoke(question)
                st.markdown(response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

            except Exception as e:
                st.error(f"Something went wrong: {e}")
