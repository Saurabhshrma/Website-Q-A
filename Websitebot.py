import streamlit as st
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.memory import ConversationBufferMemory

def process_url(url):
    """Loads and processes data from the given URL."""
    loader = WebBaseLoader(url)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="llama2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def main():
    st.title("Web Q&A Bot")
    
    url = st.text_input("Enter the URL you want to ask questions about:")
    if url:  # Process the URL only if provided
        vectorstore = process_url(url)

        # Initialize chat history and memory
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.text_input("Your question:")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Retrieve relevant information and generate a response
            llm = Ollama(model="llama2")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), memory=st.session_state.memory
            )
            response = qa_chain(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response['result']})
            
            with st.chat_message("assistant"):
                st.markdown(response['result'])



if __name__ == "__main__":
    main()




