# Import the required packages
import streamlit as st
import os
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_community import embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()


# Used for Langsmith Tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


# Create vector embeddings and store them in chroma vector store
if "vector" not in st.session_state:
    st.session_state.loader = PyPDFLoader("P19-1496.pdf")
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_doc_splits=st.session_state.text_splitter.split_documents(st.session_state.docs[:70])
    
    st.session_state.vectors = Chroma.from_documents(
    documents=st.session_state.final_doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
    )


# Show display title and use llama3 model
st.title("ChatOllama llama3 Demo")
model = ChatOllama(model="llama3")


# Define prompt template
prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


# Perform chaining and retrieval
full_document_chain = create_stuff_documents_chain(model, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, full_document_chain)

prompt=st.text_input("Enter your prompt")


# Display the response
if prompt:
    response=retrieval_chain.invoke({"input":prompt})
    st.write(response['answer'])

