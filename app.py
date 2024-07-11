import os
import openai
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import datetime

try:
    import chromadb
except ImportError:
    st.error("Could not import chromadb python package. Please install it with `pip install chromadb`.")
    st.stop()

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

# Load environment variables
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# Determine the model name based on the current date
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"

# Ensure chroma_db directory exists
persist_directory = "chroma_db"
os.makedirs(persist_directory, exist_ok=True)

system_message_content = """
You are an educational assistant. Your job is to help undergraduate students understand concepts and definitions by explaining them through short, memorable stories that even a 10-year-old can understand. If a user asks a question unrelated to education, kindly notify them that the chatbot is designed for educational purposes only.
"""

# Function to load the database
def load_db(file, chain_type, k=3):
    loader = PyPDFLoader(file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    vectordb.add_documents(docs)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k})
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0.0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True
    )
    return qa, docs, vectordb

class Chatbot:
    def __init__(self):
        self.chat_history = []
        self.loaded_file = None
        self.qa = None
        self.docs = []
        self.vectordb = None
        self.relevant_docs = []

    def call_load_db(self, file):
        self.loaded_file = file
        self.qa, self.docs, self.vectordb = load_db(file, "stuff", k=3)
        self.chat_history = []
        self.relevant_docs = []

    def clear_db(self):
        if self.vectordb:
            self.vectordb._client.reset()
        self.docs = []
        self.vectordb = None
        self.qa = None
        self.loaded_file = None
        self.chat_history = []
        self.relevant_docs = []

    def convchain(self, query):
        if not query:
            return ""
        # Clear relevant_docs before processing a new query
        self.relevant_docs = []

        # Check if the length of chat history exceeds 16,000 characters
        chat_history_length = sum(len(item[0]) + len(item[1]) for item in self.chat_history)
        if chat_history_length > 16000:
            self.chat_history = []

        # Truncate the chat history to fit within the context length limit
        token_limit = 16000
        truncated_history = []
        current_length = 0

        for q, a in reversed(self.chat_history):
            length = len(q) + len(a)
            if current_length + length <= token_limit:
                truncated_history.insert(0, (q, a))
                current_length += length
            else:
                break

        # Create a custom prompt based on the system message and current chat history
        custom_prompt = system_message_content + "\n" + "\n".join([f"User: {q}\nAssistant: {a}" for q, a in truncated_history]) + f"\nUser: {query}\nAssistant:"

        result = self.qa({"question": query, "chat_history": truncated_history, "custom_prompt": custom_prompt})
        self.chat_history.extend([(query, result["answer"])])
        self.relevant_docs = result["source_documents"]
        return result


    def get_all_vectors(self):
        if self.vectordb:
            return self.vectordb._collection.get()["documents"]
        return []

    def get_vector_count(self):
        if self.vectordb:
            return len(self.vectordb._collection.get()["documents"])
        return 0

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = Chatbot()

cb = st.session_state.chatbot

st.title("Education Chatbot Assistant")

tab1, tab2, tab3 = st.tabs(["Upload PDF", "View Data", "Chatbot"])

with tab1:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        cb.call_load_db("temp.pdf")
        st.success("File loaded successfully!")

with tab2:
    if cb.docs:
        vectors = cb.get_all_vectors()
        if vectors:
            st.write("Some Example Vectors in the vector database:")
            for vector in vectors:
                st.write(vector)
        else:
            st.write("No vectors found in the vector database.")
    else:
        st.write("No documents loaded yet.")
    
    if st.button("Clear Database"):
        cb.clear_db()
        st.success("Chroma database cleared!")
        st.experimental_rerun()

with tab3:
    # if cb.loaded_file:
    #     st.markdown(f"**Loaded File:** {cb.loaded_file}")

    query = st.text_input("Enter your question")
    if st.button("Generate Response"):
        if query:
            response = cb.convchain(query)
            if response:
                st.markdown(f"**User:** {query}")
                st.markdown(f"**ChatBot:** {response['answer']}")
                # st.markdown(f"**Generated Question:** {response['generated_question']}")
                # st.markdown("**Source Documents:**")
                # for doc in cb.relevant_docs:
                    # st.markdown(f"- {doc.page_content}")
            else:
                st.warning("Please enter a question to get a response.")

    if st.button("Clear Chat History"):
        cb.chat_history = []
        st.success("Chat history cleared!")
# 