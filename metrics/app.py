import os
import openai
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import datetime
import time
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Enhanced context precision calculation using cosine similarity
def enhanced_context_precision(retrieved_docs, relevant_docs):
    retrieved_contents = [doc.page_content.strip() for doc in retrieved_docs]
    relevant_contents = [doc.page_content.strip() for doc in relevant_docs]

    vectorizer = TfidfVectorizer().fit_transform(retrieved_contents + relevant_contents)
    vectors = vectorizer.toarray()

    retrieved_vectors = vectors[:len(retrieved_contents)]
    relevant_vectors = vectors[len(retrieved_contents):]

    similarities = cosine_similarity(retrieved_vectors, relevant_vectors)
    
    # Sum the maximum similarity scores for each retrieved document
    true_positive_sum = sum(sim.max() for sim in similarities if sim.max() > 0.3)  # Use threshold
    false_positive_sum = len(retrieved_contents) - true_positive_sum

    precision = true_positive_sum / (true_positive_sum + false_positive_sum) if (true_positive_sum + false_positive_sum) > 0 else 0
    return precision

# Enhanced context recall calculation using cosine similarity
def enhanced_context_recall(retrieved_docs, relevant_docs):
    retrieved_contents = [doc.page_content.strip() for doc in retrieved_docs]
    relevant_contents = [doc.page_content.strip() for doc in relevant_docs]

    vectorizer = TfidfVectorizer().fit_transform(retrieved_contents + relevant_contents)
    vectors = vectorizer.toarray()

    retrieved_vectors = vectors[:len(retrieved_contents)]
    relevant_vectors = vectors[len(retrieved_contents):]

    similarities = cosine_similarity(retrieved_vectors, relevant_vectors)

    # Sum the maximum similarity scores for each relevant document
    true_positive_sum = sum(sim.max() for sim in similarities.T if sim.max() > 0.3)  # Use threshold
    false_negative_sum = len(relevant_contents) - true_positive_sum

    recall = true_positive_sum / (true_positive_sum + false_negative_sum) if (true_positive_sum + false_negative_sum) > 0 else 0
    return recall

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
from langchain.schema import Document

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
def fine_tune_retrieval_model(docs, vectordb):
    embeddings = OpenAIEmbeddings()  # Re-train embeddings with more data or better parameters
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    vectordb.add_documents(docs)
    return vectordb
def improve_text_splitting(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=400)  # Adjust chunk size and overlap
    docs = text_splitter.split_documents(documents)
    return docs
def enhance_prompt_engineering(system_message, chat_history, query):
    custom_prompt = system_message + "\n" + "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history]) + f"\nUser: {query}\nAssistant:"
    return custom_prompt
def use_additional_context(retriever, query, additional_k=5):
    retrieved_docs = retriever.get_relevant_documents(query)[:additional_k]
    return retrieved_docs

# Function to load the database
def load_db(file, chain_type, k=5):
    loader = PyPDFLoader(file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    vectordb.add_documents(docs)

    # Debug: Print documents loaded into the database
    # print(f"Documents loaded into the database: {docs}")

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
        if not self.qa:
            print("Error: QA chain not initialized. Load the database first.")
            return "Error: QA chain not initialized. Load the database first."

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

        # Debug: Print retrieved documents
        print(f"Retrieved documents: {result['source_documents']}")

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
    query = st.text_input("Enter your question")
    if st.button("Generate Response"):
        if query:
            response = cb.convchain(query)
            if response:
                st.markdown(f"**User:** {query}")
                st.markdown(f"**ChatBot:** {response}")
            else:
                st.warning("Please enter a question to get a response.")

    if st.button("Clear Chat History"):
        cb.chat_history = []
        st.success("Chat history cleared!")

# Helper functions
def compute_relevance(doc_content, query):
    # Example relevance computation (to be replaced with actual implementation)
    return len(set(doc_content.split()) & set(query.split())) / len(set(query.split()))

def extract_entities_from_docs(doc_contents):
    # Example entity extraction (to be replaced with actual implementation)
    entities = []
    for content in doc_contents:
        entities.extend(content.split())
    return entities

def compute_noise_impact(doc_content, noisy_inputs):
    # Example noise impact computation (to be replaced with actual implementation)
    noise_impact = len(set(doc_content.split()) & set(noisy_inputs[0].split())) / len(set(doc_content.split()))
    return 1 - noise_impact

def compute_faithfulness(generated_answer, ground_truth):
    # Example faithfulness computation (to be replaced with actual implementation)
    return len(set(generated_answer.split()) & set(ground_truth.split())) / len(set(ground_truth.split()))

def compute_integration(generated_answer, context_docs):
    context_content = " ".join([doc.page_content for doc in context_docs])
    return len(set(generated_answer.split()) & set(context_content.split())) / len(set(generated_answer.split()))

def compute_robustness(generated_answer, counterfactual_query):
    # Example robustness computation (to be replaced with actual implementation)
    return 1 if generated_answer not in counterfactual_query else 0

def compute_rejection(generated_answer, negative_query):
    # Example rejection computation (to be replaced with actual implementation)
    return 1 if generated_answer == "This query is not appropriate for this assistant." else 0

# Metrics Calculation Functions
def calculate_context_precision(retrieved_docs, relevant_docs):
    retrieved_contents = [doc.page_content.strip() for doc in retrieved_docs]
    relevant_contents = [doc.page_content.strip() for doc in relevant_docs]

    true_positive = len(set(retrieved_contents) & set(relevant_contents))
    false_positive = len(set(retrieved_contents) - set(relevant_contents))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    return precision

def calculate_context_recall(retrieved_docs, relevant_docs):
    retrieved_contents = [doc.page_content.strip() for doc in retrieved_docs]
    relevant_contents = [doc.page_content.strip() for doc in relevant_docs]

    true_positive = len(set(retrieved_contents) & set(relevant_contents))
    false_negative = len(set(relevant_contents) - set(retrieved_contents))

    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    return recall

def calculate_context_relevance(retrieved_docs, user_query):
    relevance_scores = [compute_relevance(doc.page_content, user_query) for doc in retrieved_docs]
    average_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    return average_relevance

def calculate_context_entity_recall(retrieved_docs, relevant_entities):
    retrieved_contents = [doc.page_content for doc in retrieved_docs]
    retrieved_entities = extract_entities_from_docs(retrieved_contents)
    true_positive = len(set(retrieved_entities) & set(relevant_entities))
    false_negative = len(set(relevant_entities) - set(retrieved_entities))
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    return recall

def calculate_noise_robustness(retrieved_docs, noisy_inputs):
    robustness_scores = [compute_noise_impact(doc.page_content, noisy_inputs) for doc in retrieved_docs]
    average_robustness = sum(robustness_scores) / len(robustness_scores) if robustness_scores else 0
    return average_robustness

def calculate_faithfulness(generated_answer, ground_truth):
    faithfulness_score = compute_faithfulness(generated_answer, ground_truth)
    return faithfulness_score

def calculate_answer_relevance(generated_answer, user_query):
    relevance_score = compute_relevance(generated_answer, user_query)
    return relevance_score

def calculate_information_integration(generated_answer, context_docs):
    integration_score = compute_integration(generated_answer, context_docs)
    return integration_score

def calculate_counterfactual_robustness(generated_answer, counterfactual_query):
    robustness_score = compute_robustness(generated_answer, counterfactual_query)
    return robustness_score

def calculate_negative_rejection(generated_answer, negative_query):
    rejection_score = compute_rejection(generated_answer, negative_query)
    return rejection_score

def calculate_latency(query_function, query):
    start_time = time.time()
    query_function(query)
    end_time = time.time()
    latency = end_time - start_time
    return latency

# Sample data for testing
# Adjusting sample_relevant_docs to match the structure of the retrieved documents
sample_relevant_docs = [
    Document(page_content="""
    What is the Persona Prompt Pattern?
    The persona prompt pattern is a technique in natural language processing (NLP) where the language model is given specific character traits, backgrounds, or perspectives to adopt while generating responses. This approach helps in creating more tailored and contextually relevant outputs.
    """),
    Document(page_content="""
    7375 Teach a Prompt Pattern
    Ankit Goyal
    June 2024
    1 Comprehensive Exploration of the Persona Prompt Pattern
    1.1 What is persona pattern?
    Definition: The persona prompt pattern is a technique in natural language processing (NLP) where the language model is given specific character traits, backgrounds, or perspectives to adopt while generating responses. This approach helps in creating more tailored and contextually relevant outputs.
    1.2 Core Concepts
    1.2.1 Characterization
    """),
    Document(page_content="""
    \\textbf{Definition:} The persona prompt pattern is a technique in natural language processing (NLP) where the language model is given specific character traits, backgrounds, or perspectives to adopt while generating responses. This approach helps in creating more tailored and contextually relevant
    Page 1
    """)
]
# sample_relevant_docs = [
#     Document(page_content="""
#     7375 Teach a Prompt Pattern
#     Ankit Goyal
#     June 2024
#     1 Comprehensive Exploration of the Persona Prompt Pattern
#     1.1 What is persona pattern?
#     Definition: The persona prompt pattern is a technique in natural language processing (NLP) where the language model is given specific character traits, backgrounds, or perspectives to adopt while generating responses. This approach helps in creating more tailored and contextually relevant outputs.
#     1.2 Core Concepts
#     1.2.1 Characterization
#     """)
# ]



sample_user_query = "What is the persona prompt pattern?"
sample_generated_answer = """
The persona prompt pattern is a technique in natural language processing (NLP) where the language model is given specific character traits, backgrounds, or perspectives to adopt while generating responses. This approach helps in creating more tailored and contextually relevant outputs.
"""
sample_ground_truth = """
The persona prompt pattern is a technique in natural language processing (NLP) where the language model is given specific character traits, backgrounds, or perspectives to adopt while generating responses. This approach helps in creating more tailored and contextually relevant outputs.
"""
sample_noisy_inputs = [
    "random text not related to the document",
    "another irrelevant input"
]
sample_counterfactual_query = "What is the persona pattern?"
sample_negative_query = "Tell me a joke."

# Generate a response and get retrieved documents from the vector database
if cb.qa:
    response = cb.convchain(sample_user_query)
    retrieved_docs = cb.relevant_docs
    # print("(*********")
    # print(retrieved_docs)
    # print("(*********")
else:
    print("Error: QA chain not initialized. Please upload a PDF.")
    retrieved_docs = []

if retrieved_docs:
    # Calculate retrieval metrics
    context_precision = enhanced_context_precision(retrieved_docs, sample_relevant_docs)
    context_recall = enhanced_context_recall(retrieved_docs, sample_relevant_docs)
    context_relevance = calculate_context_relevance(retrieved_docs, sample_user_query)
    context_entity_recall = calculate_context_entity_recall(retrieved_docs, ["persona", "NLP", "language model"])
    noise_robustness = calculate_noise_robustness(retrieved_docs, sample_noisy_inputs)

    # Calculate generation metrics
    faithfulness = calculate_faithfulness(sample_generated_answer, sample_ground_truth)
    answer_relevance = calculate_answer_relevance(sample_generated_answer, sample_user_query)
    information_integration = calculate_information_integration(sample_generated_answer, retrieved_docs)
    counterfactual_robustness = calculate_counterfactual_robustness(sample_generated_answer, sample_counterfactual_query)
    negative_rejection = calculate_negative_rejection(sample_generated_answer, sample_negative_query)
    latency = calculate_latency(lambda q: random.choice(retrieved_docs), sample_user_query)  # Replace with actual query function

    # Print results including the additional context if any
    print(f"Context Precision: {context_precision:.2f}")
    print(f"Context Recall: {context_recall:.2f}")
    print(f"Context Relevance: {context_relevance:.2f}")
    print(f"Context Entity Recall: {context_entity_recall:.2f}")
    print(f"Noise Robustness: {noise_robustness:.2f}")
    print(f"Faithfulness: {faithfulness:.2f}")
    print(f"Answer Relevance: {answer_relevance:.2f}")
    print(f"Information Integration: {information_integration:.2f}")
    print(f"Counterfactual Robustness: {counterfactual_robustness:.2f}")
    print(f"Negative Rejection: {negative_rejection:.2f}")
    print(f"Latency: {latency:.2e} seconds")


    # Improve metrics
    # improved_vectordb = fine_tune_retrieval_model(retrieved_docs, cb.vectordb)
    # improved_docs = improve_text_splitting(retrieved_docs)
    # enhanced_prompt = enhance_prompt_engineering(system_message_content, [("query", "answer")], sample_user_query)

    # Ensure the database is loaded and vectordb is initialized
    # additional_context = use_additional_context(cb.vectordb.as_retriever(), sample_user_query)
else:
    print("No documents retrieved, cannot calculate metrics.")

# # Improve metrics
# if cb.qa:
#     improved_vectordb = fine_tune_retrieval_model(retrieved_docs, cb.vectordb)
#     improved_docs = improve_text_splitting(retrieved_docs)
#     enhanced_prompt = enhance_prompt_engineering(system_message_content, [("query", "answer")], sample_user_query)

#     # Ensure the database is loaded and vectordb is initialized
#     additional_context = use_additional_context(cb.vectordb.as_retriever(), sample_user_query)
# else:
#     additional_context = []  # Handle the case when vectordb is None



# Document the changes and analyze the impact
print("Improvement Methods Applied:")
print("1. Fine-tuned the retrieval model.")
print("2. Improved text splitting parameters.")
print("3. Enhanced prompt engineering.")
print("4. Provided additional context for retrieval.")
