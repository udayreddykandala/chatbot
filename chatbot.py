import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "your api key here "

# Display header
st.header("My First Chatbot")

# Sidebar file uploader
with st.sidebar:
    st.title("YOUR DOCUMENTS")
    file = st.file_uploader("Upload a PDF file and shoot the questions", type="pdf")

# Initialize variables
chunks = []
text = ""

# Extract text and create chunks
if file is not None:
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
        if text.strip():
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n"],
                chunk_size=50,
                chunk_overlap=15,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            #st.write(f"Number of chunks created: {len(chunks)}")
        else:
            st.error("The uploaded PDF does not contain readable text.")
    except Exception as e:
        st.error(f"Error during text extraction: {e}")

# Check if chunks are available and create vector store
if chunks:
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        st.error(f"An error occurred while creating the vector store: {e}")
else:
    st.info("Upload a valid PDF file to start processing.")

# User input for questions
user_question = st.text_input("Please type your question")
if user_question:
    match = vector_store.similarity_search(user_question)
    st.write(match)
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        max_tokens=1024,
        model_name = "GPT-4o"
    )
    chain = load_qa_chain(llm,chain_type="stuff")
    chain.run(input_documents = match, question = user_question)
    st.write(chain.get_answer())
