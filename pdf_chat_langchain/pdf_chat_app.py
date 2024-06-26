import streamlit as st
from PyPDF2 import PdfReader

# LangChain: Framework for developing applications powered by language models
# RecursiveCharacterTextSplitter: Splits text into smaller chunks for processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Standard Python library for OS interface (e.g., environment variables)
import os

# Google Generative AI embeddings for LangChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Google Generative AI main library
import google.generativeai as genai

# FAISS: Library for efficient similarity search and clustering of dense vectors
from langchain_community.vectorstores import FAISS

# Google Generative AI chat model for LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain's question-answering chain
from langchain.chains.question_answering import load_qa_chain

# LangChain's prompt template for creating structured prompts
from langchain.prompts import PromptTemplate

# python-dotenv: Loads environment variables from a .env file
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI with API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#Go through each Pdf file then go through aech pdf page and extract text from there and then return it
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  
        for page in pdf_reader.pages:
            # print(page.extract_text()) 
            text += page.extract_text()  # Concatenate the extracted text to the overall text
    return text 

def get_text_chunks(text):
    # Create a text splitter object with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)  # Split the text into chunks
    return chunks  # Note: This should return 'chunks', not 'text'

def get_vector_store(text_chunks):
    # Create embeddings using Google's Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create a vector store from the text chunks using the embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save the vector store locally

def get_conversational_chain():
    # Define a prompt template for the conversational AI
    prompt_template = """
                    Answer the question as detailed as possible from the provided context, make sure to
                    provide all the details, if the answer is not in provided context just say,"Answer is not available in the context", don't provide the wrong answer
                    Context:\n {context}?\n
                    Question: \n{question}\n

                    Answer:
                    """
    
    # Create a ChatGoogleGenerativeAI model instance
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # Create a PromptTemplate object with the defined template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Create a question-answering chain using the model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    # Load the saved vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Perform a similarity search with the user's question
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()  # Get the conversational chain

    # Generate a response using the chain, documents, and user question
    response = chain(
        {
            "input_documents": docs, "question": user_question
        },
        return_only_outputs=True
    )
    
    print(response)  # Print the response (for debugging)
    st.write("Reply: ", response["output_text"])  # Display the response in the Streamlit app
    
def main():
    st.set_page_config("Chat with Multiple PDFs")  # Set the page config for the Streamlit app
    st.header("Chat with PDF using Gemini")  # Add a header to the main page
    user_question = st.text_input("Ask a Question from PDF Files")  # Create a text input for user questions

    if user_question:
        user_input(user_question)  # Process the user's question if one is provided

    with st.sidebar:
        st.title("Menu: ")  # Add a title to the sidebar
        # Create a file uploader for PDF documents
        pdf_docs = st.file_uploader("Upload your PDF files and Click on the Process & Submit button!", type=['pdf'], accept_multiple_files=True)
        if st.button("Submit & Process"):  # Create a button to process the uploaded PDFs
            with st.spinner("Processing..."):  # Show a spinner while processing
                raw_text = get_pdf_text(pdf_docs)  # Extract text from the PDFs
                text_chunks = get_text_chunks(raw_text)  # Split the text into chunks
                get_vector_store(text_chunks)  # Create and save a vector store
                st.success("Done")  # Show a success message when processing is complete

if __name__ == "__main__":
    main()  # Run the main function when the script is executed