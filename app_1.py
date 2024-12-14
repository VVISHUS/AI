import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# PDF Processing Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to
    provide all the details, if the answer is not in provided context just say,"Answer is not available in the context", don't provide the wrong answer,
    Also make sure that if the user asks about to develop something from the given research paper or any doc, Implement it at any cost
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

# Nutrition Analysis Functions
def get_gemini_resp(Q, img, prompt):
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([Q, img[0], prompt])
    return response

def input_image_setup(uploaded_file):
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        img_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return img_parts
    else:
        raise FileNotFoundError("Please upload a valid file...")

# Main Application
def main():
    st.set_page_config(page_title="AI Multi-Tool")
    st.title("AI based Document & Nutrtion Analyzer")

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["PDF Chat", "Nutrition Analysis"])

    # PDF Chat Tab
    with tab1:
        st.header("Chat with PDF")
        
        with st.sidebar:
            st.title("PDF Upload")
            pdf_docs = st.file_uploader(
                "Upload your PDF files and Click on the Process & Submit button!",
                type=['pdf'],
                accept_multiple_files=True
            )
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

        user_question = st.text_input("Ask a Question from PDF Files")
        if user_question:
            response = user_input(user_question)
            st.write("Reply: ", response)

    # Nutrition Analysis Tab
    with tab2:
        st.header("AI Nutritionist")
        st.info("Just upload a food item and ask anything...")
        
        input_text = st.text_input("Input Prompt", key="nutrition_input")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            key="nutrition_image"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded food image", use_container_width=True)
            
        nutrition_prompt = """See you are an expert in nutritionist where you need to see the food items from the image and calculate the total calories
        , also provide the macros and the details of every food item with calories intake in below format
        1. Item 1 - no. of calories | protein | fats | all type of fats | carbs
        2. Item 2 - no. of calories | protein | fats | all type of fats | carbs
        ----
        ----
        """
        
        if st.button("Analyze", key="nutrition_button"):
            if uploaded_file:
                try:
                    img_data = input_image_setup(uploaded_file)
                    response = get_gemini_resp(nutrition_prompt, img_data, input_text)
                    st.subheader("Analysis Results:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please upload an image first.")

if __name__ == "__main__":
    main()
