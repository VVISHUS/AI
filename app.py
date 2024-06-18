from dotenv import load_dotenv
load_dotenv()  #loading all thr environment variables securely
import streamlit as st
import os
import google.generativeai as genai


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#function to load gemini model and get responses
model=genai.GenerativeModel("gemini-pro")
def get_gemini_response(Q):
    response=model.generate_content(Q)
    return response.text

#Initializing the streamlit app

st.set_page_config(page_title="Personalized Gemini")
st.header("Personalized Gemini LLM appliaction")
input=st.text_input("Input: ",key="input")
submit=st.button("Ask your Question...")


#When Submiti clicked

if submit:
    response=get_gemini_response(input)
    st.write(response)