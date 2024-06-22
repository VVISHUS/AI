from dotenv import load_dotenv
load_dotenv()  #loading all thr environment variables securely
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#function to load gemini model and get responses
model=genai.GenerativeModel("gemini-pro")
chat=model.start_chat(history=[])

def get_gemini_response(Q):
    # if input!="":
    #     response=chat.send_message([Q,img],stream=True)
    # else:
    response=chat.send_message(Q,stream=True)
    return response 
#Initializing the streamlit app
st.set_page_config(page_title="Q&A Demo")

st.header("Gemini LLM Application")

#Initialize session state for chat history if it doesn't exist

if 'chat_history' not in st.session_state:
    st.session_state['chat_history']=[] 

input=st.text_input("Input",key="input")
submit=st.button("Ask Your Question...")

if submit and input:
    response=get_gemini_response(input)

    #Add user query and response to session chat history
    st.session_state['chat_history'].append(("You : ",input))
    st.subheader("The Response is")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Bot : ",chunk.text))

    st.subheader("The chat history is")
    for role,text in st.session_state['chat_history']:
        st.write(f"{role}:{text}")
