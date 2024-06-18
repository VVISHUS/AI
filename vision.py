from dotenv import load_dotenv
load_dotenv()  #loading all thr environment variables securely
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#function to load gemini model and get responses
model=genai.GenerativeModel("gemini-pro-vision")
def get_gemini_response(Q,img):
    if input!="":
        response=model.generate_content([Q,img])
    else:
        response=model.generate_content(img)
    return response.text

#Initializing the streamlit app

st.set_page_config(page_title="Personalized Gemini")
st.header("Personalized Gemini LLM appliaction<multimodal>")
input=st.text_input("Input: ",key="input")
image=""
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
if uploaded_file is not None:

    # Read the image from the file
    # image = uploaded_file.read()
    image=Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image",use_column_width=True)

submit=st.button("Ask your Question...")


if submit:
    try:
        response=get_gemini_response(input,image)
        st.subheader("The response is : ")
        st.write(response)
    except Exception as e:
        st.error(f"An error occured: {str(e)}")
