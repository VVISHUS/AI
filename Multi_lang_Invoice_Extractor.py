from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#Selection of Model
def get_models_list():
    models_list = {}
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            model_name = m.name.split('/')[-1]
            # print(model_name,"\n")
           
            models_list[model_name] = m.description
    return models_list

def get_gemini_response(input,image,prompt):
    response=model.generate_content([input,image[0],prompt])    
    return response.text

#This functon is specially for st library so that it can read input data in bytes
def image_processing(uploaded_file):
    if uploaded_file is not None:
        #read the file in bytes
        bytes_data=uploaded_file.getvalue()
        image_parts=[{
            "mime_type": uploaded_file.type,
            "data": bytes_data
        }]
        return image_parts
    else:
        raise FileNotFoundError("No File Uploade3d")

#STREAMLIT Configuration

st.header('Gemini Multi Language Invoice Extractor')

models_list = get_models_list()

st.subheader("Gemini Model Selector : ")
st.info("Since this is a Demo version and only Multimodal generative models are supported, So select only thode models which works with Images.")

model_option = st.selectbox(
    'Choose a Gemini model:',
    list(models_list.keys())
)

st.write(f"You selected: **{model_option}**")
st.write(f"Description: {models_list[model_option]}")

model = genai.GenerativeModel(model_option)

input=st.text_input("Input Prompt: ",key="input")
uploaded_file=st.file_uploader("Choose an image file...",type=["jpeg","png","jpg"])
image=None
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image",use_column_width=True)

submit=st.button("Show Results")
input_prompt="""You are an expert in understanding invoices, We will upload
a image as invoice and you will have to answer any questions based on the uploades image"""

#If submit button is clicked

if submit:
    image_data=image_processing(uploaded_file)
    response=get_gemini_response(input_prompt,image_data,input)
    st.subheader("The Response is : ")
    st.write(response)


    