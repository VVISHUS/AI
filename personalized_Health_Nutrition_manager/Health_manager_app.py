from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_gemini_resp(Q,img,propmt):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([Q,img[0],propmt])
    return response


def input_image_setup(uploaded_file):
    if uploaded_file:
        bytes_data=uploaded_file.getvalue()
        img_parts=[
            {
                "mime_type":uploaded_file.type,
                "data":bytes_data
            }
        ]
        return img_parts
    else:
        raise FileNotFoundError("Please upload a valid file...")


st.set_page_config(page_title="Gemini as a Nutritionist")
st.header("ASK ME ABOUT NUTRITION...")
input=st.text_input("Input Prompt ",key="input")
uploaded_file=st.file_uploader("Choose an image...",type=["jpg","jpeg","png"])
image=""
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption="Uploaded file",use_column_width=True)

submit=st.button("Tell me about the image...")

input_prompt="""See you are an expert in nutritionist where you need to see the food items from the image and calculate the total calories
, also provide the macros and the details of every food item with calories intak in below format

1. Item 1 - no. of calories | protein | fats | all type of fats | carbs
2. Item 2 - no. of calories | protein | fats | all type of fats | carbs
----
----
"""

if submit:
    img_data=input_image_setup(uploaded_file)
    response=get_gemini_resp(input_prompt,img_data,input)
    st.subheader("Your Response : ")
    st.write(response.text)