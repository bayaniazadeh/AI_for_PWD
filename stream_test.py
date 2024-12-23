import streamlit as st 
import ollama

myModel = "llama3.2:3b"

st.title ("Scoping Review")


def gen_code(questionText):
    response = ollama.chat(model = myModel, messages = [
        {
            'role' : 'user',
            'content' : questionText,
        },

    ] )
    st.info(response['message']['content'])


with st.form("my_form"):
    text = st.text_area(
        "enter question:",
        "Over Here",
    )
    submitted = st.form_submit_button("Submit") 
    if submitted:
        gen_code(text) 