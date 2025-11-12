import streamlit as st
from langchain_groq import ChatGroq
from langchain_groq import ChatGroq

GROQ_API_KEY = "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"

llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

# Streamlit UI
st.set_page_config(page_title="FarmAI - Crop and Disease Help", layout="centered")
st.title("🌾 FarmAI Assistant")
st.write("Ask about crop care, disease treatment, pest control, and more!")

# Input from the user
query = st.text_input("Type your farming-related question here:")

if st.button("Get Advice"):
    if query:
        with st.spinner("Thinking..."):
            try:
                response = llm.invoke(query)
                st.success("Here’s what I found:")
                st.write(response.content)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
    else:
        st.warning("Please enter a question to get started.")
