import streamlit as st
from gpt4all import GPT4All

# Path to the downloaded model
PATH = "C:/llm/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/"
model_path = f"{PATH}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Load the GPT4All model 
gpt_model = GPT4All(model_name=model_path, device="cpu", allow_download=False)

# Streamlit app setup
st.title("LOCAL GPT")
st.write("""🚨 AI Info Alert! 🚨

Think of me like your enthusiastic but slightly unreliable GPS:
* I might confidently suggest turning left into a lake
* Always double-check my directions (aka information)

Proceed with a pinch of skepticism! 🕵️‍♀️🤖
""")

# User input
user_input = st.text_area("Enter your question here:")

if st.button("Get Answer"):
    if user_input:
        with gpt_model.chat_session():
            response = gpt_model.generate(user_input, max_tokens=1024)
            st.write("### Answer:")
            st.write(response)
    else:
        st.write("Please enter a question.")
