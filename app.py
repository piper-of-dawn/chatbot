import streamlit as st
from gpt4all import GPT4All

# Configure Streamlit page
st.set_page_config(
    page_title="LOCAL GPT",
    page_icon="🤖",
    initial_sidebar_state="collapsed"
)

# Customize Streamlit theme
st.markdown("""
<style>

:root {
  --primary-color: #8B4513;  /* Saddle brown */
  --background-color: #F4ECD8;  /* Soft sepia background */
  --text-color: #5D4037;  /* Dark brown text */
  --font-family: 'Georgia', serif;
}

.stApp {
    background-color: var(--background-color);
    color: var(--text-color);
    font-family: var(--font-family);
}

[data-testid="stHeader"] {
    background-color: rgba(244, 236, 216, 0.7);
}

[data-testid="stTextArea"] textarea {
    background-color: #FAEBD7;
    color: var(--text-color);
    font-family: var(--font-family);
}

[data-testid="stButton"] button {
    border: 2px solid var(--primary-color);
    background-color: transparent;
}

[data-testid="stButton"] button:hover {
    transform: scale(1.1);
    color: #FFF8DC;
    font-family: var(--font-family);
}
            
h1,p, ul li, h2, h3, h4, h5 {
    color: var(--primary-color);
    font-family: var(--font-family);
}

            
            .stApp, textarea {
    caret-color: var(--primary-color) !important;
}
           
</style>
             
""", unsafe_allow_html=True)

# Path to the downloaded model
PATH = "C:/llm/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/"
model_path = f"{PATH}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Load the GPT4All model 
gpt_model = GPT4All(model_name=model_path, device="gpu", allow_download=False, verbose=True)

# Streamlit app setup
st.title("LOCAL GPT")
st.write("""🚨 AI Info Alert! 🚨

Think of me like your enthusiastic but slightly unreliable GPS:
* I might confidently suggest turning left into a lake
* Sometimes, I even hallucinate and suggest roads that don’t exist 🚫
* Always double-check my directions (aka information)
* My master hardcoded me to be concise, so I might come off as a bit blunt! 🤷‍♀️
         
Proceed with a pinch of skepticism! 🕵️‍♀️🤖
""")
# Display an image
st.image("meme.webp", caption="AI Humor", width=200)
# User input
user_input = st.text_area("Enter your question here:")

if st.button("Get Answer"):
    if user_input:        
        response = gpt_model.generate(user_input, max_tokens=512, streaming=True)
        st.write("### Answer:")
        st.write_stream(response)
            # placeholder.text(answer)  # Use `text` to ensure it stays on one line