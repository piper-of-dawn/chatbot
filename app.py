import streamlit as st
from llama_cpp import Llama

# Configure Streamlit page
st.set_page_config(
    page_title="LOCAL GPT",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

# Path to the downloaded model
model_path = "C:/llm/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Load the Llama model
llm = Llama(model_path=model_path)

# Sidebar for parameter configuration
# Sidebar for parameter configuration
st.sidebar.header("Model Parameters")

temperature = st.sidebar.slider(
    "Temperature", 
    min_value=0.0, 
    max_value=2.0, 
    value=0.7, 
    step=0.1,
    help="Controls how random or creative the responses are."
)

max_tokens = st.sidebar.number_input(
    "Max Tokens", 
    min_value=1, 
    max_value=1024, 
    value=1024, 
    step=64,
    help="Sets the maximum length of the response."
)

top_p = st.sidebar.slider(
    "Top-p (Nucleus Sampling)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.9, 
    step=0.05,
    help="Keeps responses focused by limiting token choices to the top cumulative probability."
)

frequency_penalty = st.sidebar.slider(
    "Frequency Penalty", 
    min_value=0.0, 
    max_value=2.0, 
    value=0.0, 
    step=0.1,
    help="Discourages repetition of words already used in the response."
)

presence_penalty = st.sidebar.slider(
    "Presence Penalty", 
    min_value=0.0, 
    max_value=2.0, 
    value=0.0, 
    step=0.1,
    help="Encourages the model to bring up new topics or ideas."
)


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
# User input and response generation
user_input = st.text_area(
    "Enter your question here:",
    placeholder="Type your question and press Ctrl+Enter to get the answer...",
    key="user_input",
)




if st.button("Get Answer"):
    if user_input:
        st.write("### Answer:")
        answer = ""
        placeholder = st.empty()

        response = llm(
            user_input,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=["Q:", "\n\n"],
            stream=True,
        )
        
        for line in response:
            if "choices" in line and line["choices"]:
                token = line["choices"][0].get("text", "")
                answer += token
                placeholder.write(answer)
