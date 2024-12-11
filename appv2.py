import time
import streamlit as st
from llama_cpp import Llama
# App title
st.set_page_config(page_title="Chatbot")
# Path to the downloaded model
model_path = "C:\llm\lmstudio-community\mathstral-7B-v0.1-GGUF\mathstral-7B-v0.1-Q3_K_L.gguf"
model_path_llama = "C:/llm/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
model_path_mistral = "C:/llm/lmstudio-community/mistral-7b-instruct-v0.2.Q2_K.gguf"

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a model', ['Mathstral-7B', 'Llama2-8B', 'Mistral 7B Quantized2'], key='selected_model')
    if selected_model == 'Mathstral-7B':
        llm = Llama(model_path=model_path)
    elif selected_model == 'Mistral 7B Quantized2':
        llm = Llama(model_path=model_path_mistral)
    else:
        llm = Llama(model_path=model_path_llama)

    
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=1024, step=8)


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)



def generate_with_progress(prompt, max_tokens, **kwargs): 
    response = llm(prompt=prompt, max_tokens=max_length, temperature=temperature, top_p=top_p, stream=True,stop=["User: "], **kwargs)
    return response

if prompt := st.chat_input(disabled=False):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    with st.chat_message("assistant"):
        first_token_time = None        # Record time for the first token
        total_tokens = 0  
        metrics = st.empty()
        with st.spinner("Thinking..."):            
            for dict_message in st.session_state.messages:
                if dict_message["role"] == "user":
                    string_dialogue += "User: " + dict_message["content"] + "\n\n"
                else:
                    string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
            response = generate_with_progress(f"{string_dialogue} {prompt} Assistant: ", 512)
            placeholder = st.empty()
            full_response = ''
            start_time = time.time()
            for item in response:
                full_response += item["choices"][0]["text"] 
                total_tokens += 1 
                if first_token_time is None:
                    first_token_time = time.time() 
                placeholder.write(full_response)
                elapsed_time = time.time() - start_time
                time_to_first_token = first_token_time - start_time if first_token_time else 0
                tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
                metrics.markdown(           
                    f"`{time_to_first_token:.2f} sec to first token | \t"
                    f"{tokens_per_second:.2f} tokens/sec | \t"
                    f"{total_tokens} tokens | \t"
                    f"{elapsed_time:.2f} run time`"                
                )
  
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
