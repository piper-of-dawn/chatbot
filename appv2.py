import time
import streamlit as st
from llama_cpp import Llama
from sys_monitor import SystemResourceMonitor
import GPUtil
# App title
st.set_page_config(page_title="Chatbot")
# Path to the downloaded model
model_path = "C:\llm\lmstudio-community\mathstral-7B-v0.1-GGUF\mathstral-7B-v0.1-Q3_K_L.gguf"
model_path_llama = "C:/llm/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
model_path_mistral = "C:/llm/lmstudio-community/mistral-7b-instruct-v0.2.Q2_K.gguf"
model_path_nemotron = "C:/llm/Nemotron-Mini-4B-Instruct-IQ3_M.gguf"
model_path_codestral = "C:/llm/lmstudio-community/Llama3-DocChat-1.0-8B.Q4_K_S/Llama3-DocChat-1.0-8B.Q4_K_S.gguf"
# Replicate Credentials
monitor = SystemResourceMonitor()
def get_gpu_vram():
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        return {
            'total': gpu.memoryTotal,
            'used': gpu.memoryUsed,
            'free': gpu.memoryFree,
            'percentage': gpu.memoryUtil * 100
        }
    return None


if GPUtil.getGPUs():
    st.write("GPU is available.")
    gpu_details = get_gpu_vram()
    st.write(f"`GPU Memory: {gpu_details['used']:.2f} / {gpu_details['total']:.2f} MB ({gpu_details['percentage']:.2f}%)`")
    n_gpu_layers = 40
else:
    st.write("GPU is not available.")
    n_gpu_layers = 0



# with col6:
#     st.button('1')
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    st.subheader('Models and parameters')
    generate_kwargs = {"use_cache": True, "num_beams": 1,"batch_size": 8,}
    
    selected_model = st.sidebar.selectbox('Choose a model', ['Mathstral-7B', 'Nemotron 4B Q3', 'Llama3 Cerebras'], key='selected_model')
    n_ctx = 2048  # Set the desired context window size
    n_threads = 10
    rope_base_frequency = 100
     # Custom RoPE base frequency
    if selected_model == 'Mathstral-7B':
        llm = Llama(model_path=model_path, quantization="int8", n_ctx=n_ctx,n_threads=n_threads,)
    elif selected_model == 'Mistral 7B Quantized2':
        llm = Llama(model_path=model_path_mistral, quantization="int8", n_ctx=n_ctx,n_threads=n_threads, **generate_kwargs)
    elif selected_model == 'Nemotron 4B Q3':
        llm = Llama(model_path=model_path_nemotron, quantization="int8", n_ctx=n_ctx,n_threads=n_threads, **generate_kwargs)
    elif selected_model == 'Llama3 Cerebras':
        llm = Llama(model_path=model_path_codestral, quantization="fp16", n_ctx=n_ctx,n_threads=n_threads,n_gpu_layers=32,rope_base=rope_base_frequency,**generate_kwargs)
    else:
        llm = Llama(model_path=model_path_llama, n_ctx=n_ctx)

    
    from streamlit_utils import get_sidebar_utils
    st, temperature, top_p, max_length, repeat_penalty = get_sidebar_utils(st)

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



def generate_with_progress(prompt, max_length, **kwargs): 
    response = llm(prompt=prompt, max_tokens=max_length, temperature=temperature, top_p=top_p,repeat_penalty=repeat_penalty ,stream=True,stop=["User: ", "Assistant: "], **kwargs)
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
            dict_messages = st.session_state.messages
            for dict_message in dict_messages:       
                if dict_message["role"] == "user":
                    string_dialogue += "User: " + dict_message["content"] + "\n\n"
                else:
                    string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
            final_prompt = f"{string_dialogue} {prompt} Assistant: "
            st.markdown(f"`The size of context is {len(final_prompt.split(" "))} tokens. The larger the context -> Slower the inference`")
            response = generate_with_progress(final_prompt, max_length)
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
                # sys_resources = monitor.get_system_resources()               
            
                metrics.markdown(           
                    f"`{time_to_first_token:.2f} sec to first token | \t"
                    f"{tokens_per_second:.2f} tokens/sec | \t"
                    f"{total_tokens} tokens | \t"
                    f"{elapsed_time:.2f} run time`"                  
                )
  
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
