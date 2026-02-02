import streamlit as st
import requests
import json

st.set_page_config(page_title="CreatorAssistant", page_icon="🤖")

st.title("🤖 CreatorAssistant Chatbot")
st.caption("Fine-tuned LLM for Creator Economy Support")

API_URL = "http://localhost:8000/chat"

# Session state для истории
if "messages" not in st.session_state:
    st.session_state.messages = []

# Показываем историю
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Поле ввода
if prompt := st.chat_input("Ask me anything..."):
    # Добавляем сообщение пользователя
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Получаем ответ от API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"message": prompt, "temperature": 0.7, "max_length": 256}
                )
                response.raise_for_status()
                assistant_response = response.json()["response"]
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Сайдбар
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This chatbot is fine-tuned on creator economy support data using:
    - **Model**: Llama-2-7B
    - **Method**: LoRA (QLoRA)
    - **Framework**: PyTorch + Transformers
    """)
