# app.py
import streamlit as st
from rag import answer_query  # import your existing function

# Set page title and icon
st.set_page_config(page_title="Local RAG Assistant", page_icon="🤖")

st.title("🤖 Local RAG Assistant")
st.markdown("*Fully offline – runs on your laptop with Foundry Local*")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box for new question
if prompt := st.chat_input("Ask a question about your documents..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer_query(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})