import streamlit as st
import requests
import time

BACKEND = "http://127.0.0.1:8000"

st.set_page_config(page_title="Website RAG", layout="wide")
st.title("🌐 Website RAG Chatbot")

# -------------------------
# Sidebar – Index Website
# -------------------------

st.sidebar.header("Index Website")

url = st.sidebar.text_input("Enter Website URL")

if st.sidebar.button("Create Vector Store"):

    if url:

        with st.spinner("Creating vector store..."):
            progress = st.sidebar.progress(0)

            for percent in range(0, 90, 10):
                time.sleep(0.2)
                progress.progress(percent)

            response = requests.post(
                f"{BACKEND}/ingest",
                json={"url": url}
            )

            progress.progress(100)

        if response.status_code == 200:
            st.sidebar.success("Website indexed successfully!")
        else:
            st.sidebar.error("Indexing failed.")

# -------------------------
# Chat Section
# -------------------------

st.header("Chat with Website")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask something about the website...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    with st.spinner("Thinking..."):
        response = requests.post(
            f"{BACKEND}/chat",
            json={"question": user_input}
        )

        if response.status_code == 200:
            answer = response.json()["answer"]
        else:
            answer = "Error occurred."

    st.session_state.messages.append(("assistant", answer))

for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(message)