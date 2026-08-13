import os

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


# =========================
# LOAD ENVIRONMENT VARIABLES
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")


# =========================
# STREAMLIT CONFIGURATION
# =========================

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)


# =========================
# TITLE
# =========================

st.title("🤖 AI Chatbot")

st.write(
    "Chatbot using Streamlit, LangChain, OpenRouter and Conversation Memory."
)


# =========================
# LLM
# =========================

llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
    temperature=0
)


# =========================
# MEMORY
# =========================

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []


# =========================
# DISPLAY PREVIOUS MESSAGES
# =========================

for message in st.session_state.chat_memory:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.write(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):
            st.write(message.content)


# =========================
# USER INPUT
# =========================

user_input = st.chat_input("Type your message...")


# =========================
# CHAT PROCESSING
# =========================

if user_input:

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Add user message to memory
    st.session_state.chat_memory.append(
        HumanMessage(content=user_input)
    )

    # Get complete conversation history
    messages = st.session_state.chat_memory

    # Call LLM
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = llm.invoke(messages)

                answer = response.content

                st.write(answer)

                # Save AI response
                st.session_state.chat_memory.append(
                    AIMessage(content=answer)
                )

            except Exception as e:

                st.error(f"Error: {str(e)}")


# =========================
# CLEAR MEMORY
# =========================

st.sidebar.title("Memory")

if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.chat_memory = []

    st.rerun()
