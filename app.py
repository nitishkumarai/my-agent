import os
import streamlit as st
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def ask_agent(user_message: str, history: list) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer clearly and concisely."}
    ] + history + [
        {"role": "user", "content": user_message}
    ]
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content

# ── UI ──────────────────────────────────────────────
st.title("🤖 My AI Agent")
st.caption("Powered by Llama 3.3 70B via Groq")

# Store chat history across messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input at the bottom
if user_input := st.chat_input("Ask me anything..."):
    
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_agent(user_input, st.session_state.messages)
        st.write(response)
    
    # Save to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})
