import os
import streamlit as st
from groq import Groq
from tavily import TavilyClient

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

def search_web(query: str) -> str:
    results = tavily_client.search(query, max_results=3)
    output = ""
    for r in results["results"]:
        output += f"- {r['title']}: {r['content'][:300]}\n\n"
    return output

def ask_agent(user_message: str, history: list) -> str:
    
    # Step 1: Decide if a web search is needed
    decision_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant.
                When asked a question, decide if you need to search the web.
                If the question is about recent events, news, sports, prices, or 
                anything that may have changed after 2023, respond with exactly:
                SEARCH: <your search query>
                Otherwise just answer directly."""
            },
            {"role": "user", "content": user_message}
        ]
    )
    
    decision = decision_response.choices[0].message.content

    # Step 2: If search needed, do it and send results back to LLM
    if decision.startswith("SEARCH:"):
        query = decision.replace("SEARCH:", "").strip()
        search_results = search_web(query)
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the search results to answer accurately."
            }
        ] + history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"I searched the web and found:\n\n{search_results}"},
            {"role": "user", "content": "Now answer my question based on those search results."}
        ]
        
        final_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        return f"🔍 *Searched the web for: '{query}'*\n\n" + final_response.choices[0].message.content

    else:
        # No search needed
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer clearly and concisely."}
        ] + history + [
            {"role": "user", "content": user_message}
        ]
        final_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        return final_response.choices[0].message.content

# ── UI ────────────────────────────────────────────────────────
st.title("🤖 Nitish's Chatbot")
st.caption("Enabled by Gen AI + Web Search")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_agent(user_input, st.session_state.messages)
        st.write(response)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})