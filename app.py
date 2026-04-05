import os
import math
import json
import streamlit as st
from groq import Groq
from tavily import TavilyClient
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ── Tools ────────────────────────────────────────────────────
def search_web(query: str) -> str:
    try:
        results = tavily_client.search(query, max_results=3)
        output = ""
        for r in results["results"]:
            output += f"- {r['title']}: {r['content'][:300]}\n\n"
        return output
    except Exception as e:
        return f"Web search failed: {str(e)}. Try answering from your own knowledge."

def calculator(expression: str) -> str:
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculator error: {str(e)}"

TOOLS = {
    "search_web": search_web,
    "calculator": calculator,
}

SYSTEM_PROMPT = (
    "You are a helpful AI assistant that reasons step by step.\n\n"
    "You have access to these tools:\n"
    "- search_web(query): Search the internet for current information, news, prices, scores, events\n"
    "- calculator(expression): Compute any math expression accurately\n"
    "- final_answer(answer): Call this when you have enough information to answer the user\n\n"
    "At each step you MUST respond with ONLY valid JSON, no other text before or after it.\n"
    "Use exactly this format:\n"
    '{"thought": "your reasoning", "tool": "tool_name", "input": "tool input"}\n\n'
    "Rules:\n"
    "- Respond with ONLY the JSON object, nothing else\n"
    "- Use calculator for ANY maths - never calculate in your head\n"
    "- Use search_web for anything current or after 2023\n"
    "- For subjective questions like best, greatest, most popular - search once then call final_answer\n"
    "- For opinion questions summarise what you found and call final_answer\n"
    "- Call final_answer when you have enough information\n"
    "- If search results are unclear or insufficient, call final_answer and say you could not find reliable information — do NOT guess or make up an answer\n"
    "- Never fabricate facts, names, numbers, or dates — if unsure say so clearly\n"
    "- Keep going until you call final_answer"
)

def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end != 0:
        return text[start:end]
    return text

def classify_error(e: Exception) -> str:
    """Convert technical exceptions into friendly user messages."""
    error_str = str(e).lower()
    if "429" in str(e) or "rate limit" in error_str or "tokens per day" in error_str:
        return (
            "⚠️ **Daily token limit reached.**\n\n"
            "The free Groq tier allows 100,000 tokens per day. "
            "You've used them all up for today.\n\n"
            "**Options:**\n"
            "- Wait a few hours for your quota to reset\n"
            "- Check your usage at [console.groq.com](https://console.groq.com)\n"
            "- Upgrade to Groq's paid tier for higher limits"
        )
    elif "401" in str(e) or "authentication" in error_str or "api key" in error_str:
        return (
            "🔑 **API key error.**\n\n"
            "Your Groq API key seems to be invalid or missing.\n\n"
            "**Fix:** Check that your `GROQ_API_KEY` environment variable is set correctly."
        )
    elif "connection" in error_str or "network" in error_str or "timeout" in error_str:
        return (
            "🌐 **Connection error.**\n\n"
            "Could not reach the Groq API. This is usually temporary.\n\n"
            "**Fix:** Check your internet connection and try again."
        )
    else:
        return (
            f"❌ **Something went wrong.**\n\n"
            f"Error: {str(e)}\n\n"
            "Please try again or refresh the page."
        )

def run_react_agent(user_message: str, history: list):
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history[-6:]:
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})

        steps = []
        max_steps = 7

        for step in range(max_steps):
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.1
                )
            except Exception as e:
                return classify_error(e), steps

            raw = response.choices[0].message.content.strip()

            if "```" in raw:
                raw = raw.split("```")[1].replace("json", "").strip()
            raw = extract_json(raw)

            try:
                action = json.loads(raw)
            except json.JSONDecodeError:
                return raw, steps

            thought = action.get("thought", "")
            tool = action.get("tool", "")
            tool_input = action.get("input", "")

            steps.append(f"💭 **Thought:** {thought}")

            if tool == "final_answer":
                return tool_input, steps

            if tool in TOOLS:
                steps.append(f"🔧 **Using {tool}:** `{tool_input}`")
                observation = TOOLS[tool](tool_input)
                steps.append(f"👁️ **Result:** {observation[:500]}")
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": f"Tool result: {observation}\n\nContinue. Respond with only JSON."
                })
            else:
                steps.append(f"⚠️ Unknown tool: {tool}")
                break

        return "I was not able to find a confident answer.", steps

    except Exception as e:
        return classify_error(e), []

# ── RAG Functions ────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if not text.strip():
            return None
        return text
    except Exception as e:
        return None

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> list:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def build_rag_index(chunks: list, model: SentenceTransformer):
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve_relevant_chunks(question: str, chunks: list, index, model: SentenceTransformer, top_k: int = 3) -> str:
    question_embedding = model.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, top_k)
    relevant = [chunks[i] for i in indices[0] if i < len(chunks)]
    return "\n\n---\n\n".join(relevant)

def ask_document(question: str, chunks: list, index, model: SentenceTransformer) -> str:
    try:
        context = retrieve_relevant_chunks(question, chunks, index, model)
        context = context[:3000]
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document analysis assistant. "
                        "Answer questions based ONLY on the context provided. "
                        "If the answer is not in the context, say so clearly. "
                        "Be concise and specific. Quote relevant parts when helpful."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context from document:\n{context}\n\nQuestion: {question}"
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return classify_error(e)

# ── UI ───────────────────────────────────────────────────────
st.title("🤖 Nitish's Chatbot")
st.caption("Powered by Gen AI")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ How to use")
    st.markdown("""
    **💬 Chat tab**
    Ask me anything — I can search the web and do calculations.

    **📄 Document Q&A tab**
    Upload a PDF and ask questions about its content.

    ---
    **⚠️ Limits**
    This app uses Groq's free tier — 100,000 tokens per day.
    Heavy usage may hit the daily limit.

    ---
    **🔄 Clear chat**
    """)
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.session_state.doc_history = []
        st.rerun()

tab1, tab2 = st.tabs(["💬 Chat", "📄 Document Q&A"])

# ── Tab 1: Chat ──────────────────────────────────────────────
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask me anything...", key="chat_input"):
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                final_answer, reasoning_steps = run_react_agent(
                    user_input, st.session_state.messages
                )
            st.markdown(final_answer)
            if reasoning_steps:
                with st.expander("🔍 See reasoning and sources"):
                    for s in reasoning_steps:
                        st.markdown(s)
                    # Show disclaimer if web search was used
                    if any("search_web" in s for s in reasoning_steps):
                        st.markdown(
                            "---\n"
                            "⚠️ *This answer is based on web search results. "
                            "Please verify important information from the original sources.*"
                        )

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": final_answer})

# ── Tab 2: Document Q&A ──────────────────────────────────────
with tab2:
    st.subheader("📄 Document Q&A")
    st.write("Upload a PDF and ask questions about any part of it.")

    embedding_model = load_embedding_model()
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_pdf:
        with st.spinner("Reading and indexing document..."):
            doc_text = extract_pdf_text(uploaded_pdf)

        if doc_text is None:
            st.error(
                "❌ Could not read this PDF. "
                "This usually happens with scanned or image-based PDFs. "
                "Try a text-based PDF instead."
            )
        else:
            chunks = chunk_text(doc_text)
            index, _ = build_rag_index(chunks, embedding_model)
            st.success(f"Document indexed — {len(chunks)} chunks ready.")

            if "doc_history" not in st.session_state:
                st.session_state.doc_history = []

            for msg in st.session_state.doc_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if doc_question := st.chat_input("Ask a question about the document...", key="doc_input"):
                with st.chat_message("user"):
                    st.markdown(doc_question)

                with st.chat_message("assistant"):
                    with st.spinner("Searching document..."):
                        answer = ask_document(doc_question, chunks, index, embedding_model)
                    st.markdown(answer)

                st.session_state.doc_history.append({"role": "user", "content": doc_question})
                st.session_state.doc_history.append({"role": "assistant", "content": answer})