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

# ── ReAct Tools ──────────────────────────────────────────────
def search_web(query: str) -> str:
    results = tavily_client.search(query, max_results=3)
    output = ""
    for r in results["results"]:
        output += f"- {r['title']}: {r['content'][:300]}\n\n"
    return output

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
    "- For subjective questions like best, greatest, most popular - search once then call final_answer with your findings\n"
    "- For opinion questions you do not need a definitive answer - summarise what you found and call final_answer\n"
    "- Call final_answer when you have enough information, even if the answer is not 100% certain\n"
    "- Keep going until you call final_answer"
)

def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end != 0:
        return text[start:end]
    return text

def run_react_agent(user_message: str, history: list):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    steps = []
    max_steps = 7  # increased from 5

    for step in range(max_steps):
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1
        )

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

# ── RAG Functions ────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

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

# ── UI ───────────────────────────────────────────────────────
st.title("🤖 Nitish's Chatbot")
st.caption("Powered by Gen AI")

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
                with st.expander("🔍 See reasoning"):
                    for s in reasoning_steps:
                        st.markdown(s)

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