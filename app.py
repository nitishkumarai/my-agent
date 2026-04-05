import os
import math
import json
import streamlit as st
from groq import Groq
from tavily import TavilyClient
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import TypedDict
from langgraph.graph import StateGraph, END
import faiss

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ── Shared Tools ─────────────────────────────────────────────
def search_web(query: str) -> str:
    try:
        results = tavily_client.search(query, max_results=3)
        output = ""
        for r in results["results"]:
            output += f"- {r['title']}: {r['content'][:300]}\n\n"
        return output
    except Exception as e:
        return f"Web search failed: {str(e)}."

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

# ── ReAct System Prompt ──────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful AI assistant that reasons step by step.\n\n"
    "You have access to these tools:\n"
    "- search_web(query): Search the internet for current information\n"
    "- calculator(expression): Compute any math expression accurately\n"
    "- final_answer(answer): Call this when you have enough information\n\n"
    "At each step respond with ONLY valid JSON:\n"
    '{"thought": "your reasoning", "tool": "tool_name", "input": "tool input"}\n\n'
    "Rules:\n"
    "- ONLY return JSON, nothing else\n"
    "- Use calculator for ANY maths\n"
    "- Use search_web for anything current or after 2023\n"
    "- For subjective questions, search once then call final_answer\n"
    "- Never fabricate facts — if unsure say so\n"
    "- Always give rich contextual answers, never just a name or number\n"
    "- Always call final_answer when done"
)

def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end != 0:
        return text[start:end]
    return text

def classify_error(e: Exception) -> str:
    error_str = str(e).lower()
    if "429" in str(e) or "rate limit" in error_str or "tokens per day" in error_str:
        return (
            "⚠️ **Daily token limit reached.**\n\n"
            "The free Groq tier allows 100,000 tokens per day.\n\n"
            "**Options:**\n"
            "- Wait a few hours for your quota to reset\n"
            "- Check usage at [console.groq.com](https://console.groq.com)"
        )
    elif "401" in str(e) or "authentication" in error_str or "api key" in error_str:
        return "🔑 **API key error.** Check that your `GROQ_API_KEY` is set correctly."
    elif "connection" in error_str or "network" in error_str or "timeout" in error_str:
        return "🌐 **Connection error.** Check your internet connection and try again."
    else:
        return f"❌ **Something went wrong.**\n\nError: {str(e)}\n\nPlease try again."

# ── ReAct Agent — returns final answer + steps ───────────────
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
                    temperature=0.3,
                    max_tokens=1000
                )
            except Exception as e:
                return classify_error(e), steps, None

            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1].replace("json", "").strip()
            raw = extract_json(raw)

            try:
                action = json.loads(raw)
            except json.JSONDecodeError:
                return raw, steps, None

            thought = action.get("thought", "")
            tool = action.get("tool", "")
            tool_input = action.get("input", "")

            steps.append(f"💭 **Thought:** {thought}")

            if tool == "final_answer":
                # Return the answer and the messages for streaming
                stream_messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"The answer is: {tool_input}\n\n"
                            "Now rewrite this as a rich, well structured response "
                            "with full context. Use markdown formatting."
                        )
                    }
                ]
                return tool_input, steps, stream_messages

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

        return "I was not able to find a confident answer.", steps, None

    except Exception as e:
        return classify_error(e), [], None

# ── Streaming functions ──────────────────────────────────────
def stream_response(messages: list):
    """Stream a response word by word using Groq's streaming API."""
    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=1000,
        stream=True     # ← this is the key difference
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

def stream_document_response(question: str, context: str):
    """Stream document Q&A response."""
    stream = groq_client.chat.completions.create(
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
        max_tokens=500,
        temperature=0.1,
        stream=True
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

def stream_pipeline_response(question: str, research: str):
    """Stream the analyst agent's final report."""
    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert analyst. "
                    "Turn raw research into a clear structured report with these sections:\n"
                    "**1. Key Finding**\n"
                    "**2. Background and Context**\n"
                    "**3. Key Facts**\n"
                    "**4. Summary**\n\n"
                    "Use specific numbers and dates. Write professionally."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Raw research:\n{research}\n\n"
                    "Write a structured report."
                )
            }
        ],
        max_tokens=800,
        temperature=0.3,
        stream=True
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

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
    except Exception:
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

def retrieve_relevant_chunks(question: str, chunks: list, index, model, top_k: int = 3) -> str:
    question_embedding = model.encode([question]).astype("float32")
    distances, indices = index.search(question_embedding, top_k)
    relevant = [chunks[i] for i in indices[0] if i < len(chunks)]
    return "\n\n---\n\n".join(relevant)

# ── LangGraph Pipeline ───────────────────────────────────────
class PipelineState(TypedDict):
    question: str
    research: str
    final_report: str

def research_agent_node(state: PipelineState) -> PipelineState:
    try:
        search_results = search_web(state["question"])
    except Exception:
        search_results = "Web search unavailable."

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research specialist. "
                    "Extract key factual findings from web search results. "
                    "Be specific — include names, dates, numbers. "
                    "Do not add anything not in the search results."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {state['question']}\n\n"
                    f"Web search results:\n{search_results}\n\n"
                    "Extract the key factual findings."
                )
            }
        ],
        max_tokens=600,
        temperature=0.1
    )
    return {**state, "research": response.choices[0].message.content}

def build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("researcher", research_agent_node)
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", END)
    return graph.compile()

def run_research_only(question: str) -> str:
    """Run only the research agent — analyst streams directly in UI."""
    try:
        app = build_pipeline()
        initial_state = PipelineState(
            question=question,
            research="",
            final_report=""
        )
        final_state = app.invoke(initial_state)
        return final_state["research"]
    except Exception as e:
        return classify_error(e)

# ── UI ───────────────────────────────────────────────────────
st.title("🤖 Nitish's Chatbot")
st.caption("Powered by Gen AI")

with st.sidebar:
    st.header("ℹ️ How to use")
    st.markdown("""
    **💬 Chat**
    Ask anything — web search + calculator included.

    **📄 Document Q&A**
    Upload a PDF and ask questions about it.

    **🔬 LangGraph Pipeline**
    Two AI agents — Researcher + Analyst —
    produce a structured report with streaming output.

    ---
    **⚠️ Token Limit**
    Free tier: 100,000 tokens/day via Groq.
    ---
    """)
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.session_state.doc_history = []
        st.session_state.pipeline_history = []
        st.rerun()

tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Document Q&A", "🔬 LangGraph Pipeline"])

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
            # Step 1: Run ReAct loop (reasoning — no streaming)
            with st.spinner("Thinking..."):
                raw_answer, reasoning_steps, stream_messages = run_react_agent(
                    user_input, st.session_state.messages
                )

            # Step 2: Stream the final answer
            if stream_messages:
                full_response = st.write_stream(
                    stream_response(stream_messages)
                )
            else:
                # Fallback if streaming not available
                st.markdown(raw_answer)
                full_response = raw_answer

            # Show reasoning steps
            if reasoning_steps:
                with st.expander("🔍 See reasoning and sources"):
                    for s in reasoning_steps:
                        st.markdown(s)
                    if any("search_web" in s for s in reasoning_steps):
                        st.markdown(
                            "---\n"
                            "⚠️ *Based on web search results. "
                            "Verify important information from original sources.*"
                        )

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": full_response})

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
                "Try a text-based PDF rather than a scanned one."
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
                    try:
                        context = retrieve_relevant_chunks(
                            doc_question, chunks, index, embedding_model
                        )
                        context = context[:3000]
                        # Stream the document response
                        full_response = st.write_stream(
                            stream_document_response(doc_question, context)
                        )
                    except Exception as e:
                        full_response = classify_error(e)
                        st.markdown(full_response)

                st.session_state.doc_history.append({"role": "user", "content": doc_question})
                st.session_state.doc_history.append({"role": "assistant", "content": full_response})

# ── Tab 3: LangGraph Pipeline ─────────────────────────────────
with tab3:
    st.subheader("🔬 LangGraph Research Pipeline")
    st.write(
        "Two AI agents work in sequence — a **Researcher** searches the web "
        "and gathers current facts, then an **Analyst** streams a structured report."
    )

    st.markdown("""
    Your Question
      ↓
[🔍 Research Agent]  — searches web, extracts key facts
      ↓
[📊 Analyst Agent]   — streams structured report live
      ↓
Structured Report
""")

    st.divider()

    if "pipeline_history" not in st.session_state:
        st.session_state.pipeline_history = []

    for msg in st.session_state.pipeline_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if pipeline_question := st.chat_input(
        "Ask a research question...", key="pipeline_input"
    ):
        with st.chat_message("user"):
            st.markdown(pipeline_question)

        with st.chat_message("assistant"):
            # Step 1: Research agent runs normally
            with st.spinner("🔍 Agent 1 researching..."):
                research = run_research_only(pipeline_question)

            # Step 2: Analyst agent streams its report
            st.markdown("📊 **Agent 2 writing report...**")
            try:
                full_report = st.write_stream(
                    stream_pipeline_response(pipeline_question, research)
                )
            except Exception as e:
                full_report = classify_error(e)
                st.markdown(full_report)

            # Show raw research in expander
            if research and not research.startswith("❌"):
                with st.expander("🔍 See raw research from Agent 1"):
                    st.markdown(research)

        st.session_state.pipeline_history.append(
            {"role": "user", "content": pipeline_question}
        )
        st.session_state.pipeline_history.append(
            {"role": "assistant", "content": full_report}
        )