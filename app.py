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

# ── ReAct Agent ──────────────────────────────────────────────
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

# ── Streaming Functions ──────────────────────────────────────
def stream_response(messages: list):
    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=1000,
        stream=True
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

def stream_document_response(question: str, context: str):
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

def stream_analyst_response(question: str, research_input: str, is_stable: bool):
    if is_stable:
        system_content = (
            "You are an expert analyst with deep knowledge across all fields. "
            "Answer the question comprehensively using your knowledge. "
            "Structure your response with these sections:\n"
            "**1. Key Answer**\n"
            "**2. Background and Context**\n"
            "**3. Key Facts**\n"
            "**4. Summary**\n\n"
            "Be specific, use examples, write professionally."
        )
        user_content = (
            f"Question: {question}\n\n"
            f"Background research:\n{research_input}\n\n"
            "Write a comprehensive structured response."
        )
    else:
        system_content = (
            "You are an expert analyst. "
            "You will receive fact-checked research where claims are marked:\n"
            "✓ VERIFIED — use these freely\n"
            "⚠️ UNCERTAIN — mention with caution or omit\n"
            "❌ REMOVE — do not use under any circumstances\n\n"
            "Structure your report with these sections:\n"
            "**1. Key Finding**\n"
            "**2. Background and Context**\n"
            "**3. Key Facts**\n"
            "**4. Summary**\n\n"
            "Only use VERIFIED facts. Write professionally."
        )
        user_content = (
            f"Question: {question}\n\n"
            f"Fact-checked research:\n{research_input}\n\n"
            "Write a structured report using only verified facts."
        )

    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
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

# ── LangGraph Pipeline ────────────────────────────────────────
class PipelineState(TypedDict):
    question: str
    question_type: str
    search_results: str      # ← raw search results kept for fact checker
    research: str
    verified_research: str
    final_report: str

# ── Node 1: Router ───────────────────────────────────────────
def router_node(state: PipelineState) -> PipelineState:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You classify questions into exactly one category. "
                    "Respond with ONLY one word — no explanation.\n\n"
                    "STABLE — ONLY for questions where the answer is "
                    "100% certain to not have changed:\n"
                    "- Pure science and mathematics (speed of light, Pythagoras)\n"
                    "- Ancient history (before 1900)\n"
                    "- Classic literature (author of Hamlet, plot of War and Peace)\n"
                    "- Geography that never changes (capital of France)\n\n"
                    "CURRENT — use for EVERYTHING else, including:\n"
                    "- Any question about a living OR recently living person\n"
                    "- Deaths, births, marriages of any person\n"
                    "- Any event after 1900\n"
                    "- Words like: died, dead, currently, latest, recent, "
                    "now, today, still, alive, passed away\n"
                    "- Sports, politics, entertainment, business\n\n"
                    "MIXED — only if the question clearly needs both stable "
                    "science AND current events.\n\n"
                    "When in doubt — always choose CURRENT."
                )
            },
            {
                "role": "user",
                "content": f"Classify: {state['question']}"
            }
        ],
        max_tokens=5,
        temperature=0.0
    )

    category = response.choices[0].message.content.strip().upper()
    for valid in ["CURRENT", "STABLE", "MIXED"]:
        if valid in category:
            category = valid
            break
    else:
        category = "CURRENT"

    return {**state, "question_type": category}

# ── Node 2: Researcher ───────────────────────────────────────
def research_agent_node(state: PipelineState) -> PipelineState:
    """
    Searches web and extracts facts.
    Critically — saves raw search results into state
    so fact checker can use them directly.
    """
    raw_search = search_web(state["question"])

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research specialist. "
                    "Extract key factual findings from web search results. "
                    "Be specific — include names, dates, numbers. "
                    "Do not add anything not in the search results. "
                    "Present findings as a numbered list."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {state['question']}\n\n"
                    f"Web search results:\n{raw_search}\n\n"
                    "Extract the key factual findings as a numbered list."
                )
            }
        ],
        max_tokens=600,
        temperature=0.1
    )

    return {
        **state,
        "search_results": raw_search,          # ← save raw results
        "research": response.choices[0].message.content
    }

# ── Node 3: Fact Checker ─────────────────────────────────────
def fact_checker_node(state: PipelineState) -> PipelineState:
    """
    Checks each research claim against the ORIGINAL search results.
    Does NOT use training data — only the raw search results.
    This is the critical fix — the fact checker now has a real
    source of truth to check against.
    """
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict fact checker.\n\n"
                    "You will receive:\n"
                    "1. The original web search results\n"
                    "2. Research claims extracted from those results\n\n"
                    "Your job: check each claim ONLY against the search results.\n"
                    "Do NOT use your own training data or knowledge.\n"
                    "Only what appears in the search results counts as evidence.\n\n"
                    "Mark each claim:\n"
                    "✓ VERIFIED — claim is directly supported by the search results\n"
                    "⚠️ UNCERTAIN — claim is not clearly present in the search results\n"
                    "❌ REMOVE — claim contradicts the search results\n\n"
                    "End with a VERDICT section summarising overall reliability.\n\n"
                    "Be strict. If something is not clearly in the search results "
                    "— mark it UNCERTAIN, even if you personally know it to be true."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {state['question']}\n\n"
                    f"━━━ ORIGINAL WEB SEARCH RESULTS ━━━\n"
                    f"{state['search_results']}\n\n"
                    f"━━━ RESEARCH CLAIMS TO VERIFY ━━━\n"
                    f"{state['research']}\n\n"
                    "Check each claim against the search results above ONLY. "
                    "Do not use your own knowledge."
                )
            }
        ],
        max_tokens=700,
        temperature=0.0   # zero temperature — be consistent and strict
    )

    return {**state, "verified_research": response.choices[0].message.content}

# ── Node 4: Stable Knowledge ─────────────────────────────────
def stable_knowledge_node(state: PipelineState) -> PipelineState:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable expert across all academic fields. "
                    "Answer questions about established, stable knowledge. "
                    "Be thorough and specific. Present key facts as a numbered list."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {state['question']}\n\n"
                    "Provide comprehensive factual background as a numbered list."
                )
            }
        ],
        max_tokens=600,
        temperature=0.1
    )
    return {
        **state,
        "search_results": "",
        "research": response.choices[0].message.content,
        "verified_research": ""
    }

# ── Routing Logic ─────────────────────────────────────────────
def route_after_router(state: PipelineState) -> str:
    if state["question_type"] == "STABLE":
        return "stable"
    else:
        return "search"

# ── Build Graph ───────────────────────────────────────────────
def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("router", router_node)
    graph.add_node("researcher", research_agent_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("stable_knowledge", stable_knowledge_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "search": "researcher",
            "stable": "stable_knowledge"
        }
    )

    graph.add_edge("researcher", "fact_checker")
    graph.add_edge("fact_checker", END)
    graph.add_edge("stable_knowledge", END)

    return graph.compile()

def run_pipeline(question: str) -> tuple:
    try:
        app = build_pipeline()
        initial_state = PipelineState(
            question=question,
            question_type="",
            search_results="",
            research="",
            verified_research="",
            final_report=""
        )
        final_state = app.invoke(initial_state)
        return (
            final_state["question_type"],
            final_state["search_results"],
            final_state["research"],
            final_state["verified_research"]
        )
    except Exception as e:
        return "ERROR", "", "", classify_error(e)

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
    Smart multi-agent pipeline with real fact checking
    against original web search results.

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
            with st.spinner("Thinking..."):
                raw_answer, reasoning_steps, stream_messages = run_react_agent(
                    user_input, st.session_state.messages
                )
            if stream_messages:
                full_response = st.write_stream(stream_response(stream_messages))
            else:
                st.markdown(raw_answer)
                full_response = raw_answer

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
        "A smart pipeline that routes questions automatically"
    )

    st.markdown("""
    Works using Agents
""")

    st.divider()

    if "pipeline_history" not in st.session_state:
        st.session_state.pipeline_history = []

    for msg in st.session_state.pipeline_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if pipeline_question := st.chat_input(
        "Ask any research question...", key="pipeline_input"
    ):
        with st.chat_message("user"):
            st.markdown(pipeline_question)

        with st.chat_message("assistant"):

            with st.spinner("🧭 Routing and researching..."):
                question_type, search_results, research, verified_research = run_pipeline(
                    pipeline_question
                )

            # Show which path was taken
            if question_type == "STABLE":
                st.info("📚 **Stable knowledge** — answered from training data.")
            elif question_type == "CURRENT":
                st.info("🔍 **Current events** — web search + fact checked against sources.")
            elif question_type == "MIXED":
                st.info("🔀 **Mixed** — web search + fact checked against sources.")

            # Show intermediate outputs
            if search_results:
                with st.expander("🌐 Original Web Search Results"):
                    st.markdown(search_results)

            if research:
                label = (
                    "📚 Knowledge Node Output"
                    if question_type == "STABLE"
                    else "🔍 Agent 2 — Raw Research Claims"
                )
                with st.expander(label):
                    st.markdown(research)

            if verified_research:
                with st.expander("✅ Agent 3 — Fact Check (vs original sources)"):
                    st.markdown(verified_research)

            # Stream analyst report
            st.markdown("📊 **Analyst writing verified report...**")
            try:
                is_stable = question_type == "STABLE"
                research_input = research if is_stable else verified_research
                full_report = st.write_stream(
                    stream_analyst_response(
                        pipeline_question, research_input, is_stable
                    )
                )
            except Exception as e:
                full_report = classify_error(e)
                st.markdown(full_report)

        st.session_state.pipeline_history.append(
            {"role": "user", "content": pipeline_question}
        )
        st.session_state.pipeline_history.append(
            {"role": "assistant", "content": full_report}
        )