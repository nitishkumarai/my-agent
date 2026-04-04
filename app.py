import os
import math
import json
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
    "- Call final_answer when you are confident you have the answer\n"
    "- Keep going until you call final_answer"
)

def extract_json(text: str):
    """Extract JSON object from text even if LLM adds extra text around it."""
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
    max_steps = 5

    for step in range(max_steps):
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1
        )

        raw = response.choices[0].message.content.strip()

        # Extract JSON even if LLM wraps it in text or code blocks
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        raw = extract_json(raw)

        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            # Could not parse — return whatever the LLM said
            return raw, steps

        thought = action.get("thought", "")
        tool = action.get("tool", "")
        tool_input = action.get("input", "")

        steps.append(f"💭 **Thought:** {thought}")

        # Agent is done
        if tool == "final_answer":
            return tool_input, steps

        # Run a known tool
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


# ── UI ───────────────────────────────────────────────────────
st.title("🤖 Nitish's Chatbot")
st.caption("Powered by Gen AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me anything..."):
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