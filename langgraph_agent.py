# langgraph_agent.py — Your first LangGraph pipeline
# This rebuilds your ReAct agent using LangGraph's graph structure

import os
from typing import TypedDict, Annotated
from groq import Groq
from tavily import TavilyClient
from langgraph.graph import StateGraph, END
import math
import json

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# ── Step 1: Define State ─────────────────────────────────────
# State is the shared memory that flows between every node
# Every node reads from it and writes back to it

class AgentState(TypedDict):
    question: str           # the user's original question
    messages: list          # full conversation history
    tool_result: str        # result from the last tool call
    final_answer: str       # the agent's final answer
    steps: int              # how many steps taken so far

# ── Step 2: Define Tools ─────────────────────────────────────
def search_web(query: str) -> str:
    try:
        results = tavily_client.search(query, max_results=3)
        output = ""
        for r in results["results"]:
            output += f"- {r['title']}: {r['content'][:300]}\n\n"
        return output
    except Exception as e:
        return f"Search failed: {str(e)}"

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
    "- search_web(query): Search for current information\n"
    "- calculator(expression): Compute math accurately\n"
    "- final_answer(answer): Call when you have the answer\n\n"
    "Respond ONLY with valid JSON:\n"
    '{"thought": "your reasoning", "tool": "tool_name", "input": "tool input"}\n\n'
    "Rules:\n"
    "- ONLY return JSON, nothing else\n"
    "- Use calculator for ANY maths\n"
    "- Use search_web for anything after 2023\n"
    "- For subjective questions, search once then call final_answer\n"
    "- Never fabricate facts — if unsure say so\n"
    "- Always call final_answer when done"
)

# ── Step 3: Define Nodes ─────────────────────────────────────
# Each node is a function that takes state and returns updated state

def reason_node(state: AgentState) -> AgentState:
    """
    The brain of the agent.
    Looks at the current state and decides what to do next.
    Returns a tool to call or a final answer.
    """
    print(f"\n[REASON NODE] Step {state['steps'] + 1}")

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=state["messages"],
        temperature=0.1,
        max_tokens=500
    )

    raw = response.choices[0].message.content.strip()
    print(f"[REASON NODE] LLM response: {raw[:100]}...")

    # Extract JSON even if LLM adds extra text
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end != 0:
        raw = raw[start:end]

    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        # If JSON fails, treat as final answer
        return {
            **state,
            "final_answer": raw,
            "steps": state["steps"] + 1
        }

    thought = action.get("thought", "")
    tool = action.get("tool", "")
    tool_input = action.get("input", "")

    print(f"[REASON NODE] Thought: {thought}")
    print(f"[REASON NODE] Tool: {tool} | Input: {tool_input}")

    # Update messages with this reasoning step
    updated_messages = state["messages"] + [
        {"role": "assistant", "content": raw}
    ]

    # If final answer — store it
    if tool == "final_answer":
        return {
            **state,
            "messages": updated_messages,
            "final_answer": tool_input,
            "steps": state["steps"] + 1
        }

    # Otherwise store which tool to run next
    return {
        **state,
        "messages": updated_messages,
        "tool_result": f"PENDING:{tool}:{tool_input}",
        "steps": state["steps"] + 1
    }

def tool_node(state: AgentState) -> AgentState:
    """
    The hands of the agent.
    Runs whatever tool the reason_node decided to use.
    Feeds the result back into messages so reason_node can see it.
    """
    print(f"\n[TOOL NODE]")

    # Parse which tool to run
    pending = state["tool_result"]
    _, tool_name, tool_input = pending.split(":", 2)

    print(f"[TOOL NODE] Running: {tool_name}({tool_input})")

    # Run the tool
    if tool_name in TOOLS:
        result = TOOLS[tool_name](tool_input)
    else:
        result = f"Unknown tool: {tool_name}"

    print(f"[TOOL NODE] Result: {result[:100]}...")

    # Add result to messages so reason_node sees it next iteration
    updated_messages = state["messages"] + [
        {
            "role": "user",
            "content": f"Tool result: {result}\n\nContinue. Respond with only JSON."
        }
    ]

    return {
        **state,
        "messages": updated_messages,
        "tool_result": result
    }

# ── Step 4: Define Routing Logic ─────────────────────────────
# This function decides which node to go to after reason_node

def should_continue(state: AgentState) -> str:
    """
    Router function — decides next step after reasoning.
    Returns the name of the next node to go to.
    """
    # If we have a final answer — we're done
    if state.get("final_answer"):
        print("[ROUTER] → END")
        return "end"

    # If max steps reached — stop
    if state["steps"] >= 7:
        print("[ROUTER] → END (max steps)")
        return "end"

    # Otherwise run the tool
    print("[ROUTER] → tool_node")
    return "tool"

# ── Step 5: Build the Graph ──────────────────────────────────
def build_graph():
    """
    Connects all nodes and edges into a runnable graph.
    This is the LangGraph equivalent of your while loop.
    """

    # Create the graph with our state definition
    graph = StateGraph(AgentState)

    # Add nodes — each node is a function
    graph.add_node("reason", reason_node)
    graph.add_node("tool", tool_node)

    # Set entry point — where the graph starts
    graph.set_entry_point("reason")

    # Add conditional edges from reason_node
    # After reasoning, go to tool_node OR end
    graph.add_conditional_edges(
        "reason",           # from this node
        should_continue,    # call this function to decide
        {
            "tool": "tool", # if function returns "tool" → go to tool_node
            "end": END      # if function returns "end" → stop
        }
    )

    # After tool_node always go back to reason_node
    graph.add_edge("tool", "reason")

    # Compile and return the runnable graph
    return graph.compile()

# ── Step 6: Run the Agent ────────────────────────────────────
def run_agent(question: str) -> str:
    """Run the LangGraph agent on a question."""

    # Build the graph
    app = build_graph()

    # Set initial state
    initial_state = AgentState(
        question=question,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        tool_result="",
        final_answer="",
        steps=0
    )

    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"{'='*50}")

    # Run the graph
    final_state = app.invoke(initial_state)

    return final_state.get("final_answer", "Could not find an answer.")

# ── Step 7: Test it ──────────────────────────────────────────
if __name__ == "__main__":
    print("\nTest 1: Maths question")
    answer = run_agent("What is 1847 multiplied by 293?")
    print(f"\nFinal Answer: {answer}")

    print("\nTest 2: Current events")
    answer = run_agent("Who is the current Prime Minister of India?")
    print(f"\nFinal Answer: {answer}")

# ── Two Agent Pipeline ───────────────────────────────────────

class PipelineState(TypedDict):
    question: str
    research: str      # output from research agent
    final_report: str  # output from analyst agent

def research_agent_node(state: PipelineState) -> PipelineState:
    """
    Agent 1: Researcher
    Searches the web for current information first,
    then combines with LLM knowledge for a full answer.
    """
    print("\n[RESEARCH AGENT] Searching web...")

    # Step 1: Search the web first
    try:
        search_results = search_web(state["question"])
        print(f"[RESEARCH AGENT] Web results found.")
    except Exception as e:
        search_results = f"Web search failed: {str(e)}"
        print(f"[RESEARCH AGENT] Web search failed, using LLM knowledge only.")

    # Step 2: Ask LLM to synthesise web results into research findings
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research specialist. "
                    "You have been given web search results on a topic. "
                    "Your job is to extract and present the key factual findings "
                    "from these results. Be specific — include names, dates, numbers. "
                    "Do not add information not present in the search results. "
                    "If the search results are insufficient, say so clearly."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {state['question']}\n\n"
                    f"Web search results:\n{search_results}\n\n"
                    "Extract the key factual findings from these results."
                )
            }
        ],
        max_tokens=800,
        temperature=0.1
    )

    research = response.choices[0].message.content
    print(f"[RESEARCH AGENT] Research complete: {research[:150]}...")

    return {
        **state,
        "research": research
    }

def analyst_agent_node(state: PipelineState) -> PipelineState:
    """
    Agent 2: Analyst
    Takes raw research and produces a structured, insightful report.
    """
    print("\n[ANALYST AGENT] Analysing...")

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert analyst. "
                    "You receive raw research and turn it into a clear, "
                    "structured report with these sections:\n"
                    "1. Key Finding\n"
                    "2. Background and Context\n"
                    "3. Key Facts\n"
                    "4. Summary\n\n"
                    "Be specific, use numbers and dates where available. "
                    "Write in a professional tone."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question asked: {state['question']}\n\n"
                    f"Raw research:\n{state['research']}\n\n"
                    "Write a structured report based on this research."
                )
            }
        ],
        max_tokens=1000,
        temperature=0.3
    )

    report = response.choices[0].message.content
    print(f"[ANALYST AGENT] Report ready.")

    return {
        **state,
        "final_report": report
    }

def build_pipeline():
    """Build the two agent pipeline graph."""

    graph = StateGraph(PipelineState)

    # Add both agents as nodes
    graph.add_node("researcher", research_agent_node)
    graph.add_node("analyst", analyst_agent_node)

    # Set flow: start → researcher → analyst → end
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", END)

    return graph.compile()

def run_pipeline(question: str) -> str:
    """Run the two agent pipeline."""

    app = build_pipeline()

    initial_state = PipelineState(
        question=question,
        research="",
        final_report=""
    )

    print(f"\n{'='*50}")
    print(f"Pipeline Question: {question}")
    print(f"{'='*50}")

    final_state = app.invoke(initial_state)
    return final_state["final_report"]

# ── Test the pipeline ────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: Single ReAct agent
    print("\n--- TEST 1: Single ReAct Agent ---")
    answer = run_agent("What is 1847 multiplied by 293?")
    print(f"\nFinal Answer: {answer}")

    # Test 2: Two agent pipeline
    print("\n--- TEST 2: Two Agent Pipeline ---")
    report = run_pipeline("Who is the Vice President of India and what is his background?")
    print(f"\nFinal Report:\n{report}")