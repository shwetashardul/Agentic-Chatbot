# LangGraph Agentic AI Chatbot

A modular, extensible agentic AI chatbot built with **LangGraph**, **LangChain**, **Groq LLM**, and **Streamlit**. This project demonstrates how to build stateful, graph-based AI agents with a clean separation of concerns and a production-ready architecture that scales from a simple chatbot to complex multi-tool reasoning agents.

---

## Table of Contents

- [Overview](#overview)
- [Agentic AI Architecture](#agentic-ai-architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Code Modularity](#code-modularity)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Use Cases](#use-cases)
- [Extending the Project](#extending-the-project)
- [Roadmap](#roadmap)

---

## Overview

This application provides an end-to-end implementation of an **agentic AI chatbot** using the LangGraph framework for stateful graph execution. Users interact through a Streamlit web interface, selecting their LLM provider, model, and use case — all without touching code.

Two use cases are currently implemented:

- **Basic Chatbot** — stateful multi-turn conversation with streaming responses
- **Chatbot With Web** — ReAct-style agent that uses Tavily web search to answer questions with real-time information

---

## Agentic AI Architecture

### What Makes This "Agentic"?

Traditional LLM applications are stateless: a user sends a message, the LLM responds, and the exchange ends. Agentic AI systems are different — they maintain **state across steps**, can **invoke tools**, make **conditional decisions**, and execute **multi-step reasoning loops**.

This project uses **LangGraph** to implement agents as directed graphs.

### Basic Chatbot Graph

```
[User Input]
     │
     ▼
[HumanMessage → State]
     │
     ▼
┌─────────────────┐
│  LangGraph      │
│  State Machine  │
│                 │
│  START          │
│    │            │
│    ▼            │
│  [chatbot node] │ ◄── LLM invoked here
│    │            │
│    ▼            │
│  END            │
└─────────────────┘
     │
     ▼
[AIMessage → Streamed to UI]
```

### Chatbot With Web (ReAct Agent) Graph

```
[User Input]
     │
     ▼
[HumanMessage → State]
     │
     ▼
┌──────────────────────────────────────┐
│  LangGraph ReAct State Machine       │
│                                      │
│  START                               │
│    │                                 │
│    ▼                                 │
│  [chatbot node] ──── tool call? ─── YES ──► [tools node]
│       ▲                  │                       │
│       │                  NO                      │
│       │                  │                       │
│       └──────────────────┘      ◄────────────────┘
│                  │                               │
│                  ▼                               │
│                END                               │
└──────────────────────────────────────────────────┘
     │
     ▼
[ToolMessage + AIMessage → Displayed in UI]
```

### Core Agentic Concepts Used

| Concept | Implementation |
|---|---|
| **Stateful Execution** | `StateGraph` with `add_messages` reducer accumulates full conversation history |
| **Graph-Based Control Flow** | LangGraph nodes and edges define the agent's execution path |
| **Tool Use (ReAct Pattern)** | LLM bound to tools via `bind_tools()`; `tools_condition` routes conditionally to `ToolNode` |
| **Streaming Responses** | `.stream()` on compiled graph enables real-time output (Basic Chatbot) |
| **Invoke with Tool Results** | `.invoke()` used for tool-enabled graph to capture full tool + AI message chain |
| **Extensible Tool Use** | `tools/` module defines and registers tools; add new tools in one place |
| **Use Case Dispatch** | `setup_graph()` routes to different graph configurations based on user-selected use case |
| **Message Reducers** | LangGraph's `add_messages` ensures messages are appended (not overwritten) across turns |

### Agent Execution Flow

```
load_langgraph_agenticai_app()
        │
        ├── LoadStreamlitUI.load_streamlit_ui()       # render sidebar controls
        │         └── returns user_controls (LLM, model, API key, use case)
        │
        ├── GroqLLM(user_controls).get_llm_model()    # initialize LLM
        │
        ├── GraphBuilder(llm).setup_graph(usecase)    # build & compile graph
        │         ├── basic_chatbot_build_graph()          [Basic Chatbot]
        │         │     └── StateGraph → chatbot node → compile()
        │         └── chatbot_with_tools_build_graph()     [Chatbot With Web]
        │               └── StateGraph → chatbot + tools nodes
        │                             → tools_condition edge → compile()
        │
        └── DisplayResultStreamlit.display_result_on_ui()
                  ├── Basic Chatbot: graph.stream() → parse → render
                  └── Chatbot With Web: graph.invoke() → ToolMessage + AIMessage → render
```

---

## Features

- **Stateful conversation** — full message history maintained in LangGraph state
- **Streaming responses** — AI output appears in real time (Basic Chatbot)
- **Web search (ReAct agent)** — Chatbot With Web uses Tavily to fetch live information and reason over it
- **Tool call visibility** — tool search results are shown inline in the UI before the final answer
- **Multi-model support** — switch between Groq models (Llama, Qwen, GPT-OSS) from the sidebar
- **Config-driven UI** — use cases, LLM options, and model lists are defined in an INI config file
- **Clean output** — `<think>` reasoning tags are automatically stripped from model responses
- **Modular architecture** — each concern (LLM, state, graph, nodes, tools, UI) is isolated in its own module
- **Extensible design** — add new agents, tools, or use cases with minimal changes

---

## Project Structure

```
Agentic-Chatbot/
├── app.py                                   # Application entry point
├── requirements.txt                         # Python dependencies
├── .gitignore
└── src/
    └── langgraphagenticai/
        ├── main.py                          # App orchestrator
        │
        ├── LLMS/                            # LLM provider integrations
        │   └── groqllm.py                   # Groq LLM wrapper (ChatGroq)
        │
        ├── graph/                           # LangGraph graph definitions
        │   └── graph_builder.py             # Builds & compiles state graphs for each use case
        │
        ├── nodes/                           # Graph node implementations
        │   ├── basic_chatbot_node.py        # Basic chatbot node (direct LLM invoke)
        │   └── chatbot_with_Tool_node.py    # Tool-enabled chatbot node (llm.bind_tools)
        │
        ├── state/                           # LangGraph state schema
        │   └── state.py                     # TypedDict state with add_messages reducer
        │
        ├── tools/                           # Tool definitions
        │   └── search_tool.py              # Tavily web search tool + ToolNode factory
        │
        └── ui/                              # User interface layer
            ├── uiconfigfile.ini             # UI configuration (models, use cases)
            ├── uiconfigfile.py              # INI config loader
            └── streamlitui/
                ├── loadui.py                # Streamlit sidebar & input rendering
                └── display_result.py        # Response streaming & display (both use cases)
```

---

## Code Modularity

The project is organized into **six distinct layers**, each with a single responsibility:

### 1. Entry Point (`app.py`)
Minimal entry point — imports and calls the main orchestrator. Nothing else lives here.

### 2. Orchestrator (`main.py`)
Coordinates the full application lifecycle: load UI → configure LLM → build graph → display results. Does not implement any logic itself.

### 3. LLM Layer (`LLMS/`)
Encapsulates LLM provider configuration. Adding a new provider (e.g., OpenAI) means adding a new file here without touching any other layer.

```python
# groqllm.py
class GroqLLM:
    def get_llm_model(self) -> ChatGroq: ...
```

### 4. State Layer (`state/`)
Defines the data contract for the LangGraph state machine. Isolated so that state schema changes don't cascade across the codebase.

```python
# state.py
class State(TypedDict):
    messages: Annotated[List, add_messages]
```

### 5. Graph, Nodes & Tools (`graph/`, `nodes/`, `tools/`)
The graph builder wires nodes and tools together into an executable agent. Each node is a self-contained processing unit. Tools are defined once and reused across any graph that needs them.

```python
# graph_builder.py
class GraphBuilder:
    def setup_graph(self, usecase: str) -> CompiledGraph: ...
    def basic_chatbot_build_graph(self): ...
    def chatbot_with_tools_build_graph(self): ...

# basic_chatbot_node.py
class BasicChatbotNode:
    def process(self, state: State) -> dict: ...

# chatbot_with_Tool_node.py
class ChatbotWithToolNode:
    def create_chatbot(self, tools) -> Callable: ...  # returns llm.bind_tools node fn

# search_tool.py
def get_tools() -> list: ...           # returns [TavilySearchResults]
def create_tool_node(tools) -> ToolNode: ...
```

### 6. UI Layer (`ui/`)
Fully decoupled from agent logic. The Streamlit UI collects inputs and displays outputs — it has no knowledge of how the graph works internally.

```
ui/
├── uiconfigfile.ini     ← configuration (no code changes needed for new models/use cases)
├── uiconfigfile.py      ← loads INI config
└── streamlitui/
    ├── loadui.py        ← renders sidebar controls
    └── display_result.py ← streams/invokes graph and renders output per use case
```

---

## Tech Stack

| Technology | Role |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful agent graph orchestration |
| [LangChain](https://python.langchain.com/) | LLM abstractions and message types |
| [Groq](https://groq.com/) | Ultra-fast LLM inference (Llama, Qwen, etc.) |
| [Streamlit](https://streamlit.io/) | Web UI framework |
| [Tavily](https://tavily.com/) | Web search tool for the ReAct agent use case |
| [FAISS](https://github.com/facebookresearch/faiss) | Vector store (ready for future RAG use cases) |
| [langchain-openai](https://python.langchain.com/docs/integrations/llms/openai) | OpenAI integration (ready for future use) |

---

## Prerequisites

- Python 3.9 or higher
- A [Groq API key](https://console.groq.com/) (free tier available)
- A [Tavily API key](https://tavily.com/) (free tier available — required for **Chatbot With Web**)
- Git

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Agentic-Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API keys

You can enter API keys directly in the Streamlit sidebar, or set them as environment variables:

```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your_groq_key"
$env:TAVILY_API_KEY="your_tavily_key"

# macOS/Linux
export GROQ_API_KEY=your_groq_key
export TAVILY_API_KEY=your_tavily_key
```

Or create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

> **Note:** `TAVILY_API_KEY` is only required when using the **Chatbot With Web** use case.

---

## Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Usage

1. **Select LLM** — choose your LLM provider from the sidebar (currently: Groq)
2. **Select Model** — pick a model (e.g., `llama-3.1-8b-instant`, `qwen/qwen3-32b`)
3. **Enter API Key** — paste your Groq API key in the sidebar
4. **Select Use Case** — choose `Basic Chatbot` or `Chatbot With Web`
5. **Chat** — type your message and press Enter

---

## Configuration

All UI options are controlled via [src/langgraphagenticai/ui/uiconfigfile.ini](src/langgraphagenticai/ui/uiconfigfile.ini):

```ini
[DEFAULT]
PAGE_TITLE         = LangGraph: Build Stateful Agentic AI graph
LLM_OPTIONS        = Groq
USECASE_OPTIONS    = Basic Chatbot, Chatbot With Web
GROQ_MODEL_OPTIONS = qwen/qwen3-32b, llama-3.1-8b-instant, openai/gpt-oss-20b
```

To add a new model or use case, update this file — no Python changes needed (except wiring the new use case in `graph_builder.py`).

---

## How It Works

### 1. State Management

LangGraph tracks the entire conversation as a list of messages in the agent's **state**. The `add_messages` reducer ensures each new message is appended to the history rather than replacing it.

```python
class State(TypedDict):
    messages: Annotated[List, add_messages]
```

### 2. Graph Compilation

`GraphBuilder` creates a `StateGraph`, registers nodes and edges, and compiles it:

```python
# Basic Chatbot
graph_builder.add_node("chatbot", self.basic_chatbot_node.process)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Chatbot With Web (ReAct)
graph_builder.add_node("chatbot", chatbot_node)       # LLM with bound tools
graph_builder.add_node("tools", tool_node)             # Tavily ToolNode
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)  # route on tool call
graph_builder.add_edge("tools", "chatbot")             # loop back after tool runs
```

### 3. Tool Binding (ReAct Pattern)

The `ChatbotWithToolNode` binds Tavily search to the LLM. When the LLM decides to search the web, it emits a tool call; `tools_condition` detects this and routes to the `ToolNode`, which executes the search and returns results back to the LLM.

```python
llm_with_tools = self.llm.bind_tools(tools)

def chatbot_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
```

### 4. Streaming vs Invoke

| Use Case | Method | Why |
|---|---|---|
| Basic Chatbot | `graph.stream()` | Real-time token streaming to UI |
| Chatbot With Web | `graph.invoke()` | Wait for full tool + reasoning cycle to complete |

### 5. Output Display

Tool results and final AI responses are rendered separately in the UI:

```python
if isinstance(message, ToolMessage):
    st.write("Tool Call Start")
    st.write(message.content)   # raw search results
    st.write("Tool Call End")
elif isinstance(message, AIMessage) and message.content:
    st.write(cleaned_output)    # final answer
```

### 6. Output Cleaning

Chain-of-thought reasoning wrapped in `<think>` tags is stripped before display:

```python
content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
```

---

## Use Cases

### Basic Chatbot

A straightforward stateful chatbot. The LLM responds directly to user messages with no tools. Full conversation history is maintained across turns via LangGraph state.

**Best for:** General Q&A, conversation, reasoning tasks that don't need live data.

### Chatbot With Web

A ReAct-style agent powered by Tavily web search. When the LLM determines it needs current information, it calls the search tool, receives results, and synthesizes a final answer — all in one turn from the user's perspective.

**Best for:** Questions about recent events, live data, or anything requiring up-to-date information.

---

## Extending the Project

### Add a New Use Case

1. Add the use case name to `uiconfigfile.ini` under `USECASE_OPTIONS`
2. Create a new node file in `src/langgraphagenticai/nodes/`
3. Add a new graph builder method in `graph_builder.py`
4. Register the new use case in `setup_graph()`:

```python
def setup_graph(self, usecase: str):
    if usecase == "Basic Chatbot":
        self.basic_chatbot_build_graph()
    elif usecase == "Chatbot With Web":
        self.chatbot_with_tools_build_graph()
    elif usecase == "My New Agent":
        self.my_new_agent_build_graph()
```

5. Add a display branch in `display_result.py` for the new use case

### Add New Tools

1. Define new tools in `src/langgraphagenticai/tools/search_tool.py` (or a new file)
2. Add them to the list returned by `get_tools()`
3. They are automatically available to any graph that calls `get_tools()`

### Add a New LLM Provider

1. Create a new file in `src/langgraphagenticai/LLMS/` (e.g., `openaillm.py`)
2. Add the provider name to `LLM_OPTIONS` in `uiconfigfile.ini`
3. Update `main.py` to initialize the new LLM class based on the selected provider

---

## Roadmap

- [x] Basic stateful chatbot with streaming
- [x] ReAct agent with Tavily web search
- [ ] RAG (Retrieval-Augmented Generation) use case using FAISS
- [ ] OpenAI / Anthropic LLM provider support
- [ ] Persistent conversation history (database-backed)
- [ ] Multi-agent workflows with LangGraph's supervisor pattern
- [ ] Docker containerization

---

## License

This project is open source. Feel free to use and extend it.
