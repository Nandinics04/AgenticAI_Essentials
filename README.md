# Agentic LangGraph

Four pieces of work:

1. A step-by-step LangGraph chatbot in `1-BasicChatBot/basicChatBot.ipynb`. Each section replaces the graph with a richer one: a single LLM node, tool calling, a ReAct loop, checkpointed memory, streaming, and a human-in-the-loop interrupt.
2. An MCP client in `MCP/` that loads tools from a local math server and a local weather server, then answers with a LangChain agent.
3. A tool-calling agent in `2-Debugging/` that runs in LangGraph Studio, with LangSmith tracing turned on.
4. Two multi-agent graphs in `3-Agents/multiaiagents.ipynb`: a researcher-writer pipeline, then a supervisor that routes work to a researcher, an analyst, and a writer.

The larger model is Groq's `openai/gpt-oss-120b`, used by the chatbot notebook and the MCP client. The debugging agent and the multi-agent notebook use the smaller Groq model `openai/gpt-oss-20b`, which is the current stand-in for the retired `llama3-8b-8192` and `llama-3.1-8b-instant` models.

## Project layout

```
AgenticLangraph/
├── 1-BasicChatBot/
│   └── basicChatBot.ipynb    # chatbot notebook
├── 2-Debugging/
│   ├── debugging.ipynb       # tool-calling agent notebook
│   ├── agents.py             # graph exported for LangGraph Studio
│   └── langgraph.json        # Studio config
├── 3-Agents/
│   └── multiaiagents.ipynb   # researcher/writer and supervisor graphs
├── MCP/
│   ├── mathserver.py         # add and multiply over stdio
│   ├── weather.py            # get_weather over streamable HTTP
│   └── client.py             # agent that calls both servers
├── pyproject.toml            # Python 3.13+ and package dependencies
├── requirements.txt          # pip-style dependency list
├── .env                      # API keys (not committed)
└── .gitignore
```

## Setup

Python 3.13 or newer. From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
LANGCHAIN_API_KEY=your_langsmith_key
```

`LANGCHAIN_API_KEY` is used by `2-Debugging/agents.py` so Studio runs show up in the LangSmith project named `debugging`. The notebooks and `MCP/client.py` call `load_dotenv()`, so those keys are picked up before the model and tools are created. Open the notebooks and select the `.venv` kernel.

Run Python files with that same environment. A plain `python` command can point at the global interpreter, which does not have these packages and fails with `No module named 'mcp'`. From the repo root:

```powershell
.\.venv\Scripts\Activate.ps1
```

The prompt then shows `(.venv)`.

Dependencies: `langgraph`, `langgraph-cli[inmem]`, `langchain`, `langchain-groq`, `langchain-tavily`, `langchain-mcp-adapters`, `langchain-openai`, `mcp`, `langsmith`, `python-dotenv`, and `ipykernel`.

`datetime` is part of the Python standard library. The import is `from datetime import datetime`. Nothing extra is installed for it.

## Shared state

Every graph in the notebook uses the same state shape:

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

`messages` is the conversation. `add_messages` is the reducer: each node returns new messages, and LangGraph appends them to the list. A normal list assignment would overwrite history.

`graph.invoke` and `graph.stream` take a state dict, because the compiled graph expects `State`:

```python
graph.invoke({"messages": "Hi"})
```

`"messages"` is the state field. `"Hi"` is the user turn. `add_messages` turns that string into a `HumanMessage`. A bare string such as `graph.invoke("Hi")` does not match the state schema.

A node reads `state["messages"]` and writes the model reply back under the same key:

```python
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
```

The list item is the `AIMessage` itself. Returning `{"messages": {"messages": [AIMessage]}}` makes `add_messages` treat that inner dict as a chat message. It then fails with `Message dict must contain 'role' and 'content' keys`, because the dict has neither `role` nor `type`.

## 1. Basic chatbot

Graph: `START -> llmchatbot -> END`.

`llmchatbot` sends the message list to the model and appends the reply. Two ways of running it are in the notebook:

- `graph.invoke({"messages": "Hi"})` waits until the graph finishes and returns the full state. The latest reply is `response["messages"][-1].content`.
- `graph.stream({"messages": "Hi, how are you?"})` yields one update per node as that node finishes. The default mode is `"updates"`, so each event looks like `{"llmchatbot": {"messages": [AIMessage]}}`. With one node, the loop prints a single reply. With several nodes, a line appears after each node.

The graph is drawn with `graph.get_graph().draw_mermaid_png()`.

## 2. Chatbot with tools

Two tools are bound to the model:

| Tool | What it does |
| --- | --- |
| `TavilySearch(max_length=2)` | Web search. Used for questions about current events, such as recent AI news. |
| `multiply(a: int, b: int) -> int` | Returns `a * b`. |

`llm.bind_tools(tools)` sends each tool's name, description, and argument schema to the model. `bind_tools` expects a flat list of tools (a LangChain tool, a function, a Pydantic model, or a dict schema). A nested list, which happens if a cell like `tools = [tools, multiply]` is run twice, raises `Unsupported function`.

`multiply` uses a Google-style docstring whose `Args:` names match the parameters (`a` and `b`). Writing `a(int)` in the docstring makes LangChain look for a parameter named `a(int)` and raises `Arg a(int) in docstring not found in function signature`. A type in parentheses is stripped only when there is a space: `a (int)`. The `: int` hints already supply the types.

Graph:

```
START -> tool_calling_llm --tool call--> tools -> END
                         --no tool call--> END
```

`tool_calling_llm` calls `llm_with_tools`. The model either writes a normal reply or fills `tool_calls` on the `AIMessage`. `tools_condition` does not ask the model again. It reads the last message:

- `tool_calls` is non-empty: route to the `tools` node.
- `tool_calls` is empty: route to `END`.

`ToolNode` runs the named tool with the arguments the model supplied and appends a `ToolMessage`. In this version the `tools` node goes to `END`, so the tool result is stored and the model does not get another turn to summarize it.

Tried in the notebook:

- `"What is the recent ai news?"` routes to Tavily search.
- `"What is 2 multiplied by 3 and then multipled with 10?"` routes to `multiply`. The first graph stops after that tool result.

## 3. ReAct agent

Same nodes, with one edge changed: `tools` goes back to `tool_calling_llm`.

```
START -> tool_calling_llm --tool call--> tools
                         ^                 |
                         |-----------------|
                         --no tool call--> END
```

After a tool result is appended, the model runs again. It can call another tool or answer in text. The loop ends when the latest assistant message has no tool call.

Tried with: `"Give me the recent ai news and then multiply 5 by 10"`. The model can search, then multiply, then answer.

## 4. Memory

Without a checkpointer, each `invoke` starts from the messages you pass in. A later question such as `"Hey, What is my Name?"` has no record of an earlier introduction.

`MemorySaver` stores checkpoints in process memory. The graph is compiled with `graph.compile(checkpointer=memory)`, and every call passes the same thread:

```python
config = {"configurable": {"thread_id": 1}}
graph.invoke({"messages": "Hi my name is Nandini Amaravadi"}, config=config)
graph.invoke({"messages": "Hey, do you know my name?"}, config=config)
```

The second call resumes that thread, so the model still has the name from the first turn. A different `thread_id` is a separate conversation.

## 5. Streaming

A small graph, `START -> superbot -> END`, is compiled with `MemorySaver` so streamed turns stay on one thread. `superbot` calls the model with no tools.

`graph.stream` supports two modes used in the notebook:

| Mode | What each chunk contains |
| --- | --- |
| `"updates"` | Only the state change from the node that just ran, keyed by node name. |
| `"values"` | The full state after that node, including every message so far. |

Both calls in the notebook use `thread_id` `1`, so the football message is appended to the same thread as the name introduction.

`stream` is synchronous. `astream` is the async counterpart.

## 6. Human-in-the-loop

The last graph pauses a tool so a person can answer before the agent continues.

Tools:

- `TavilySearch`
- `human_assistant`, decorated with `@tool`. Its description is `"Request assistance from a human"`. The function calls `interrupt({"query": query})` and later returns `human_response["data"]`.

Graph: `START -> chatbot`, then `tools_condition` to `tools` or `END`, and `tools -> chatbot`. Compiled with the same `MemorySaver`.

`interrupt` needs a checkpointer. When the model calls `human_assistant`, execution stops inside the tool and the graph surfaces the query. The notebook streams that pause with `stream_mode="values"`.

The run is resumed with the same `thread_id` and a `Command`:

```python
human_response = (
    "We, the experts are here to help We'd recommend you to check out LangGraph to build your agent."
    "It's much more reliable and extensible than autonomous agents."
)
graph.stream(Command(resume={"data": human_response}), config, stream_mode="values")
```

`interrupt` returns that payload, `human_assistant` returns the `"data"` string as the tool result, and `chatbot` runs again with that result in the message list.

Prompt used: `"I need some expert guidance and assistance for building AI Agent. Could you request assistance for me?"`

## How a tool call is chosen

`bind_tools` attaches the tool schemas to the model. On each model node, the model sees the conversation plus those schemas and returns one `AIMessage`. A question about current news tends to produce a `tavily_search` tool call. A multiplication question tends to produce a `multiply` tool call. A greeting tends to produce text and an empty `tool_calls` list. `tools_condition` only inspects that list.

## 7. MCP client

MCP (Model Context Protocol) serves tools from separate processes. `MCP/client.py` connects to two servers, turns their tools into LangChain tools, and passes those tools to an agent.

| File | Server | Transport | Tools |
| --- | --- | --- | --- |
| `mathserver.py` | `FastMCP("Math")` | `stdio` | `add(a, b)`, `multiply(a, b)` |
| `weather.py` | `FastMCP("Weather")` | `streamable-http` on `http://localhost:8000/mcp` | `get_weather(location)` returns `"It's raining in California"` |

The math server is started by the client. Its config uses `"command": "python"` and `"args": ["mathserver.py"]` with `"transport": "stdio"`, so the client speaks to that process over stdin and stdout. Run `client.py` from the `MCP` folder so `mathserver.py` is found.

The weather server is not started by the client. Start it first, in its own terminal, with the virtual environment active:

```powershell
cd MCP
python weather.py
```

Leave that process running. The client then connects with `"transport": "streamable-http"` and `"url": "http://localhost:8000/mcp"`.

`client.py` builds the agent like this:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

tools = await client.get_tools()
model = ChatGroq(model="openai/gpt-oss-120b")
agent = create_agent(model=model, tools=tools)
```

`MultiServerMCPClient` is exported from `langchain_mcp_adapters.client`, not from the top-level `langchain_mcp_adapters` package. `create_agent` lives in `langchain.agents`. `langchain.agents` and `langchain_core.agents` do not provide `create_react_agent`.

The client is async. `get_tools()` and the agent call are awaited. The agent method is `ainvoke`. `invoke` returns a dict immediately, and `await` on that dict raises `TypeError: object dict can't be used in 'await' expression`.

Prompts in `client.py`:

- `"What is (3 + 5) x 10?"` uses the math tools and prints `Math Response:` (`80`).
- `"What is the weather in California?"` uses `get_weather` and prints `Weather Response:`.

`qwen-qwq-32b` is decommissioned on Groq and returns HTTP 400 `model_decommissioned`. This client uses `openai/gpt-oss-120b` instead.

## 8. Debugging and LangGraph Studio

`2-Debugging/debugging.ipynb` and `2-Debugging/agents.py` build the same tool-calling loop used earlier:

```
START -> tool_calling_llm --tool call--> tools
                          ^                 |
                          |-----------------|
                          --no tool call--> END
```

The only tool is `add(a, b)`. The model is loaded with an explicit provider, because `openai/gpt-oss-20b` does not tell `init_chat_model` which service to call:

```python
llm = init_chat_model(model="openai/gpt-oss-20b", model_provider="groq")
```

`agents.py` also sets `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_PROJECT=debugging`.

LangGraph Studio reads `langgraph.json` from the current directory. From the repo root, `langgraph dev` fails with `Path 'langgraph.json' does not exist` because that file is in `2-Debugging/`. Start Studio from that folder:

```powershell
cd 2-Debugging
langgraph dev
```

`langgraph.json` registers one graph:

```json
{
  "dependencies": ["."],
  "graphs": {
    "tool_agent": "./agents.py:tool_agent"
  },
  "env": "../.env"
}
```

`tool_agent` is the compiled graph returned by `make_tool_call()`. A prompt such as `"What is 2 plus 2?"` often gets a direct answer and an empty `tool_calls` list, so `add` never runs. `tools_condition` only routes to `tools` when that list is non-empty. Forcing `tool_choice="add"` on every model call loops forever here, because `tools` goes back to the model. Force it only on the first call, then let the following call answer from the `ToolMessage`.

## 9. Multi-agent graphs

`3-Agents/multiaiagents.ipynb` uses `ChatGroq(model_name="openai/gpt-oss-20b")`. Web search uses `from langchain_tavily import TavilySearch`. The older `langchain_community.tools.tavily_search.TavilySearchResults` import still runs, but it warns that `langchain-community` is no longer maintained.

`create_agent` is imported from `langchain.agents`. There is no `langgraph.agents` module.

### Researcher and writer

State is `AgentState(MessagesState)` plus `next_agent`. Tools are `search_web` (Tavily, `max_results=2`) and `write_summary`.

```
researcher -> writer -> END
```

`research_agent` binds `search_web` and sets `next_agent` to `"writer"`. `writer_agent` summarizes the conversation and sets `next_agent` to `"end"`. `execute_tools` is defined in the notebook, and this compiled graph does not include it as a node, so a tool call on the researcher message is not executed before the writer runs.

Tried with: `"Reseach about the agentic AI in business"`. The researcher message requests `search_web`, then the writer returns a summary of agentic AI in business.

### Supervisor

`SupervisorState` extends `MessagesState` with `next_agent`, `research_data`, `analysis`, `final_report`, `task_complete`, and `current_task`.

Nodes: `supervisor`, `researcher`, `analyst`, `writer`. The entry point is `supervisor`. Every node uses `add_conditional_edges` with `router`. `add_edge` only accepts a start node and an end node, so passing `router` and a path map raises `StateGraph.add_edge() takes 3 positional arguments but 4 were given`.

`router` reads `next_agent`. It returns `END` when the value is `"end"` or `task_complete` is true. Otherwise it returns `supervisor`, `researcher`, `analyst`, or `writer`.

The supervisor prompt is a normal string inside `ChatPromptTemplate`. An `f"""..."""` string evaluates `{has_research}` immediately, before `supervisor_agent` creates that variable, and raises `NameError: name 'has_research' is not defined`. The template fills `{has_research}`, `{has_analysis}`, `{has_report}`, and `{task}` when `chain.invoke` runs.

Tried with: `"Research about the latest trends in AI"`. The supervisor printed `researcher`, then `analyst`, then `writer`. The result state has `research_data`, `analysis`, `final_report`, and `task_complete=True`.
