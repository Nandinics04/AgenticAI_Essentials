# Agentic LangGraph

A step-by-step LangGraph chatbot, built in `1-BasicChatBot/basicChatBot.ipynb`. Each section of the notebook replaces the graph with a richer one: a single LLM node, tool calling, a ReAct loop, checkpointed memory, streaming, and a human-in-the-loop interrupt.

The model is Groq's `openai/gpt-oss-120b`, loaded with `init_chat_model("groq:openai/gpt-oss-120b")`.

## Project layout

```
AgenticLangraph/
├── 1-BasicChatBot/
│   └── basicChatBot.ipynb    # the working notebook
├── pyproject.toml            # Python 3.13+ and package dependencies
├── requirements.txt          # pip-style dependency list
├── .env                      # GROQ_API_KEY and TAVILY_API_KEY (not committed)
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
```

The notebook calls `load_dotenv()`, so those keys are picked up before the model and the search tool are created. Open `1-BasicChatBot/basicChatBot.ipynb` and select the `.venv` kernel.

Dependencies: `langgraph`, `langchain`, `langchain-groq`, `langchain-tavily`, `langsmith`, `python-dotenv`, and `ipykernel`.

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
