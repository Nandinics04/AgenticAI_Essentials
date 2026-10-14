from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv

load_dotenv()

import os
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="debugging"

### USe of LangGaph studio to build a tool calling agent

from langchain.chat_models import init_chat_model
llm=init_chat_model(model="openai/gpt-oss-20b", model_provider="groq")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def make_tool_call():
    from langchain_core.tools import tool

    @tool
    def add(a:int,b:int)->int:
        """Add two numbers"""
        return a+b

    tools=[add]
    tool_node=ToolNode([add])
    llm_with_tools=llm.bind_tools([add])

    from langgraph.prebuilt import tools_condition


    def call_llm_model(state:State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder=StateGraph(State)
    graph_builder.add_node("tool_calling_llm", call_llm_model)
    graph_builder.add_node("tools",ToolNode(tools))

    graph_builder.add_edge(START, "tool_calling_llm")
    graph_builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition
    )
    graph_builder.add_edge("tools", "tool_calling_llm")
    graph=graph_builder.compile()
    return graph


tool_agent=make_tool_call()
        
   
