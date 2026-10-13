from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    client=MultiServerMCPClient(
        {
            "math":{
                "command": "python",
                "args":["mathserver.py"],
                "transport": "stdio"
            },
            "weather":{
                "url": "http://localhost:8000/mcp",
                "transport": "streamable-http"
            }
        }
    )
    import os
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    tools=await client.get_tools()
    model=ChatGroq(model="openai/gpt-oss-120b")
    agent=create_agent(
        model=model,
        tools=tools,
    )
    math_response=await agent.ainvoke(
        {"messages":[{"role":"user", "content": "What is (3 + 5) x 10?"}]}
    )
    print("Math Response:", math_response["messages"][-1].content)

    weather_response=await agent.ainvoke(
        {"messages": [{"role":"user", "content": "What is the weather in California?"}]}
    )
    print("Weather Response:", weather_response["messages"][-1].content)

asyncio.run(main())

