import os
import logging
from typing import Annotated, TypedDict, Union
from dotenv import load_dotenv

from google import genai
from google.genai import types

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Define the state of the agent
class State(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]

# Define tools (Example: simple search placeholder or calculation)
def get_weather(city: str) -> str:
    """Useful for getting weather information."""
    return f"The weather in {city} is sunny and 25°C."

tools = [get_weather]
tool_node = ToolNode(tools)

# Initialize the Gemini model via Google GenAI SDK
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_ID = "gemini-2.0-flash" # Updated to a valid model ID for ReAct pattern

def call_model(state: State):
    messages = state["messages"]
    # Convert LangGraph messages to Google GenAI format if necessary
    # For simplicity, assuming messages are compatible or handled by SDK wrapper
    # LangGraph's prebuilt ReAct works well with a model that supports tool calling
    
    # Implementation of a simple ReAct step
    # Note: In a real scenario, you'd use a more robust translation or a LangChain wrapper
    # But here we follow the google-genai SDK as requested.
    
    # This is a simplified ReAct integration
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=messages,
        config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name="get_weather",
                    description="Useful for getting weather information.",
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "city": types.Schema(type="STRING")
                        },
                        required=["city"]
                    )
                )
            ])]
        )
    )
    return {"messages": [response]}

def should_continue(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# Construct the graph
workflow = StateGraph(State)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# Connection pool for PostgresSaver
DB_URI = os.getenv("DATABASE_URL")

def get_agent_app():
    # In production, use a persistent pool
    pool = ConnectionPool(conninfo=DB_URI, max_size=20)
    checkpointer = PostgresSaver(pool)
    # Ensure tables are created
    checkpointer.setup()
    return workflow.compile(checkpointer=checkpointer)

# For direct script usage
if __name__ == "__main__":
    app = get_agent_app()
    print("Agent initialized with Postgres persistence.")
