import os
import logging
from typing import Annotated, TypedDict, Union
from dotenv import load_dotenv

from google import genai
from google.genai import types

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver # For extreme fallback
from langgraph.checkpoint.sqlite import SqliteSaver # For local dev
import sqlite3
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
# Default to gemini-3.1-pro-preview for subscriber accounts, can be overridden in .env
MODEL_ID = os.getenv("MODEL_ID", "gemini-3.1-pro-preview") 

def call_model(state: State):
    messages = state["messages"]
    logger.info(f"Model called with {len(messages)} messages.")
    
    # Ensure messages are in the format expected by the SDK
    formatted_contents = []
    for msg in messages:
        if isinstance(msg, types.Content):
            formatted_contents.append(msg)
        elif isinstance(msg, dict):
            # Handle dictionary format
            role = msg.get("role", "user")
            parts = []
            for p in msg.get("parts", []):
                if isinstance(p, dict) and "text" in p:
                    parts.append(types.Part(text=p["text"]))
                elif isinstance(p, str):
                    parts.append(types.Part(text=p))
                elif hasattr(p, "text"):
                    parts.append(types.Part(text=p.text))
            formatted_contents.append(types.Content(role=role, parts=parts))
        else:
            # Fallback for other types
            logger.warning(f"Unexpected message type: {type(msg)}")
            formatted_contents.append(types.Content(role="user", parts=[types.Part(text=str(msg))]))

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=formatted_contents,
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
        
        # Extract the candidate's message
        if not response.candidates:
            logger.error("No candidates returned from model.")
            raise ValueError("No candidates returned from model.")
            
        new_message = response.candidates[0].content
        logger.info(f"Model responded with: {new_message}")
        return {"messages": [new_message]}
    except Exception as e:
        logger.error(f"Error in call_model: {e}")
        raise e

def should_continue(state: State):
    last_message = state["messages"][-1]
    # Check if the last message has tool calls
    if hasattr(last_message, "parts") and last_message.parts:
        for part in last_message.parts:
            if hasattr(part, "function_call") and part.function_call:
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
    if DB_URI and "postgresql" in DB_URI:
        try:
            logger.info(f"Attempting to use PostgresSaver with {DB_URI}")
            # Use a short timeout for the initial connection check
            pool = ConnectionPool(conninfo=DB_URI, max_size=20, timeout=5.0)
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            return workflow.compile(checkpointer=checkpointer)
        except Exception as e:
            logger.warning(f"Failed to connect to Postgres: {e}. Falling back to SQLite.")
    
    logger.info("Using SqliteSaver for persistence (Local Development).")
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer)

# For direct script usage
if __name__ == "__main__":
    app = get_agent_app()
    print("Agent initialized with Postgres persistence.")
