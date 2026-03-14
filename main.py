import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agent import get_agent_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LangGraph ReAct Agent API")

# Lazy initialization of the agent app
agent_app = None

class ChatRequest(BaseModel):
    thread_id: str
    message: str

@app.on_event("startup")
async def startup_event():
    global agent_app
    try:
        logger.info("Initializing LangGraph agent with Postgres persistence...")
        agent_app = get_agent_app()
        logger.info("Agent initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        # In a real production app, you might want to retry or exit
        raise e

@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected" if agent_app else "initializing"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if not agent_app:
        raise HTTPException(status_code=503, detail="Agent is not initialized yet.")
    
    config = {"configurable": {"thread_id": request.thread_id}}
    
    try:
        # Construct the input for LangGraph
        input_message = {"messages": [{"role": "user", "parts": [request.message]}]}
        
        # Invoke the graph
        logger.info(f"Invoking agent for thread_id: {request.thread_id}")
        result = agent_app.invoke(input_message, config)
        
        # Extract the last message content
        final_message = result["messages"][-1]
        
        # If result is from Google GenAI SDK, we extract the text
        # Adjust based on exact response structure
        if hasattr(final_message, "text"):
            answer = final_message.text
        elif isinstance(final_message, dict) and "content" in final_message:
            answer = final_message["content"]
        else:
            answer = str(final_message)

        return {
            "thread_id": request.thread_id,
            "answer": answer
        }
    except Exception as e:
        logger.error(f"Error during chat invocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
