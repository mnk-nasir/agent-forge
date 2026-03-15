import os
import subprocess
import sys

def main():
    print("--- Local Development Runner (No Docker) ---")
    
    # Check for critical dependencies
    try:
        import fastapi
        import langgraph
        import langgraph.checkpoint.sqlite
    except ImportError:
        print("Required dependencies missing. Installing/Updating from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "-r", "requirements.txt"])

    # Ensure .env is loaded (done in agent.py)
    
    print("Starting FastAPI server on http://localhost:8000")
    print("Persistence mode: SQLite (local file 'checkpoints.sqlite')")
    
    # Run uvicorn via python -m to ensure it works even if not on PATH
    try:
        subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"])
    except KeyboardInterrupt:
        print("\nStopping server...")

if __name__ == "__main__":
    main()
