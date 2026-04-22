#!/bin/bash
# Source environment variables from .env if present
if [ -f .env ]; then
  echo "Sourcing .env for environment variables..."
  set -o allexport
  source .env
  set +o allexport
fi

# Start FastAPI backend (Uvicorn) in background (old port: 8000)
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!
echo $FASTAPI_PID > .fastapi_pid

# Start Gradio frontend in background (old port: 7860)
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT=8000
python gradio_app.py &
GRADIO_PID=$!
echo $GRADIO_PID > .gradio_pid

# Start Streamlit frontend in background (force port 8501)
streamlit run app.py --server.port 8501 &
STREAMLIT_PID=$!
echo $STREAMLIT_PID > .streamlit_pid

echo
echo "To stop all, run: bash stop_legalsearch_stack.sh"
echo

wait
