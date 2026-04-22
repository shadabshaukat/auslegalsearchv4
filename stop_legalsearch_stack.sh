#!/bin/bash

echo "Stopping AUSLegalSearch v2 services..."

for service in fastapi gradio streamlit; do
    PIDFILE=".${service}_pid"
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "Stopped $service (PID $PID)."
        else
            echo "No running $service process with PID $PID."
        fi
        rm -f "$PIDFILE"
    else
        echo "No $PIDFILE found for $service."
    fi
done

echo "All services stopped."
