#!/bin/bash

# SageMaker inference server entrypoint
# This wrapper ensures uvicorn starts correctly regardless of SageMaker arguments

cd /opt/program

exec python -m uvicorn serve:app \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 1
