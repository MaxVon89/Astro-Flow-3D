# Astro-Flow-3D Agent System - Web Server

This repository contains the agent system for the Astro-Flow-3D project, implementing a LangGraph-inspired multi-agent framework for astronomical data processing.

## Web Server

The web server exposes API endpoints for task execution:

### Endpoints:
- `GET /health` - Health check
- `POST /run_task` - Execute any specified task  
- `POST /run_preprocessing` - Execute preprocessing pipeline
- `POST /build_dataset` - Build dataset for MBA model
- `POST /validate_mba` - Validate MBA reconstruction

### Usage Example:
```bash
# Start the web server
python web_server.py

# Test health check
curl http://localhost:8080/health

# Run preprocessing task
curl -X POST http://localhost:8080/run_preprocessing

# Run dataset building
curl -X POST http://localhost:8080/build_dataset
```

## Implementation Details

This implementation provides:
- Structured agent workflow with planning → execution → validation phases
- Memory management for tracking session data
- Safety-first execution with validation
- Support for the 4-day execution plan outlined in the Plans directory
- Integration with existing Astro-Flow-3D data processing pipelines

The system implements the "both-agents" architecture as specified in the project documentation.