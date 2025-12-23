#!/usr/bin/env python3
"""
PostgreSQL Query Optimizer Server - Main Entry Point

Run this file to start the MeTTa-powered query optimization demo server.

Usage:
    python pg_optimizer_server.py

Then open http://localhost:8001 in your browser.
"""

import uvicorn
from pg_query_optimizer.optimizer_server import app


def main():
    """Start the optimization server."""
    print("=" * 60)
    print("PostgreSQL Query Optimizer - MeTTa Demo")
    print("=" * 60)
    print()
    print("Starting server at http://localhost:8001")
    print()
    print("Endpoints:")
    print("  GET  /                  - Interactive dashboard")
    print("  GET  /api/demo-queries  - List demo queries")
    print("  GET  /api/analyze/{id}  - Analyze a demo query")
    print("  GET  /api/analyze-all   - Analyze all demo queries")
    print("  GET  /api/metrics       - Get optimization metrics")
    print("  GET  /health            - Health check")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )


if __name__ == "__main__":
    main()
