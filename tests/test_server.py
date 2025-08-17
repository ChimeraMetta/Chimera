#!/usr/bin/env python3
"""
Minimal test server to diagnose connectivity issues
"""
from fastapi import FastAPI
import uvicorn
from datetime import datetime

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Test server root", "time": datetime.now().isoformat()}

@app.get("/test")
async def test():
    return {"status": "Working", "time": datetime.now().isoformat()}

if __name__ == "__main__":
    print("Starting test server on 0.0.0.0:8000")
    print("This is a minimal server for testing connectivity")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")