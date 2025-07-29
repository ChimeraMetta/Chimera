#!/usr/bin/env python3
"""
Demo endpoints to trigger different types of errors for self-healing demonstration
"""

import asyncio
import time
import threading
import random
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks
import psutil

router = APIRouter(prefix="/demo", tags=["Demo Endpoints"])

# Global variables to simulate issues
memory_hogs = []  # List to hold memory-consuming objects
cpu_intensive_tasks = []
fake_connections = []

class MemoryHog:
    """Class to consume memory for memory leak simulation"""
    def __init__(self, size_mb: int = 10):
        # Create a large list to consume memory
        self.data = [0] * (size_mb * 1024 * 256)  # Roughly size_mb MB
        self.metadata = {
            'created_at': time.time(),
            'size_mb': size_mb,
            'id': random.randint(1000, 9999)
        }

def cpu_intensive_operation(duration: int = 5):
    """CPU intensive operation that runs for specified duration"""
    start_time = time.time()
    result = 0
    
    while time.time() - start_time < duration:
        # Perform CPU-intensive calculations
        for i in range(100000):
            result += i ** 2 % 1000
            result = result % 1000000
    
    return result

class FakeConnection:
    """Simulate database/external connections"""
    def __init__(self, connection_id: int):
        self.id = connection_id
        self.created_at = time.time()
        self.status = "active"
        self.query_count = 0
    
    def execute_query(self, query: str):
        """Simulate query execution"""
        time.sleep(random.uniform(0.1, 0.5))  # Simulate network latency
        self.query_count += 1
        return f"Result for query {self.query_count}: {query[:50]}..."

@router.get("/")
@router.get("")
async def demo_overview():
    """Overview of available demo endpoints"""
    return {
        "message": "Chimera Self-Healing Demo Endpoints",
        "available_demos": {
            "memory_leak": "/demo/memory-leak - Gradually consume memory to trigger memory leak detection",
            "cpu_overload": "/demo/cpu-overload - Create CPU intensive tasks to trigger CPU healing",
            "connection_issues": "/demo/connection-issues - Simulate connection pool problems",
            "request_failures": "/demo/request-failures - Generate request handling failures",
            "status": "/demo/status - View current demo system status",
            "reset": "/demo/reset - Reset all demo conditions"
        },
        "current_status": {
            "memory_hogs": len(memory_hogs),
            "cpu_tasks": len(cpu_intensive_tasks),
            "fake_connections": len(fake_connections)
        }
    }

@router.post("/memory-leak")
@router.get("/memory-leak")
async def trigger_memory_leak(size_mb: int = 75, count: int = 5):
    """
    Trigger memory leak by creating memory-consuming objects
    
    Args:
        size_mb: Size of each memory hog in MB
        count: Number of memory hogs to create
    """
    before_memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"[DEMO] Starting memory leak simulation - Current memory: {before_memory:.1f}MB")
    
    # Create memory hogs
    for i in range(count):
        memory_hog = MemoryHog(size_mb)
        memory_hogs.append(memory_hog)
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"[DEMO] Created memory hog {i+1}/{count} ({size_mb}MB) - Total memory: {current_memory:.1f}MB")
        await asyncio.sleep(0.1)  # Small delay between allocations
    
    after_memory = psutil.Process().memory_info().rss / 1024 / 1024
    memory_increase = after_memory - before_memory
    print(f"[DEMO] Memory leak simulation complete - Final memory: {after_memory:.1f}MB (increased by {memory_increase:.1f}MB)")
    
    return {
        "message": f"Created {count} memory hogs of {size_mb}MB each",
        "memory_before_mb": before_memory,
        "memory_after_mb": after_memory,
        "memory_increase_mb": memory_increase,
        "total_memory_hogs": len(memory_hogs),
        "note": "Self-healing should trigger when memory usage exceeds threshold"
    }

@router.post("/cpu-overload")
@router.get("/cpu-overload")
async def trigger_cpu_overload(duration: int = 10, num_threads: int = 4):
    """
    Trigger CPU overload by starting CPU-intensive background tasks
    
    Args:
        duration: Duration in seconds for each CPU task
        num_threads: Number of concurrent CPU-intensive threads
    """
    # Get initial CPU reading with proper interval
    psutil.Process().cpu_percent()  # First call to initialize
    await asyncio.sleep(0.1)  # Wait for measurement interval
    before_cpu = psutil.Process().cpu_percent()
    
    print(f"[DEMO] Starting CPU overload simulation with {num_threads} threads for {duration} seconds")
    
    def cpu_task(task_id: int):
        print(f"[DEMO] CPU task {task_id} started")
        try:
            # More aggressive CPU usage
            start_time = time.time()
            result = 0
            iterations = 0
            
            while time.time() - start_time < duration:
                # Heavy mathematical operations without sleep
                for i in range(1000000):  # Increased from 100000
                    result += i ** 2
                    result = result % 1000000
                    # Add more operations to ensure CPU usage
                    for j in range(10):
                        result = (result * 13 + 7) % 1000000
                iterations += 1
            
            cpu_intensive_tasks.append({
                'task_id': task_id,
                'result': result,
                'iterations': iterations,
                'completed_at': time.time()
            })
            print(f"[DEMO] CPU task {task_id} completed after {iterations} iterations")
        except Exception as e:
            cpu_intensive_tasks.append({
                'task_id': task_id,
                'error': str(e),
                'completed_at': time.time()
            })
            print(f"[DEMO] CPU task {task_id} failed: {e}")
    
    # Start CPU-intensive threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=cpu_task, args=(i,))
        thread.daemon = True
        thread.start()
        threads.append(thread)
        # Stagger thread starts slightly
        await asyncio.sleep(0.01)
    
    # Wait a moment for threads to start consuming CPU
    await asyncio.sleep(0.5)
    after_cpu = psutil.Process().cpu_percent()
    
    # Monitor CPU for a few seconds
    max_cpu = after_cpu
    for _ in range(5):
        await asyncio.sleep(0.5)
        current_cpu = psutil.Process().cpu_percent()
        max_cpu = max(max_cpu, current_cpu)
        print(f"[DEMO] Current CPU usage: {current_cpu:.1f}%")
    
    return {
        "message": f"Started {num_threads} CPU-intensive tasks for {duration} seconds each",
        "cpu_before_percent": before_cpu,
        "cpu_after_percent": after_cpu,
        "max_cpu_observed": max_cpu,
        "estimated_completion": f"{duration} seconds",
        "active_cpu_tasks": len([t for t in threads if t.is_alive()]),
        "note": "Self-healing should trigger when CPU usage exceeds threshold (75%)"
    }

@router.post("/connection-issues")
@router.get("/connection-issues")
async def trigger_connection_issues(connection_count: int = 50):
    """
    Simulate connection pool issues by creating many fake connections
    
    Args:
        connection_count: Number of connections to create
    """
    initial_connections = len(fake_connections)
    
    # Create fake connections
    for i in range(connection_count):
        conn = FakeConnection(initial_connections + i)
        fake_connections.append(conn)
        
        # Simulate some connection activity
        if random.random() < 0.3:  # 30% chance of running a query
            conn.execute_query(f"SELECT * FROM table_{i}")
    
    # Simulate some connection failures
    failed_connections = random.randint(1, min(5, connection_count // 10))
    for i in range(failed_connections):
        if fake_connections:
            fake_connections[-(i+1)].status = "failed"
    
    return {
        "message": f"Created {connection_count} fake connections",
        "total_connections": len(fake_connections),
        "active_connections": len([c for c in fake_connections if c.status == "active"]),
        "failed_connections": len([c for c in fake_connections if c.status == "failed"]),
        "note": "Self-healing should trigger when connection count exceeds threshold"
    }

@router.post("/request-failures")
@router.get("/request-failures")
async def trigger_request_failures(failure_rate: float = 0.7):
    """
    Simulate request handling failures
    
    Args:
        failure_rate: Probability of failure (0.0 to 1.0)
    """
    
    # Simulate different types of request failures
    if random.random() < failure_rate:
        failure_type = random.choice([
            "timeout",
            "database_error", 
            "validation_error",
            "resource_exhaustion",
            "external_service_failure"
        ])
        
        error_messages = {
            "timeout": "Request timeout after 30 seconds",
            "database_error": "Database connection failed - connection pool exhausted",
            "validation_error": "Invalid request parameters provided",
            "resource_exhaustion": "Too many concurrent requests - resource limit exceeded",
            "external_service_failure": "External API service unavailable"
        }
        
        # Add some delay to simulate a slow failing request
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        raise HTTPException(
            status_code=500, 
            detail=f"{failure_type}: {error_messages[failure_type]}"
        )
    
    # If no failure, simulate a slow successful request
    await asyncio.sleep(random.uniform(0.1, 1.0))
    
    return {
        "message": "Request completed successfully (avoided failure)",
        "failure_rate": failure_rate,
        "note": "Self-healing should activate on request failures"
    }

@router.get("/status")
async def demo_status():
    """Get current status of all demo systems"""
    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
    current_cpu = psutil.Process().cpu_percent()
    
    return {
        "system_metrics": {
            "memory_usage_mb": current_memory,
            "cpu_percent": current_cpu,
            "timestamp": time.time()
        },
        "demo_status": {
            "memory_hogs": {
                "count": len(memory_hogs),
                "total_estimated_mb": sum(hog.metadata['size_mb'] for hog in memory_hogs),
                "oldest_created": min([hog.metadata['created_at'] for hog in memory_hogs]) if memory_hogs else None
            },
            "cpu_tasks": {
                "completed_tasks": len(cpu_intensive_tasks),
                "last_completed": max([task.get('completed_at', 0) for task in cpu_intensive_tasks]) if cpu_intensive_tasks else None
            },
            "fake_connections": {
                "total": len(fake_connections),
                "active": len([c for c in fake_connections if c.status == "active"]),
                "failed": len([c for c in fake_connections if c.status == "failed"]),
                "total_queries": sum(c.query_count for c in fake_connections)
            }
        },
        "healing_triggers": {
            "memory_threshold_mb": 400,
            "cpu_threshold_percent": 75,
            "connection_threshold": 80,
            "memory_exceeded": current_memory > 400,
            "cpu_exceeded": current_cpu > 75,
            "connections_exceeded": len(fake_connections) > 80
        }
    }

@router.post("/reset")
@router.get("/reset")
async def reset_demo_conditions():
    """Reset all demo conditions to clean state"""
    global memory_hogs, cpu_intensive_tasks, fake_connections
    
    before_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Clear all demo objects
    memory_count = len(memory_hogs)
    cpu_count = len(cpu_intensive_tasks)
    conn_count = len(fake_connections)
    
    memory_hogs.clear()
    cpu_intensive_tasks.clear()
    fake_connections.clear()
    
    # Force garbage collection to free memory
    import gc
    gc.collect()
    
    after_memory = psutil.Process().memory_info().rss / 1024 / 1024
    memory_freed = before_memory - after_memory
    
    return {
        "message": "All demo conditions reset",
        "cleared": {
            "memory_hogs": memory_count,
            "cpu_tasks": cpu_count,
            "fake_connections": conn_count
        },
        "memory_impact": {
            "before_mb": before_memory,
            "after_mb": after_memory,
            "freed_mb": memory_freed
        }
    }

@router.post("/stress-test")
@router.get("/stress-test")
async def comprehensive_stress_test(background_tasks: BackgroundTasks):
    """
    Run a comprehensive stress test that triggers multiple healing mechanisms
    """
    
    async def stress_sequence():
        """Run stress test sequence in background"""
        try:
            # Phase 1: Memory stress
            await asyncio.sleep(2)
            for i in range(3):
                memory_hogs.append(MemoryHog(30))  # 30MB each
                await asyncio.sleep(1)
            
            # Phase 2: Connection stress
            await asyncio.sleep(3)
            for i in range(60):
                fake_connections.append(FakeConnection(i))
            
            # Phase 3: CPU stress
            await asyncio.sleep(2)
            def cpu_stress():
                cpu_intensive_operation(8)
            
            threads = []
            for i in range(6):
                thread = threading.Thread(target=cpu_stress)
                thread.daemon = True
                thread.start()
                threads.append(thread)
            
        except Exception as e:
            print(f"Stress test error: {e}")
    
    # Start stress test in background
    background_tasks.add_task(stress_sequence)
    
    return {
        "message": "Comprehensive stress test started",
        "phases": [
            "Phase 1 (2s): Memory allocation - 3x30MB objects",
            "Phase 2 (3s): Connection creation - 60 connections", 
            "Phase 3 (2s): CPU stress - 6 intensive threads"
        ],
        "expected_duration": "~15 seconds",
        "monitoring_note": "Watch the dashboard for healing actions during this test"
    }