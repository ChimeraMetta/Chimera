# Chimera FastAPI Self-Healing Server

A FastAPI server with autonomous self-healing capabilities using MeTTa-powered analysis and pattern-based recovery mechanisms.

## Features

✅ **Working FastAPI server with self-healing capabilities:**
- Error detection and classification system
- Pattern-based healing responses 
- Performance monitoring dashboard
- Real-time metrics collection

✅ **Demo automatic recovery from:**
- Memory leaks
- High CPU utilization  
- Connection pool issues
- Request handling failures

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python fastapi_healing_server.py
```

The server will start on `http://localhost:8000`

### 3. View Dashboard
Open your browser to `http://localhost:8000` to see the real-time monitoring dashboard.

## API Endpoints

### Core Endpoints
- `GET /` - Performance monitoring dashboard (HTML)
- `GET /health` - Health check with detailed diagnostics
- `GET /docs` - FastAPI automatic documentation

### Demo Endpoints
- `GET /demo/` - Overview of available demo endpoints
- `POST /demo/memory-leak` - Trigger memory leak for healing demo
- `POST /demo/cpu-overload` - Trigger CPU overload for healing demo  
- `POST /demo/connection-issues` - Trigger connection pool issues
- `POST /demo/request-failures` - Trigger request handling failures (probabilistic)
- `GET /demo/request-failures-immediate` - Immediately trigger request failure healing
- `GET /demo/status` - View current demo system status
- `POST /demo/reset` - Reset all demo conditions
- `POST /demo/stress-test` - Comprehensive stress test

## Self-Healing Capabilities

### 1. Memory Leak Detection & Recovery
**Detection:**
- Monitors memory usage every 5 seconds
- Triggers when memory exceeds 400MB threshold
- Pattern-based error message detection

**Healing Actions:**
- Force garbage collection
- Clear internal caches and metric history
- Reset connection pools
- Log memory freed and recovery actions

**Demo:**
```bash
curl -X POST "http://localhost:8000/demo/memory-leak?size_mb=60&count=5"
```

### 2. CPU Overload Detection & Recovery  
**Detection:**
- Monitors CPU usage continuously
- Triggers when CPU usage exceeds 75%
- Detects processing timeouts and thread exhaustion

**Healing Actions:**
- Reduce background monitoring frequency
- Enable request throttling
- Yield CPU control to other tasks
- Lower processing thresholds

**Demo:**
```bash
curl -X POST "http://localhost:8000/demo/cpu-overload?duration=15&num_threads=6"
```

### 3. Connection Pool Issue Recovery
**Detection:**
- Monitors active connection count
- Triggers when connections exceed 80
- Pattern detection for connection failures

**Healing Actions:**
- Reset circuit breakers for failed endpoints
- Clear and reinitialize connection pools  
- Update connection state tracking
- Log connection recovery actions

**Demo:**
```bash
curl -X POST "http://localhost:8000/demo/connection-issues?connection_count=100"
```

### 4. Request Handling Failure Recovery
**Detection:**
- Monitors request failures via middleware
- Exception handling for various failure types
- Request latency and timeout tracking
- Error rate calculation over sliding window

**Healing Actions:**
- Enable graceful degradation mode
- Reduce request timeout thresholds
- Implement circuit breaker patterns
- Log failure recovery strategies

**Demo (Probabilistic failures):**
```bash
curl -X POST "http://localhost:8000/demo/request-failures?failure_rate=0.8"
```

**Demo (Immediate healing trigger):**
```bash
# Instantly triggers healing without waiting for error threshold
curl -X GET "http://localhost:8000/demo/request-failures-immediate"
```

## Architecture

### Self-Healing Manager
Central component that orchestrates healing responses:
- `SelfHealingManager` - Core healing orchestration
- `ErrorClassifier` - Pattern-based error categorization
- Background monitoring thread for continuous health checks
- Metrics collection and healing action logging

### MeTTa Integration
Leverages existing Chimera components:
- `DynamicMonitor` - Runtime monitoring and MeTTa rule loading
- `AutonomousEvolution` - Evolutionary healing strategies
- Pattern-based reasoning for healing decisions

### Monitoring & Metrics
- Real-time system metrics (memory, CPU, connections)
- Historical metrics with configurable retention
- Healing action tracking and success rates
- Interactive web dashboard with charts

### Healing Strategies

**Pattern-Based Detection:**
- Memory usage patterns and thresholds
- CPU utilization monitoring
- Connection count and state tracking  
- Request failure rate analysis

**Recovery Actions:**
- Garbage collection and memory cleanup
- Resource throttling and scaling
- Connection pool management
- Graceful degradation modes

## Demo Usage

### Comprehensive Stress Test
Run all healing mechanisms in sequence:
```bash
curl -X POST "http://localhost:8000/demo/stress-test"
```

This will trigger:
1. Memory allocation (3x30MB objects)
2. Connection creation (60 connections)
3. CPU stress (6 intensive threads)

Watch the dashboard at `http://localhost:8000` to see healing actions in real-time.

### Individual Tests
Test specific healing capabilities:
```bash
# Memory leak healing
curl -X POST "http://localhost:8000/demo/memory-leak?size_mb=50&count=4"

# CPU overload healing  
curl -X POST "http://localhost:8000/demo/cpu-overload?duration=20&num_threads=8"

# Connection issues healing
curl -X POST "http://localhost:8000/demo/connection-issues?connection_count=120"

# Request failure healing (probabilistic)
curl -X POST "http://localhost:8000/demo/request-failures?failure_rate=0.9"

# Request failure healing (immediate trigger)
curl -X GET "http://localhost:8000/demo/request-failures-immediate"
```

### Reset Demo Environment
```bash
curl -X POST "http://localhost:8000/demo/reset"
```

## Configuration

### Healing Thresholds
Modify thresholds in `SelfHealingManager.__init__()`:
```python
self.thresholds = {
    'memory_mb': 400,        # Memory usage threshold
    'cpu_percent': 75,       # CPU usage threshold  
    'connection_count': 80,  # Connection count threshold
    'request_latency_ms': 3000,  # Request latency threshold
    'error_rate': 0.1        # Error rate threshold
}
```

### Monitoring Frequency
Change monitoring interval in `start_monitoring()`:
```python
time.sleep(5)  # Check every 5 seconds
```

## Logs and Debugging

The server provides detailed logging for:
- Healing actions triggered and their success
- System metrics and threshold breaches
- Error detection and classification
- Recovery strategy execution

Check the console output for real-time healing activity.

## Integration with Existing Chimera Components

The FastAPI healing server leverages:
- **MeTTa reasoning engine** for pattern-based healing decisions
- **Autonomous evolution system** for adaptive healing strategies  
- **Dynamic monitoring** for runtime system observation
- **Existing logging infrastructure** for consistent output formatting

This creates a complete self-healing system that combines symbolic reasoning with practical infrastructure monitoring and recovery.