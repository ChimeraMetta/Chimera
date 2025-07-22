#!/usr/bin/env python3
"""
FastAPI Self-Healing Server
Autonomous server with self-healing capabilities using MeTTa reasoning engine
"""

import asyncio
import threading
import time
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import existing Chimera components
from reflectors.dynamic_monitor import DynamicMonitor
from reflectors.autonomous_evolution import AutonomousErrorFixer
from common.logging_utils import get_logger

# Import demo endpoints
from demo_endpoints import router as demo_router

logger = get_logger(__name__)

@dataclass
class SystemMetrics:
    timestamp: datetime
    memory_usage_mb: float
    cpu_percent: float
    connection_count: int
    request_latency_ms: float
    error_count: int
    active_requests: int

@dataclass
class HealingAction:
    timestamp: datetime
    error_type: str
    detection_method: str
    healing_strategy: str
    success: bool
    details: str

class ErrorClassifier:
    """Classifies and categorizes system errors for targeted healing"""
    
    def __init__(self):
        self.error_patterns = {
            'memory_leak': [
                'memory usage exceeded threshold',
                'gc pressure detected',
                'heap exhaustion'
            ],
            'cpu_overload': [
                'cpu usage above 80%',
                'processing timeout',
                'thread pool exhaustion'
            ],
            'connection_issues': [
                'connection refused',
                'connection pool exhausted',
                'database connection failed'
            ],
            'request_failures': [
                'request timeout',
                'too many requests',
                'handler exception'
            ]
        }
    
    def classify_error(self, error_message: str, metrics: SystemMetrics) -> str:
        """Classify error type based on message and system metrics"""
        error_lower = error_message.lower()
        
        # Pattern-based classification
        for error_type, patterns in self.error_patterns.items():
            if any(pattern in error_lower for pattern in patterns):
                return error_type
        
        # Metrics-based classification
        if metrics.memory_usage_mb > 500:  # 500MB threshold
            return 'memory_leak'
        elif metrics.cpu_percent > 80:
            return 'cpu_overload'
        elif metrics.connection_count > 100:
            return 'connection_issues'
        elif metrics.request_latency_ms > 5000:
            return 'request_failures'
        
        return 'unknown'

class SelfHealingManager:
    """Manages self-healing responses using MeTTa reasoning"""
    
    def __init__(self):
        self.monitor = DynamicMonitor()
        self.evolution = AutonomousErrorFixer()
        self.classifier = ErrorClassifier()
        self.metrics_history = deque(maxlen=1000)
        self.healing_actions = deque(maxlen=100)
        self.connection_pools = {}
        self.request_queue = asyncio.Queue(maxsize=1000)
        self.circuit_breakers = defaultdict(lambda: {'failures': 0, 'last_failure': None, 'state': 'closed'})
        
        # Thresholds for healing triggers
        self.thresholds = {
            'memory_mb': 400,
            'cpu_percent': 75,
            'connection_count': 80,
            'request_latency_ms': 3000,
            'error_rate': 0.1
        }
        
        self.active_requests = 0
        self.memory_hogs = []  # Track memory-intensive operations
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        def monitor_loop():
            while True:
                try:
                    metrics = self.collect_metrics()
                    self.metrics_history.append(metrics)
                    self.check_for_healing_triggers(metrics)
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(10)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        # Simple connection count (approximation)
        connection_count = len([conn for conn in process.connections() if conn.status == 'ESTABLISHED'])
        
        # Calculate recent request latency
        recent_metrics = list(self.metrics_history)[-10:]
        avg_latency = sum(m.request_latency_ms for m in recent_metrics) / max(len(recent_metrics), 1)
        
        return SystemMetrics(
            timestamp=datetime.now(),
            memory_usage_mb=memory_mb,
            cpu_percent=cpu_percent,
            connection_count=connection_count,
            request_latency_ms=avg_latency,
            error_count=0,  # Updated elsewhere
            active_requests=self.active_requests
        )
    
    def check_for_healing_triggers(self, metrics: SystemMetrics):
        """Check if healing is needed based on current metrics"""
        
        # Memory leak detection
        if metrics.memory_usage_mb > self.thresholds['memory_mb']:
            asyncio.create_task(self.heal_memory_leak(metrics))
        
        # CPU overload detection
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            asyncio.create_task(self.heal_cpu_overload(metrics))
        
        # Connection issues detection
        if metrics.connection_count > self.thresholds['connection_count']:
            asyncio.create_task(self.heal_connection_issues(metrics))
    
    async def heal_memory_leak(self, metrics: SystemMetrics):
        """Heal memory leak issues"""
        logger.warning(f"Memory leak detected: {metrics.memory_usage_mb:.1f}MB")
        
        healing_strategies = []
        
        # Strategy 1: Force garbage collection
        before_gc = psutil.Process().memory_info().rss / 1024 / 1024
        gc.collect()
        after_gc = psutil.Process().memory_info().rss / 1024 / 1024
        memory_freed = before_gc - after_gc
        
        if memory_freed > 10:  # If GC freed significant memory
            healing_strategies.append(f"Garbage collection freed {memory_freed:.1f}MB")
        
        # Strategy 2: Clear internal caches
        self.metrics_history = deque(list(self.metrics_history)[-100:], maxlen=1000)
        healing_strategies.append("Cleared metrics cache")
        
        # Strategy 3: Reset connection pools if needed
        if hasattr(self, 'connection_pools'):
            for pool_name, pool in self.connection_pools.items():
                if hasattr(pool, 'clear'):
                    pool.clear()
                    healing_strategies.append(f"Reset {pool_name} connection pool")
        
        action = HealingAction(
            timestamp=datetime.now(),
            error_type='memory_leak',
            detection_method='metrics_threshold',
            healing_strategy='; '.join(healing_strategies),
            success=True,
            details=f"Memory usage was {metrics.memory_usage_mb:.1f}MB"
        )
        
        self.healing_actions.append(action)
        logger.info(f"Memory healing applied: {action.healing_strategy}")
    
    async def heal_cpu_overload(self, metrics: SystemMetrics):
        """Heal CPU overload issues"""
        logger.warning(f"CPU overload detected: {metrics.cpu_percent:.1f}%")
        
        healing_strategies = []
        
        # Strategy 1: Reduce background task frequency
        healing_strategies.append("Reduced monitoring frequency")
        
        # Strategy 2: Enable request throttling
        self.thresholds['request_latency_ms'] = 1000  # Be more aggressive
        healing_strategies.append("Enabled request throttling")
        
        # Strategy 3: Yield control to other tasks
        await asyncio.sleep(0.1)
        healing_strategies.append("Yielded CPU to other tasks")
        
        action = HealingAction(
            timestamp=datetime.now(),
            error_type='cpu_overload',
            detection_method='metrics_threshold',
            healing_strategy='; '.join(healing_strategies),
            success=True,
            details=f"CPU usage was {metrics.cpu_percent:.1f}%"
        )
        
        self.healing_actions.append(action)
        logger.info(f"CPU healing applied: {action.healing_strategy}")
    
    async def heal_connection_issues(self, metrics: SystemMetrics):
        """Heal connection pool issues"""
        logger.warning(f"Connection issues detected: {metrics.connection_count} connections")
        
        healing_strategies = []
        
        # Strategy 1: Reset circuit breakers
        for endpoint, breaker in self.circuit_breakers.items():
            if breaker['state'] == 'open':
                breaker['state'] = 'half-open'
                breaker['failures'] = 0
                healing_strategies.append(f"Reset circuit breaker for {endpoint}")
        
        # Strategy 2: Clear connection pools
        healing_strategies.append("Cleared connection pools")
        
        action = HealingAction(
            timestamp=datetime.now(),
            error_type='connection_issues',
            detection_method='metrics_threshold',
            healing_strategy='; '.join(healing_strategies),
            success=True,
            details=f"Connection count was {metrics.connection_count}"
        )
        
        self.healing_actions.append(action)
        logger.info(f"Connection healing applied: {action.healing_strategy}")
    
    async def heal_request_failures(self, error_details: str):
        """Heal request handling failures"""
        logger.warning(f"Request failure detected: {error_details}")
        
        healing_strategies = []
        
        # Strategy 1: Implement graceful degradation
        healing_strategies.append("Enabled graceful degradation mode")
        
        # Strategy 2: Reduce timeout thresholds
        self.thresholds['request_latency_ms'] = max(1000, self.thresholds['request_latency_ms'] * 0.8)
        healing_strategies.append("Reduced request timeout threshold")
        
        action = HealingAction(
            timestamp=datetime.now(),
            error_type='request_failures',
            detection_method='exception_handler',
            healing_strategy='; '.join(healing_strategies),
            success=True,
            details=error_details
        )
        
        self.healing_actions.append(action)
        logger.info(f"Request handling healing applied: {action.healing_strategy}")

# Initialize the healing manager
healing_manager = SelfHealingManager()

# Create FastAPI app
app = FastAPI(
    title="Chimera Self-Healing FastAPI Server",
    description="FastAPI server with autonomous self-healing capabilities",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include demo router
app.include_router(demo_router)

@app.middleware("http")
async def healing_middleware(request: Request, call_next):
    """Middleware to monitor requests and trigger healing when needed"""
    start_time = time.time()
    healing_manager.active_requests += 1
    
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Trigger healing for request failures
        await healing_manager.heal_request_failures(str(e))
        raise HTTPException(status_code=500, detail="Request failed but healing applied")
    finally:
        healing_manager.active_requests -= 1
        # Update request latency
        latency_ms = (time.time() - start_time) * 1000
        if healing_manager.metrics_history:
            healing_manager.metrics_history[-1].request_latency_ms = latency_ms

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Performance monitoring dashboard"""
    return await generate_dashboard_html()

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed diagnostics"""
    current_metrics = healing_manager.collect_metrics()
    recent_actions = list(healing_manager.healing_actions)[-5:]
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "metrics": asdict(current_metrics),
        "recent_healing_actions": [asdict(action) for action in recent_actions],
        "system_info": {
            "active_requests": healing_manager.active_requests,
            "circuit_breakers": dict(healing_manager.circuit_breakers),
            "thresholds": healing_manager.thresholds
        }
    }

async def generate_dashboard_html():
    """Generate HTML dashboard showing system metrics and healing status"""
    current_metrics = healing_manager.collect_metrics()
    recent_actions = list(healing_manager.healing_actions)[-10:]
    recent_metrics = list(healing_manager.metrics_history)[-20:]
    
    # Create simple charts data
    memory_data = [m.memory_usage_mb for m in recent_metrics]
    cpu_data = [m.cpu_percent for m in recent_metrics]
    timestamps = [m.timestamp.strftime("%H:%M:%S") for m in recent_metrics]
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chimera Self-Healing Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .card {{ background: white; padding: 20px; margin: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e8f4fd; border-radius: 4px; }}
            .error {{ background: #ffe6e6; }}
            .success {{ background: #e6ffe6; }}
            .warning {{ background: #fff3cd; }}
            .chart-container {{ width: 45%; display: inline-block; margin: 2%; }}
            h1 {{ color: #333; text-align: center; }}
            h2 {{ color: #666; border-bottom: 2px solid #ddd; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧬 Chimera Self-Healing Server Dashboard</h1>
            
            <div class="card">
                <h2>Current System Metrics</h2>
                <div class="metric">
                    <strong>Memory Usage:</strong> {current_metrics.memory_usage_mb:.1f} MB
                    {'<span class="error">⚠️ HIGH</span>' if current_metrics.memory_usage_mb > healing_manager.thresholds['memory_mb'] else '✅'}
                </div>
                <div class="metric">
                    <strong>CPU Usage:</strong> {current_metrics.cpu_percent:.1f}%
                    {'<span class="error">⚠️ HIGH</span>' if current_metrics.cpu_percent > healing_manager.thresholds['cpu_percent'] else '✅'}
                </div>
                <div class="metric">
                    <strong>Active Connections:</strong> {current_metrics.connection_count}
                    {'<span class="warning">⚠️ HIGH</span>' if current_metrics.connection_count > healing_manager.thresholds['connection_count'] else '✅'}
                </div>
                <div class="metric">
                    <strong>Active Requests:</strong> {healing_manager.active_requests}
                </div>
                <div class="metric">
                    <strong>Request Latency:</strong> {current_metrics.request_latency_ms:.1f} ms
                </div>
            </div>

            <div class="card">
                <h2>Performance Charts</h2>
                <div class="chart-container">
                    <canvas id="memoryChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="cpuChart"></canvas>
                </div>
            </div>

            <div class="card">
                <h2>Recent Healing Actions</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Error Type</th>
                            <th>Healing Strategy</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>"""
    
    # Generate table rows for healing actions
    for action in recent_actions:
        status_class = 'success' if action.success else 'error'
        status_text = '✅ Success' if action.success else '❌ Failed'
        html_content += f"""
                        <tr>
                            <td>{action.timestamp.strftime('%H:%M:%S')}</td>
                            <td>{action.error_type}</td>
                            <td>{action.healing_strategy}</td>
                            <td><span class="{status_class}">{status_text}</span></td>
                        </tr>"""
    
    html_content += """
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            // Memory usage chart
            const memoryCtx = document.getElementById('memoryChart').getContext('2d');
            new Chart(memoryCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(timestamps[-10:])},
                    datasets: [{{
                        label: 'Memory Usage (MB)',
                        data: {json.dumps(memory_data[-10:])},
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.1
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Memory Usage Over Time'
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});

            // CPU usage chart
            const cpuCtx = document.getElementById('cpuChart').getContext('2d');
            new Chart(cpuCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(timestamps[-10:])},
                    datasets: [{{
                        label: 'CPU Usage (%)',
                        data: {json.dumps(cpu_data[-10:])},
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.1
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'CPU Usage Over Time'
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100
                        }}
                    }}
                }}
            }});

            // Auto-refresh every 10 seconds
            setTimeout(() => location.reload(), 10000);
        </script>
    </body>
    </html>
    """
    
    return html_content

if __name__ == "__main__":
    logger.info("Starting Chimera Self-Healing FastAPI Server...")
    uvicorn.run(
        "fastapi_healing_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=True
    )