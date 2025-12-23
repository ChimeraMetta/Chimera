"""
PostgreSQL Query Optimizer Server - FastAPI server with live dashboard.

This module provides:
1. FastAPI REST endpoints for query optimization
2. Interactive HTML dashboard with Chart.js visualizations
3. Demo endpoints for showcasing MeTTa-powered optimization
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from .demo_queries import DEMO_QUERIES, get_demo_query, DemoQuery
from .optimization_engine import OptimizationEngine, OptimizationResult


# FastAPI App
app = FastAPI(
    title="PostgreSQL Query Optimizer",
    description="MeTTa-powered PostgreSQL query optimization demo",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global optimization engine
optimization_engine = OptimizationEngine()


# Request/Response Models
class AnalyzeRequest(BaseModel):
    query_id: str


class AnalyzeResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the interactive optimization dashboard."""
    return get_dashboard_html()


@app.get("/api/demo-queries")
async def list_demo_queries():
    """List all available demo queries."""
    return {
        "queries": [
            {
                "id": q.id,
                "name": q.name,
                "description": q.description,
                "anti_patterns": q.anti_patterns,
                "original_duration_ms": q.original_duration_ms,
                "improvement_percentage": q.improvement_percentage
            }
            for q in DEMO_QUERIES
        ]
    }


@app.post("/api/analyze")
async def analyze_query(request: AnalyzeRequest):
    """Analyze a demo query and return optimization suggestions."""
    result = optimization_engine.analyze_demo_query(request.query_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Query {request.query_id} not found")
    return {"success": True, "result": result.to_dict()}


@app.get("/api/analyze/{query_id}")
async def analyze_query_get(query_id: str):
    """Analyze a demo query (GET version for easy testing)."""
    result = optimization_engine.analyze_demo_query(query_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Query {query_id} not found")
    return {"success": True, "result": result.to_dict()}


@app.get("/api/analyze-all")
async def analyze_all_queries():
    """Analyze all demo queries."""
    results = optimization_engine.analyze_all_demos()
    return {
        "success": True,
        "results": [r.to_dict() for r in results],
        "summary": optimization_engine.get_metrics_summary()
    }


@app.get("/api/metrics")
async def get_metrics():
    """Get optimization metrics summary."""
    return optimization_engine.get_metrics_summary()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "pg-query-optimizer",
        "timestamp": datetime.now().isoformat()
    }


def get_dashboard_html() -> str:
    """Generate the interactive dashboard HTML."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PostgreSQL Query Optimizer - MeTTa Demo</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e4;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        h1 {
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle {
            color: #888;
            margin-top: 10px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.2em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .query-selector {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .query-btn {
            padding: 10px 20px;
            background: linear-gradient(135deg, #7b2cbf, #00d4ff);
            border: none;
            border-radius: 25px;
            color: white;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s;
        }
        .query-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0,212,255,0.3);
        }
        .query-btn.active {
            box-shadow: 0 0 20px rgba(0,212,255,0.5);
        }
        .sql-display {
            background: #0d1117;
            border-radius: 10px;
            padding: 15px;
            font-family: 'Fira Code', monospace;
            font-size: 0.85em;
            overflow-x: auto;
            white-space: pre-wrap;
            border: 1px solid #30363d;
        }
        .sql-display.original {
            border-left: 4px solid #ff6b6b;
        }
        .sql-display.optimized {
            border-left: 4px solid #51cf66;
        }
        .pattern-list {
            list-style: none;
        }
        .pattern-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .pattern-item.high {
            background: rgba(255,107,107,0.2);
            border-left: 3px solid #ff6b6b;
        }
        .pattern-item.medium {
            background: rgba(255,193,7,0.2);
            border-left: 3px solid #ffc107;
        }
        .pattern-item.low {
            background: rgba(81,207,102,0.2);
            border-left: 3px solid #51cf66;
        }
        .severity-badge {
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.75em;
            text-transform: uppercase;
        }
        .severity-badge.high { background: #ff6b6b; color: white; }
        .severity-badge.medium { background: #ffc107; color: black; }
        .severity-badge.low { background: #51cf66; color: white; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            background: linear-gradient(90deg, #00d4ff, #51cf66);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-label {
            color: #888;
            font-size: 0.85em;
            margin-top: 5px;
        }
        .reasoning-trace {
            background: #0d1117;
            border-radius: 10px;
            padding: 15px;
            font-family: 'Fira Code', monospace;
            font-size: 0.8em;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #30363d;
        }
        .reasoning-line {
            padding: 2px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .reasoning-line.metta {
            color: #7b2cbf;
        }
        .reasoning-line.result {
            color: #51cf66;
        }
        .chart-container {
            height: 250px;
            margin-top: 15px;
        }
        .full-width {
            grid-column: 1 / -1;
        }
        .improvement-bar {
            background: #1a1a2e;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            margin: 10px 0;
        }
        .improvement-fill {
            height: 100%;
            background: linear-gradient(90deg, #51cf66, #00d4ff);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            transition: width 0.5s ease;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #888;
        }
        .suggestion-box {
            background: rgba(81,207,102,0.1);
            border: 1px solid #51cf66;
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
        }
        .suggestion-title {
            color: #51cf66;
            font-weight: bold;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>PostgreSQL Query Optimizer</h1>
            <p class="subtitle">MeTTa-Powered Query Analysis & Optimization Demo</p>
        </header>

        <div class="metrics-grid" id="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" id="metric-queries">0</div>
                <div class="metric-label">Queries Analyzed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="metric-patterns">0</div>
                <div class="metric-label">Patterns Detected</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="metric-suggestions">0</div>
                <div class="metric-label">Optimizations</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" id="metric-improvement">0%</div>
                <div class="metric-label">Avg Improvement</div>
            </div>
        </div>

        <div class="card full-width">
            <h2>Select Demo Query</h2>
            <div class="query-selector" id="query-selector">
                <!-- Query buttons will be added here -->
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Original Query (Slow)</h2>
                <div class="sql-display original" id="original-sql">
                    Select a demo query to analyze...
                </div>
                <div style="margin-top: 15px; display: flex; gap: 20px;">
                    <div>
                        <span style="color: #888;">Duration:</span>
                        <span id="original-duration" style="color: #ff6b6b; font-weight: bold;">-</span>
                    </div>
                    <div>
                        <span style="color: #888;">Cost:</span>
                        <span id="original-cost" style="color: #ff6b6b; font-weight: bold;">-</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Optimized Query</h2>
                <div class="sql-display optimized" id="optimized-sql">
                    Optimization suggestions will appear here...
                </div>
                <div style="margin-top: 15px; display: flex; gap: 20px;">
                    <div>
                        <span style="color: #888;">Duration:</span>
                        <span id="optimized-duration" style="color: #51cf66; font-weight: bold;">-</span>
                    </div>
                    <div>
                        <span style="color: #888;">Cost:</span>
                        <span id="optimized-cost" style="color: #51cf66; font-weight: bold;">-</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Detected Patterns</h2>
                <ul class="pattern-list" id="pattern-list">
                    <li class="loading">Select a query to see detected patterns...</li>
                </ul>
            </div>

            <div class="card">
                <h2>Performance Improvement</h2>
                <div class="improvement-bar">
                    <div class="improvement-fill" id="improvement-bar" style="width: 0%;">
                        <span id="improvement-text">0%</span>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="performance-chart"></canvas>
                </div>
            </div>
        </div>

        <div class="card full-width">
            <h2>MeTTa Reasoning Trace</h2>
            <div class="reasoning-trace" id="reasoning-trace">
                <div class="reasoning-line">Waiting for query analysis...</div>
            </div>
        </div>

        <div class="card full-width" id="suggestion-card" style="display: none;">
            <h2>Optimization Suggestions</h2>
            <div id="suggestions-container"></div>
        </div>
    </div>

    <script>
        let performanceChart = null;

        // Initialize the dashboard
        async function init() {
            await loadDemoQueries();
            await updateMetrics();
        }

        // Load demo queries and create buttons
        async function loadDemoQueries() {
            try {
                const response = await fetch('/api/demo-queries');
                const data = await response.json();
                const container = document.getElementById('query-selector');
                container.innerHTML = '';

                data.queries.forEach((query, index) => {
                    const btn = document.createElement('button');
                    btn.className = 'query-btn';
                    btn.textContent = query.name;
                    btn.onclick = () => analyzeQuery(query.id, btn);
                    container.appendChild(btn);
                });
            } catch (error) {
                console.error('Error loading queries:', error);
            }
        }

        // Analyze a query
        async function analyzeQuery(queryId, btn) {
            // Update button states
            document.querySelectorAll('.query-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            try {
                const response = await fetch(`/api/analyze/${queryId}`);
                const data = await response.json();

                if (data.success) {
                    displayResult(data.result);
                    await updateMetrics();
                }
            } catch (error) {
                console.error('Error analyzing query:', error);
            }
        }

        // Display analysis result
        function displayResult(result) {
            // Original query
            document.getElementById('original-sql').textContent = result.original_sql;
            document.getElementById('original-duration').textContent = result.original_duration_ms + 'ms';
            document.getElementById('original-cost').textContent = result.original_cost.toLocaleString();

            // Optimized query
            document.getElementById('optimized-sql').textContent = result.optimized_sql;
            document.getElementById('optimized-duration').textContent = result.optimized_duration_ms + 'ms';
            document.getElementById('optimized-cost').textContent = result.optimized_cost.toLocaleString();

            // Improvement bar
            const improvement = result.improvement_percentage.toFixed(1);
            document.getElementById('improvement-bar').style.width = improvement + '%';
            document.getElementById('improvement-text').textContent = improvement + '% faster';

            // Patterns
            const patternList = document.getElementById('pattern-list');
            patternList.innerHTML = '';
            result.detected_patterns.forEach(pattern => {
                const li = document.createElement('li');
                li.className = `pattern-item ${pattern.severity}`;
                li.innerHTML = `
                    <span>${pattern.pattern_type}: ${pattern.description}</span>
                    <span class="severity-badge ${pattern.severity}">${pattern.severity}</span>
                `;
                patternList.appendChild(li);
            });

            // Reasoning trace
            const traceContainer = document.getElementById('reasoning-trace');
            traceContainer.innerHTML = '';
            result.reasoning_trace.forEach(line => {
                const div = document.createElement('div');
                div.className = 'reasoning-line';
                if (line.includes('match') || line.includes('pattern-detected')) {
                    div.classList.add('metta');
                } else if (line.includes('->')) {
                    div.classList.add('result');
                }
                div.textContent = line;
                traceContainer.appendChild(div);
            });

            // Suggestions
            const suggestionCard = document.getElementById('suggestion-card');
            const suggestionsContainer = document.getElementById('suggestions-container');
            if (result.suggestions && result.suggestions.length > 0) {
                suggestionCard.style.display = 'block';
                suggestionsContainer.innerHTML = '';
                result.suggestions.forEach(suggestion => {
                    const div = document.createElement('div');
                    div.className = 'suggestion-box';
                    div.innerHTML = `
                        <div class="suggestion-title">${suggestion.optimization_type}: ${suggestion.description}</div>
                        <pre class="sql-display optimized" style="margin-top: 10px;">${suggestion.suggested_action}</pre>
                        <div style="margin-top: 10px; color: #51cf66;">
                            Expected improvement: ${suggestion.expected_improvement_pct.toFixed(1)}%
                        </div>
                    `;
                    suggestionsContainer.appendChild(div);
                });
            } else {
                suggestionCard.style.display = 'none';
            }

            // Update chart
            updateChart(result);
        }

        // Update the performance chart
        function updateChart(result) {
            const ctx = document.getElementById('performance-chart').getContext('2d');

            if (performanceChart) {
                performanceChart.destroy();
            }

            performanceChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Duration (ms)', 'Cost'],
                    datasets: [
                        {
                            label: 'Original',
                            data: [result.original_duration_ms, result.original_cost / 1000],
                            backgroundColor: 'rgba(255, 107, 107, 0.7)',
                            borderColor: '#ff6b6b',
                            borderWidth: 1
                        },
                        {
                            label: 'Optimized',
                            data: [result.optimized_duration_ms, result.optimized_cost / 1000],
                            backgroundColor: 'rgba(81, 207, 102, 0.7)',
                            borderColor: '#51cf66',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#e4e4e4' }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            ticks: { color: '#888' }
                        },
                        x: {
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            ticks: { color: '#888' }
                        }
                    }
                }
            });
        }

        // Update metrics
        async function updateMetrics() {
            try {
                const response = await fetch('/api/metrics');
                const metrics = await response.json();

                document.getElementById('metric-queries').textContent = metrics.total_queries_analyzed || 0;
                document.getElementById('metric-patterns').textContent = metrics.total_patterns_detected || 0;
                document.getElementById('metric-suggestions').textContent = metrics.total_suggestions_generated || 0;
                document.getElementById('metric-improvement').textContent =
                    (metrics.average_improvement_pct || 0).toFixed(1) + '%';
            } catch (error) {
                console.error('Error updating metrics:', error);
            }
        }

        // Initialize on load
        init();
    </script>
</body>
</html>'''
