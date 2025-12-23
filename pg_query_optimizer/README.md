# PostgreSQL Query Optimizer - MeTTa Demo

MeTTa-powered PostgreSQL query optimization proof-of-concept demonstrating symbolic reasoning for query analysis.

## Overview

This demo shows how MeTTa (Meta Type Talk) can be used to:
1. **Detect query anti-patterns** - Full table scans, bad joins, N+1 queries, missing indexes
2. **Generate optimization suggestions** - Index recommendations, query rewrites
3. **Show before/after comparisons** - Performance improvement metrics

## Quick Start

```bash
# From the Chimera directory
python3 pg_optimizer_server.py
```

Then open http://localhost:8001 in your browser.

## Demo Scenarios

### 1. Full Table Scan
```sql
SELECT * FROM orders WHERE customer_email = 'user@example.com';
```
- **Pattern**: Missing index causes full table scan on 1M rows
- **Suggestion**: CREATE INDEX idx_orders_customer_email
- **Improvement**: 2500ms → 15ms (99.4% faster)

### 2. Inefficient JOIN
```sql
SELECT o.*, p.name FROM orders o, products p
WHERE o.product_id = p.id AND o.status = 'pending';
```
- **Pattern**: Implicit join with cartesian product risk
- **Suggestion**: Rewrite as explicit INNER JOIN
- **Improvement**: 8000ms → 120ms (98.5% faster)

### 3. N+1 Subquery
```sql
SELECT * FROM users u
WHERE (SELECT COUNT(*) FROM orders WHERE user_id = u.id) > 5;
```
- **Pattern**: Correlated subquery executes once per row
- **Suggestion**: Rewrite as JOIN with GROUP BY
- **Improvement**: 5000ms → 80ms (98.4% faster)

### 4. Missing Aggregation Index
```sql
SELECT product_id, SUM(quantity) FROM order_items
GROUP BY product_id ORDER BY SUM(quantity) DESC;
```
- **Pattern**: GROUP BY requires full scan and filesort
- **Suggestion**: CREATE covering index
- **Improvement**: 3500ms → 45ms (98.7% faster)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Interactive dashboard |
| `/api/demo-queries` | GET | List all demo queries |
| `/api/analyze/{query_id}` | GET | Analyze a specific demo query |
| `/api/analyze-all` | GET | Analyze all demo queries |
| `/api/metrics` | GET | Get optimization metrics |
| `/health` | GET | Health check |

## Architecture

```
pg_query_optimizer/
├── demo_queries.py      # Synthetic slow queries with anti-patterns
├── query_analyzer.py    # SQL parsing → MeTTa atoms
├── pattern_detector.py  # MeTTa-powered pattern detection
├── optimization_engine.py # Suggestion generation
└── optimizer_server.py  # FastAPI + dashboard

metta/
└── query_ontology.metta # Pattern detection rules
```

## MeTTa Integration

The system uses MeTTa symbolic reasoning for pattern detection:

```lisp
;; Pattern detection rule example
(= (detect-pattern full-table-scan $query-id $atoms)
   (match $atoms
     (, (query-table $table)
        (where-condition $column $op)
        (table-row-count $table $rows)
        (table-missing-index $table $idx-col))
     (if (> $rows 10000)
         (pattern-detected $query-id full-table-scan high ...)
         Empty)))
```

## Dependencies

- Python 3.8+ (or PyPy 3.8+)
- fastapi
- uvicorn
- **hyperon** (REQUIRED - MeTTa implementation)

### Installing hyperon

hyperon provides pre-built wheels for:
- Linux (x86-64, i686, ARM64)
- macOS ARM64 (Apple Silicon) - **PyPy only**

```bash
# Option 1: Use PyPy (recommended for macOS)
pypy3 -m pip install hyperon fastapi uvicorn

# Option 2: Linux with CPython
pip install hyperon

# Option 3: Build from source
git clone https://github.com/trueagi-io/hyperon-experimental
cd hyperon-experimental
# Follow build instructions in README

# Option 4: Use Docker
docker build -t chimera .
docker run -p 8001:8001 chimera python pg_optimizer_server.py
```

## Files

| File | Lines | Description |
|------|-------|-------------|
| `demo_queries.py` | ~180 | 4 demo queries with simulated metrics |
| `query_analyzer.py` | ~350 | SQL parsing and MeTTa atom generation |
| `pattern_detector.py` | ~300 | MeTTa/fallback pattern detection |
| `optimization_engine.py` | ~250 | Suggestion generation |
| `optimizer_server.py` | ~400 | FastAPI server + HTML dashboard |
| `query_ontology.metta` | ~150 | MeTTa pattern rules |
