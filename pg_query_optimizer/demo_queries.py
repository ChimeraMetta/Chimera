"""
Demo queries with known anti-patterns for the PostgreSQL Query Optimizer PoC.

Each demo query includes:
- Original slow SQL with anti-pattern
- Expected MeTTa-detected patterns
- Optimized SQL suggestion
- Simulated performance metrics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class DemoQuery:
    """Represents a demo slow query with its optimization data."""

    id: str
    name: str
    description: str

    # Original query info
    original_sql: str
    original_duration_ms: float
    original_cost: float

    # Anti-patterns this query demonstrates
    anti_patterns: List[str]

    # Optimization suggestion
    optimized_sql: str
    optimization_type: str
    index_suggestion: str = ""

    # Simulated optimized metrics
    optimized_duration_ms: float = 0.0
    optimized_cost: float = 0.0

    # Table metadata for MeTTa reasoning
    table_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def improvement_percentage(self) -> float:
        """Calculate the percentage improvement."""
        if self.original_duration_ms == 0:
            return 0.0
        return ((self.original_duration_ms - self.optimized_duration_ms)
                / self.original_duration_ms * 100)

    @property
    def cost_reduction(self) -> float:
        """Calculate the cost reduction."""
        if self.original_cost == 0:
            return 0.0
        return ((self.original_cost - self.optimized_cost)
                / self.original_cost * 100)


# Demo Query 1: Full Table Scan - Missing Index
DEMO_QUERY_FULL_SCAN = DemoQuery(
    id="demo_1",
    name="Full Table Scan",
    description="Query scans entire orders table due to missing index on customer_email",

    original_sql="""SELECT * FROM orders
WHERE customer_email = 'user@example.com'""",
    original_duration_ms=2500.0,
    original_cost=125000.0,

    anti_patterns=["full-table-scan", "missing-index", "select-star"],

    optimized_sql="""SELECT order_id, customer_email, total, created_at
FROM orders
WHERE customer_email = 'user@example.com'""",
    optimization_type="create-index",
    index_suggestion="CREATE INDEX idx_orders_customer_email ON orders(customer_email);",

    optimized_duration_ms=15.0,
    optimized_cost=8.5,

    table_metadata={
        "orders": {
            "row_count": 1000000,
            "columns": ["order_id", "customer_email", "product_id", "quantity", "total", "status", "created_at"],
            "indexes": ["PRIMARY KEY (order_id)"],
            "has_index_on_email": False
        }
    }
)


# Demo Query 2: Inefficient JOIN - Implicit Cross Join
DEMO_QUERY_BAD_JOIN = DemoQuery(
    id="demo_2",
    name="Inefficient JOIN",
    description="Implicit join syntax creates potential cartesian product before filtering",

    original_sql="""SELECT o.*, p.name
FROM orders o, products p
WHERE o.product_id = p.id AND o.status = 'pending'""",
    original_duration_ms=8000.0,
    original_cost=450000.0,

    anti_patterns=["implicit-join", "cartesian-product-risk", "select-star", "missing-join-index"],

    optimized_sql="""SELECT o.order_id, o.customer_email, o.total, o.created_at, p.name
FROM orders o
INNER JOIN products p ON o.product_id = p.id
WHERE o.status = 'pending'""",
    optimization_type="rewrite-join",
    index_suggestion="CREATE INDEX idx_orders_status_product ON orders(status, product_id);",

    optimized_duration_ms=120.0,
    optimized_cost=2500.0,

    table_metadata={
        "orders": {
            "row_count": 1000000,
            "columns": ["order_id", "customer_email", "product_id", "quantity", "total", "status", "created_at"],
            "indexes": ["PRIMARY KEY (order_id)"],
        },
        "products": {
            "row_count": 50000,
            "columns": ["id", "name", "price", "category"],
            "indexes": ["PRIMARY KEY (id)"],
        }
    }
)


# Demo Query 3: N+1 Subquery - Correlated Subquery
DEMO_QUERY_N_PLUS_1 = DemoQuery(
    id="demo_3",
    name="N+1 Subquery",
    description="Correlated subquery executes once per row in outer query",

    original_sql="""SELECT * FROM users u
WHERE (SELECT COUNT(*) FROM orders WHERE user_id = u.id) > 5""",
    original_duration_ms=5000.0,
    original_cost=280000.0,

    anti_patterns=["correlated-subquery", "n-plus-one", "select-star"],

    optimized_sql="""SELECT u.id, u.name, u.email
FROM users u
INNER JOIN (
    SELECT user_id, COUNT(*) as order_count
    FROM orders
    GROUP BY user_id
    HAVING COUNT(*) > 5
) o ON u.id = o.user_id""",
    optimization_type="subquery-to-join",
    index_suggestion="CREATE INDEX idx_orders_user_id ON orders(user_id);",

    optimized_duration_ms=80.0,
    optimized_cost=1200.0,

    table_metadata={
        "users": {
            "row_count": 100000,
            "columns": ["id", "name", "email", "created_at"],
            "indexes": ["PRIMARY KEY (id)"],
        },
        "orders": {
            "row_count": 1000000,
            "columns": ["order_id", "user_id", "product_id", "total", "created_at"],
            "indexes": ["PRIMARY KEY (order_id)"],
            "has_index_on_user_id": False
        }
    }
)


# Demo Query 4: Missing Aggregation Index
DEMO_QUERY_AGGREGATION = DemoQuery(
    id="demo_4",
    name="Missing Aggregation Index",
    description="GROUP BY requires full table scan and filesort due to missing covering index",

    original_sql="""SELECT product_id, SUM(quantity) as total_qty
FROM order_items
GROUP BY product_id
ORDER BY SUM(quantity) DESC""",
    original_duration_ms=3500.0,
    original_cost=185000.0,

    anti_patterns=["aggregation-full-scan", "filesort-required", "missing-covering-index"],

    optimized_sql="""SELECT product_id, SUM(quantity) as total_qty
FROM order_items
GROUP BY product_id
ORDER BY total_qty DESC""",
    optimization_type="create-covering-index",
    index_suggestion="CREATE INDEX idx_order_items_product_qty ON order_items(product_id, quantity);",

    optimized_duration_ms=45.0,
    optimized_cost=950.0,

    table_metadata={
        "order_items": {
            "row_count": 5000000,
            "columns": ["id", "order_id", "product_id", "quantity", "price"],
            "indexes": ["PRIMARY KEY (id)", "INDEX (order_id)"],
            "has_covering_index": False
        }
    }
)


# Collection of all demo queries
DEMO_QUERIES: List[DemoQuery] = [
    DEMO_QUERY_FULL_SCAN,
    DEMO_QUERY_BAD_JOIN,
    DEMO_QUERY_N_PLUS_1,
    DEMO_QUERY_AGGREGATION,
]


def get_demo_query(query_id: str) -> DemoQuery | None:
    """Get a demo query by ID."""
    for query in DEMO_QUERIES:
        if query.id == query_id:
            return query
    return None


def get_random_demo_query() -> DemoQuery:
    """Get a random demo query for demonstration."""
    import random
    return random.choice(DEMO_QUERIES)
