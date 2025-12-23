"""
PostgreSQL Query Optimizer - MeTTa-powered query optimization PoC demo.

REQUIRES: hyperon package (MeTTa implementation)

This module demonstrates how MeTTa symbolic reasoning can be used to:
1. Detect query anti-patterns (full table scans, bad joins, N+1 queries)
2. Generate optimization suggestions (indexes, query rewrites)
3. Show before/after performance comparisons

If hyperon is not installed, the module will exit with installation instructions.
"""

# These imports don't require hyperon
from .demo_queries import DemoQuery, DEMO_QUERIES
from .query_analyzer import QueryAnalyzer, QueryStructure

# These imports require hyperon - will exit if not available
from .pattern_detector import MeTTaQueryPatternDetector, QueryPattern, check_metta_available
from .optimization_engine import OptimizationEngine, OptimizationSuggestion

__all__ = [
    'DemoQuery',
    'DEMO_QUERIES',
    'QueryAnalyzer',
    'QueryStructure',
    'MeTTaQueryPatternDetector',
    'QueryPattern',
    'OptimizationEngine',
    'OptimizationSuggestion',
    'check_metta_available',
]
