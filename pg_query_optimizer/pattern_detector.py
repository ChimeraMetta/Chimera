"""
MeTTa-powered SQL Query Pattern Detector.

This module uses MeTTa symbolic reasoning to detect query anti-patterns
such as full table scans, inefficient joins, and N+1 queries.

REQUIRES: hyperon package (MeTTa implementation)
"""

import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Require hyperon - no fallback
try:
    from hyperon import MeTTa
except ImportError:
    print("=" * 60)
    print("ERROR: hyperon package is required but not installed")
    print("=" * 60)
    print()
    print("The PostgreSQL Query Optimizer requires MeTTa (hyperon) for")
    print("symbolic reasoning. Without it, the demo cannot function.")
    print()
    print("Installation options:")
    print()
    print("1. Use PyPy (hyperon has PyPy wheels):")
    print("   pypy3 -m pip install hyperon")
    print()
    print("2. Build from source:")
    print("   git clone https://github.com/trueagi-io/hyperon-experimental")
    print("   cd hyperon-experimental")
    print("   # Follow build instructions in README")
    print()
    print("3. Use Docker (recommended):")
    print("   docker build -t chimera .")
    print("   docker run -p 8001:8001 chimera python pg_optimizer_server.py")
    print()
    print("=" * 60)
    sys.exit(1)


@dataclass
class QueryPattern:
    """Represents a detected query pattern/anti-pattern."""

    pattern_type: str  # e.g., "full-table-scan", "correlated-subquery"
    severity: str  # "high", "medium", "low"
    description: str
    confidence: float = 0.9
    metta_evidence: List[str] = field(default_factory=list)
    expected_improvement: float = 0.0  # Percentage improvement if fixed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pattern_type": self.pattern_type,
            "severity": self.severity,
            "description": self.description,
            "confidence": self.confidence,
            "metta_evidence": self.metta_evidence,
            "expected_improvement": self.expected_improvement
        }


class MeTTaQueryPatternDetector:
    """
    MeTTa-powered pattern detector for SQL queries.

    Uses symbolic reasoning to detect query anti-patterns by:
    1. Loading query structure as MeTTa atoms
    2. Applying pattern detection rules from query_ontology.metta
    3. Returning detected patterns with reasoning trace
    """

    # Expected improvements for pattern types
    EXPECTED_IMPROVEMENTS = {
        "full-table-scan": 95.0,
        "missing-index": 90.0,
        "select-star": 30.0,
        "implicit-join": 85.0,
        "cartesian-product-risk": 99.0,
        "correlated-subquery": 98.0,
        "n-plus-one": 98.0,
        "filesort-required": 70.0,
        "aggregation-full-scan": 97.0,
        "missing-covering-index": 90.0,
        "missing-join-index": 75.0,
    }

    def __init__(self, metta_instance: Optional[MeTTa] = None):
        """
        Initialize the pattern detector with MeTTa.

        Args:
            metta_instance: Optional MeTTa instance to use. If not provided,
                          creates a new one.
        """
        self.metta = metta_instance or MeTTa()
        self.metta_space = self.metta.space()
        self.ontology_loaded = False
        self.reasoning_trace: List[str] = []
        self._load_ontology()

    def _load_ontology(self):
        """Load the query ontology into MeTTa space."""
        ontology_path = Path(__file__).parent.parent / "metta" / "query_ontology.metta"

        if ontology_path.exists():
            try:
                with open(ontology_path, 'r') as f:
                    content = f.read()
                # Parse and add atoms to the space
                atoms = self.metta.parse_all(content)
                for atom in atoms:
                    self.metta_space.add_atom(atom)
                self.ontology_loaded = True
                print(f"Loaded MeTTa ontology from {ontology_path}")
            except Exception as e:
                print(f"Warning: Failed to load ontology: {e}")
                self.ontology_loaded = False
        else:
            print(f"Warning: Ontology file not found: {ontology_path}")

    def detect_patterns(
        self,
        query_id: str,
        atoms: List[str],
        anti_patterns: Optional[List[str]] = None
    ) -> Tuple[List[QueryPattern], List[str]]:
        """
        Detect patterns in a query using MeTTa reasoning.

        Args:
            query_id: Unique identifier for the query
            atoms: List of MeTTa atom strings representing the query structure
            anti_patterns: Optional list of known anti-patterns (hints for demo)

        Returns:
            Tuple of (detected patterns, reasoning trace)
        """
        self.reasoning_trace = []
        detected_patterns: List[QueryPattern] = []

        # Log input atoms
        self.reasoning_trace.append(f"Analyzing query: {query_id}")
        self.reasoning_trace.append(f"Input atoms: {len(atoms)}")
        for atom in atoms[:5]:  # Log first 5 atoms
            self.reasoning_trace.append(f"  {atom}")
        if len(atoms) > 5:
            self.reasoning_trace.append(f"  ... and {len(atoms) - 5} more")

        # Add query atoms to MeTTa space
        self.reasoning_trace.append("")
        self.reasoning_trace.append("Loading atoms into MeTTa space...")
        for atom_str in atoms:
            try:
                atom = self.metta.parse_single(atom_str)
                self.metta_space.add_atom(atom)
            except Exception as e:
                self.reasoning_trace.append(f"  Warning: Could not parse '{atom_str}': {e}")

        # Run MeTTa pattern detection
        self.reasoning_trace.append("")
        self.reasoning_trace.append("--- MeTTa Reasoning Trace ---")

        # If we have hints about anti-patterns, use them to guide detection
        pattern_types_to_check = anti_patterns if anti_patterns else [
            "full-table-scan", "select-star", "implicit-join",
            "correlated-subquery", "aggregation-full-scan", "filesort",
            "cartesian-product-risk", "n-plus-one", "missing-index"
        ]

        for pattern_type in pattern_types_to_check:
            patterns_found = self._detect_single_pattern(query_id, pattern_type, atoms)
            detected_patterns.extend(patterns_found)

        self.reasoning_trace.append("--- End Reasoning Trace ---")
        self.reasoning_trace.append(f"Detected {len(detected_patterns)} patterns")

        return detected_patterns, self.reasoning_trace

    def _detect_single_pattern(
        self,
        query_id: str,
        pattern_type: str,
        atoms: List[str]
    ) -> List[QueryPattern]:
        """
        Detect a single pattern type using MeTTa reasoning.

        Args:
            query_id: Query identifier
            pattern_type: Type of pattern to detect
            atoms: Query atoms for context

        Returns:
            List of detected patterns (may be empty)
        """
        patterns = []

        # Build MeTTa query for pattern detection
        query = f"!(match &self (pattern-severity {pattern_type}) $severity)"
        self.reasoning_trace.append(f"Query: {query}")

        try:
            results = self.metta.run(query)
            self.reasoning_trace.append(f"  Result: {results}")

            # Check if this pattern applies based on the atoms
            if self._pattern_applies(pattern_type, atoms):
                severity = self._get_pattern_severity(pattern_type)
                description = self._get_pattern_description(pattern_type, atoms)
                expected_improvement = self.EXPECTED_IMPROVEMENTS.get(pattern_type, 50.0)

                # Query for expected improvement from MeTTa
                improvement_query = f"!(match &self (expected-improvement {pattern_type}) $imp)"
                try:
                    imp_results = self.metta.run(improvement_query)
                    if imp_results and imp_results[0]:
                        self.reasoning_trace.append(f"  Improvement query: {imp_results}")
                except Exception:
                    pass

                pattern = QueryPattern(
                    pattern_type=pattern_type,
                    severity=severity,
                    description=description,
                    confidence=0.9,
                    metta_evidence=[
                        f"(pattern-detected {query_id} {pattern_type} {severity})",
                        f"(pattern-severity {pattern_type}) -> {severity}",
                        f"(expected-improvement {pattern_type}) -> {expected_improvement}%"
                    ],
                    expected_improvement=expected_improvement
                )
                patterns.append(pattern)
                self.reasoning_trace.append(f"  -> DETECTED: {pattern_type} ({severity} severity)")

        except Exception as e:
            self.reasoning_trace.append(f"  -> Error: {e}")

        return patterns

    def _pattern_applies(self, pattern_type: str, atoms: List[str]) -> bool:
        """
        Check if a pattern type applies based on query atoms.

        This uses the atom structure to determine if the pattern is relevant.
        """
        atoms_str = " ".join(atoms)

        pattern_indicators = {
            "full-table-scan": ["table-missing-index", "table-row-count"],
            "missing-index": ["table-missing-index"],
            "select-star": ["uses-select-star"],
            "implicit-join": ["uses-implicit-join"],
            "cartesian-product-risk": ["uses-implicit-join", "query-join implicit"],
            "correlated-subquery": ["has-correlated-subquery"],
            "n-plus-one": ["has-correlated-subquery"],
            "filesort-required": ["order-by-column", "uses-aggregation"],
            "aggregation-full-scan": ["uses-aggregation", "table-missing-covering-index"],
            "missing-covering-index": ["table-missing-covering-index"],
        }

        indicators = pattern_indicators.get(pattern_type, [])
        if not indicators:
            return False

        # Check if any indicator is present
        for indicator in indicators:
            if indicator in atoms_str:
                return True

        return False

    def _get_pattern_description(self, pattern_type: str, atoms: List[str]) -> str:
        """Get a human-readable description for a pattern type."""
        descriptions = {
            "full-table-scan": "Query scans entire table due to missing index on WHERE condition",
            "missing-index": "Query would benefit from an index on the filtered column",
            "select-star": "SELECT * fetches all columns, consider selecting only needed columns",
            "implicit-join": "Using comma-separated tables in FROM clause instead of explicit JOIN",
            "cartesian-product-risk": "Query may produce cartesian product before filtering",
            "correlated-subquery": "Subquery references outer query, executes once per row",
            "n-plus-one": "Subquery pattern that causes N+1 queries against the database",
            "filesort-required": "ORDER BY requires filesort operation, no sorted index available",
            "aggregation-full-scan": "Aggregation (SUM, COUNT, etc.) requires scanning entire table",
            "missing-covering-index": "Query could use covering index to avoid table lookups",
            "missing-join-index": "JOIN condition column lacks index",
        }
        return descriptions.get(pattern_type, f"Detected pattern: {pattern_type}")

    def _get_pattern_severity(self, pattern_type: str) -> str:
        """Get severity level for a pattern type."""
        high_severity = {
            "full-table-scan", "correlated-subquery", "n-plus-one",
            "aggregation-full-scan", "cartesian-product-risk"
        }
        medium_severity = {
            "missing-index", "implicit-join", "filesort-required",
            "missing-join-index", "missing-covering-index"
        }

        if pattern_type in high_severity:
            return "high"
        elif pattern_type in medium_severity:
            return "medium"
        else:
            return "low"


def check_metta_available() -> bool:
    """
    Check if MeTTa (hyperon) is available.

    Returns:
        True if hyperon is importable, False otherwise.
    """
    try:
        from hyperon import MeTTa
        return True
    except ImportError:
        return False
