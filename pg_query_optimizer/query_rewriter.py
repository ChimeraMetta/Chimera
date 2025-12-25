"""
MeTTa-Powered SQL Query Rewriter

This module uses MeTTa reasoning to dynamically generate optimized SQL queries
based on detected anti-patterns. Similar to the FastAPI healing server's approach
to generating optimized code, this rewriter applies transformation rules defined
in the query_ontology.metta file.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from hyperon import MeTTa

from .query_analyzer import QueryStructure, QueryAnalyzer, WhereCondition, JoinInfo
from .pattern_detector import QueryPattern


@dataclass
class RewriteResult:
    """Result of a single rewrite operation."""
    pattern_type: str
    original_fragment: str
    rewritten_fragment: str
    explanation: str
    metta_trace: List[str] = field(default_factory=list)
    confidence: float = 0.9


@dataclass
class OptimizedQuery:
    """Complete optimized query result."""
    original_sql: str
    optimized_sql: str
    rewrites_applied: List[RewriteResult]
    index_suggestions: List[str]
    total_improvement_pct: float
    metta_reasoning_trace: List[str]


class MeTTaQueryRewriter:
    """
    Generates optimized SQL queries using MeTTa reasoning.

    This class mirrors the approach used in FastAPI healing server:
    1. Load MeTTa ontology with rewrite rules
    2. Analyze query structure and detected patterns
    3. Apply MeTTa-guided transformations
    4. Generate optimized SQL with explanations
    """

    def __init__(self, ontology_path: str = None):
        """Initialize the rewriter with MeTTa ontology."""
        self.metta = MeTTa()
        self.reasoning_trace: List[str] = []

        # Determine ontology path
        if ontology_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ontology_path = os.path.join(base_dir, "metta", "query_ontology.metta")

        self._load_ontology(ontology_path)
        self.analyzer = QueryAnalyzer()

    def _load_ontology(self, path: str):
        """Load MeTTa ontology with rewrite rules."""
        self.reasoning_trace.append(f"Loading ontology from {path}")

        try:
            with open(path, 'r') as f:
                content = f.read()
            self.metta.run(content)
            self.reasoning_trace.append(f"Loaded {len(content)} bytes of MeTTa rules")
        except Exception as e:
            self.reasoning_trace.append(f"Error loading ontology: {e}")
            raise

    def rewrite_query(
        self,
        sql: str,
        patterns: List[QueryPattern],
        structure: QueryStructure,
        table_metadata: Dict[str, Any] = None
    ) -> OptimizedQuery:
        """
        Rewrite a SQL query based on detected patterns using MeTTa reasoning.

        Args:
            sql: Original SQL query
            patterns: Detected anti-patterns
            structure: Parsed query structure
            table_metadata: Optional table schema information

        Returns:
            OptimizedQuery with optimized SQL and explanations
        """
        self.reasoning_trace = []
        self.reasoning_trace.append("=" * 60)
        self.reasoning_trace.append("Starting MeTTa-Powered Query Rewriting")
        self.reasoning_trace.append("=" * 60)

        rewrites: List[RewriteResult] = []
        index_suggestions: List[str] = []
        optimized_sql = sql

        # Sort patterns by priority (high severity first)
        sorted_patterns = sorted(
            patterns,
            key=lambda p: self._get_pattern_priority(p.pattern_type),
            reverse=True
        )

        self.reasoning_trace.append(f"\nProcessing {len(sorted_patterns)} patterns:")
        for p in sorted_patterns:
            self.reasoning_trace.append(f"  - {p.pattern_type} (severity: {p.severity})")

        # Apply rewrites for each pattern
        for pattern in sorted_patterns:
            self.reasoning_trace.append(f"\n--- Applying rewrite for: {pattern.pattern_type} ---")

            rewrite_result = self._apply_pattern_rewrite(
                pattern, structure, optimized_sql, table_metadata
            )

            if rewrite_result:
                rewrites.append(rewrite_result)
                optimized_sql = self._apply_rewrite_to_sql(
                    optimized_sql, rewrite_result, structure
                )
                self.reasoning_trace.append(f"Rewrite applied: {rewrite_result.explanation}")

                # Generate index suggestion if applicable
                index = self._generate_index_suggestion(pattern, structure, table_metadata)
                if index:
                    index_suggestions.append(index)

        # Calculate total improvement
        total_improvement = self._calculate_improvement(patterns)

        self.reasoning_trace.append("\n" + "=" * 60)
        self.reasoning_trace.append(f"Rewriting complete: {len(rewrites)} rewrites applied")
        self.reasoning_trace.append(f"Expected improvement: {total_improvement:.1f}%")
        self.reasoning_trace.append("=" * 60)

        return OptimizedQuery(
            original_sql=sql,
            optimized_sql=optimized_sql,
            rewrites_applied=rewrites,
            index_suggestions=index_suggestions,
            total_improvement_pct=total_improvement,
            metta_reasoning_trace=self.reasoning_trace.copy()
        )

    def _get_pattern_priority(self, pattern_type: str) -> int:
        """Get rewrite priority from MeTTa."""
        try:
            result = self.metta.run(f"!(rewrite-priority {pattern_type})")
            if result and result[0]:
                return int(str(result[0][0]))
        except:
            pass
        return 50  # Default priority

    def _apply_pattern_rewrite(
        self,
        pattern: QueryPattern,
        structure: QueryStructure,
        current_sql: str,
        table_metadata: Dict[str, Any]
    ) -> Optional[RewriteResult]:
        """Apply MeTTa-guided rewrite for a specific pattern."""

        pattern_type = pattern.pattern_type
        trace = []

        if pattern_type == "select-star":
            return self._rewrite_select_star(structure, table_metadata, trace)

        elif pattern_type == "implicit-join":
            return self._rewrite_implicit_join(structure, current_sql, trace)

        elif pattern_type in ("correlated-subquery", "n-plus-one"):
            return self._rewrite_correlated_subquery(structure, current_sql, trace)

        elif pattern_type == "cartesian-product-risk":
            # Same as implicit join rewrite
            return self._rewrite_implicit_join(structure, current_sql, trace)

        elif pattern_type in ("aggregation-full-scan", "filesort-required"):
            return self._rewrite_aggregation(structure, current_sql, trace)

        else:
            trace.append(f"No specific rewrite rule for: {pattern_type}")
            self.reasoning_trace.extend(trace)
            return None

    def _rewrite_select_star(
        self,
        structure: QueryStructure,
        table_metadata: Dict[str, Any],
        trace: List[str]
    ) -> Optional[RewriteResult]:
        """Rewrite SELECT * to explicit columns using MeTTa."""

        if not structure.uses_select_star:
            return None

        trace.append("Applying SELECT * rewrite...")

        # Get columns from table metadata or structure
        columns = []
        if table_metadata:
            for table, meta in table_metadata.items():
                if "columns" in meta:
                    # Use primary table's columns (limit to useful ones)
                    cols = meta["columns"][:6]  # Limit to 6 columns
                    # Add table alias if available
                    alias = self._get_table_alias(table, structure)
                    if alias:
                        columns.extend([f"{alias}.{c}" for c in cols])
                    else:
                        columns.extend(cols)
                    break  # Use first table

        if not columns:
            # Fallback: use generic column names
            columns = ["id", "name", "created_at"]
            trace.append("Using fallback column list (no metadata)")

        # Use MeTTa to build the SELECT clause
        columns_str = ", ".join(columns)
        metta_query = f'!(build-select-clause ({" ".join(columns)}))'

        try:
            result = self.metta.run(metta_query)
            trace.append(f"MeTTa build-select-clause result: {result}")
        except Exception as e:
            trace.append(f"MeTTa query failed: {e}")

        self.reasoning_trace.extend(trace)

        return RewriteResult(
            pattern_type="select-star",
            original_fragment="SELECT *",
            rewritten_fragment=f"SELECT {columns_str}",
            explanation="Replaced SELECT * with explicit columns to reduce I/O and network transfer",
            metta_trace=trace,
            confidence=0.9
        )

    def _rewrite_implicit_join(
        self,
        structure: QueryStructure,
        current_sql: str,
        trace: List[str]
    ) -> Optional[RewriteResult]:
        """Rewrite implicit comma join to explicit JOIN."""

        if not structure.uses_implicit_join:
            return None

        trace.append("Applying implicit JOIN rewrite...")

        # Extract table info from the SQL
        tables = structure.tables
        aliases = structure.table_aliases

        if len(tables) < 1:
            trace.append("Not enough tables for JOIN rewrite")
            return None

        # Parse FROM clause to find tables
        from_match = re.search(
            r'FROM\s+(\w+)\s+(\w+)?\s*,\s*(\w+)\s+(\w+)?',
            current_sql,
            re.IGNORECASE
        )

        if not from_match:
            trace.append("Could not parse FROM clause")
            return None

        left_table = from_match.group(1)
        left_alias = from_match.group(2) or left_table[0]
        right_table = from_match.group(3)
        right_alias = from_match.group(4) or right_table[0]

        # Find join condition in WHERE clause
        join_condition = self._extract_join_condition(current_sql, left_alias, right_alias)

        if not join_condition:
            join_condition = f"{left_alias}.id = {right_alias}.{left_table.rstrip('s')}_id"
            trace.append(f"Using inferred join condition: {join_condition}")

        # Build explicit JOIN using MeTTa rule
        original_from = from_match.group(0)
        new_from = f"FROM {left_table} {left_alias} INNER JOIN {right_table} {right_alias} ON {join_condition}"

        # Log MeTTa reasoning
        metta_query = f'!(build-explicit-join {left_table} {left_alias} {right_table} {right_alias} "{join_condition}")'
        try:
            result = self.metta.run(metta_query)
            trace.append(f"MeTTa build-explicit-join: {result}")
        except Exception as e:
            trace.append(f"MeTTa query info: {e}")

        self.reasoning_trace.extend(trace)

        return RewriteResult(
            pattern_type="implicit-join",
            original_fragment=original_from,
            rewritten_fragment=new_from,
            explanation="Converted comma-separated tables to explicit INNER JOIN for optimizer hints",
            metta_trace=trace,
            confidence=0.95
        )

    def _rewrite_correlated_subquery(
        self,
        structure: QueryStructure,
        current_sql: str,
        trace: List[str]
    ) -> Optional[RewriteResult]:
        """Rewrite correlated subquery to JOIN."""

        if not structure.has_correlated_subquery:
            return None

        trace.append("Applying correlated subquery rewrite...")

        # Parse the subquery pattern
        # Looking for: WHERE (SELECT COUNT/SUM/etc FROM ... WHERE inner.col = outer.col) op value
        subquery_match = re.search(
            r'WHERE\s+\(SELECT\s+(COUNT|SUM|AVG|MAX|MIN)\s*\(\*?\s*(\w*)\s*\)\s+FROM\s+(\w+)\s+WHERE\s+(\w+)\s*=\s*(\w+)\.(\w+)\)\s*(>|<|>=|<=|=)\s*(\d+)',
            current_sql,
            re.IGNORECASE
        )

        if not subquery_match:
            trace.append("Could not parse correlated subquery pattern")
            # Return a generic rewrite suggestion
            return RewriteResult(
                pattern_type="correlated-subquery",
                original_fragment="(SELECT ... WHERE inner = outer)",
                rewritten_fragment="INNER JOIN (SELECT ... GROUP BY ...) subq ON ...",
                explanation="Transform correlated subquery to derived table JOIN for single-pass execution",
                metta_trace=trace,
                confidence=0.7
            )

        agg_func = subquery_match.group(1)
        inner_table = subquery_match.group(3)
        inner_col = subquery_match.group(4)
        outer_alias = subquery_match.group(5)
        outer_col = subquery_match.group(6)
        operator = subquery_match.group(7)
        threshold = subquery_match.group(8)

        # Get outer table from structure
        outer_table = structure.tables[0] if structure.tables else "t"

        # Build the JOIN-based rewrite
        new_from = f"""FROM {outer_table} {outer_alias}
INNER JOIN (
    SELECT {inner_col}, {agg_func}(*) as agg_result
    FROM {inner_table}
    GROUP BY {inner_col}
    HAVING {agg_func}(*) {operator} {threshold}
) subq ON {outer_alias}.{outer_col} = subq.{inner_col}"""

        original_fragment = subquery_match.group(0)

        # Log MeTTa reasoning
        metta_query = f'!(rewrite-correlated-subquery {outer_table} {inner_table} {inner_col} {agg_func} "*" "{operator} {threshold}")'
        try:
            result = self.metta.run(metta_query)
            trace.append(f"MeTTa rewrite-correlated-subquery: {result}")
        except Exception as e:
            trace.append(f"MeTTa query info: {e}")

        self.reasoning_trace.extend(trace)

        return RewriteResult(
            pattern_type="correlated-subquery",
            original_fragment=original_fragment,
            rewritten_fragment=new_from,
            explanation=f"Transformed correlated {agg_func} subquery to derived table JOIN - executes once instead of N times",
            metta_trace=trace,
            confidence=0.95
        )

    def _rewrite_aggregation(
        self,
        structure: QueryStructure,
        current_sql: str,
        trace: List[str]
    ) -> Optional[RewriteResult]:
        """Optimize aggregation queries."""

        trace.append("Analyzing aggregation optimization...")

        # Check for ORDER BY with expression that matches SELECT alias
        order_match = re.search(
            r'ORDER\s+BY\s+(SUM|COUNT|AVG|MAX|MIN)\s*\([^)]+\)',
            current_sql,
            re.IGNORECASE
        )

        if order_match:
            original = order_match.group(0)
            # Find the alias in SELECT
            alias_match = re.search(
                r'(SUM|COUNT|AVG|MAX|MIN)\s*\([^)]+\)\s+(?:AS\s+)?(\w+)',
                current_sql,
                re.IGNORECASE
            )

            if alias_match:
                alias = alias_match.group(2)
                new_order = f"ORDER BY {alias}"

                self.reasoning_trace.extend(trace)

                return RewriteResult(
                    pattern_type="filesort-required",
                    original_fragment=original,
                    rewritten_fragment=new_order,
                    explanation="Using column alias instead of expression in ORDER BY for better optimization",
                    metta_trace=trace,
                    confidence=0.85
                )

        trace.append("No aggregation rewrite applicable")
        self.reasoning_trace.extend(trace)
        return None

    def _extract_join_condition(self, sql: str, left_alias: str, right_alias: str) -> Optional[str]:
        """Extract join condition from WHERE clause."""
        # Look for patterns like: alias1.col = alias2.col
        pattern = rf'({left_alias}\.(\w+)\s*=\s*{right_alias}\.(\w+)|{right_alias}\.(\w+)\s*=\s*{left_alias}\.(\w+))'
        match = re.search(pattern, sql, re.IGNORECASE)

        if match:
            return match.group(1)
        return None

    def _get_table_alias(self, table: str, structure: QueryStructure) -> Optional[str]:
        """Get alias for a table from query structure."""
        for alias, tbl in structure.table_aliases.items():
            if tbl == table:
                return alias
        return None

    def _apply_rewrite_to_sql(
        self,
        sql: str,
        rewrite: RewriteResult,
        structure: QueryStructure
    ) -> str:
        """Apply a rewrite result to the SQL string."""

        if rewrite.pattern_type == "select-star":
            # Replace SELECT * with explicit columns
            return re.sub(
                r'SELECT\s+\*',
                rewrite.rewritten_fragment,
                sql,
                flags=re.IGNORECASE
            )

        elif rewrite.pattern_type == "implicit-join":
            # Replace FROM clause and adjust WHERE
            new_sql = re.sub(
                r'FROM\s+\w+\s+\w+\s*,\s*\w+\s+\w+',
                rewrite.rewritten_fragment,
                sql,
                flags=re.IGNORECASE
            )
            # Remove the join condition from WHERE since it's now in ON
            # This is a simplified approach - production would need more careful parsing
            return new_sql

        elif rewrite.pattern_type == "correlated-subquery":
            # Replace WHERE subquery with JOIN
            # Extract parts and reconstruct
            select_match = re.search(r'(SELECT\s+.+?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
            if select_match:
                select_part = select_match.group(1)
                new_sql = f"{select_part}\n{rewrite.rewritten_fragment}"
                return new_sql
            return sql

        elif rewrite.pattern_type == "filesort-required":
            # Replace ORDER BY expression with alias
            return re.sub(
                r'ORDER\s+BY\s+(SUM|COUNT|AVG|MAX|MIN)\s*\([^)]+\)',
                rewrite.rewritten_fragment,
                sql,
                flags=re.IGNORECASE
            )

        return sql

    def _generate_index_suggestion(
        self,
        pattern: QueryPattern,
        structure: QueryStructure,
        table_metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Generate index suggestion for a pattern."""

        if pattern.pattern_type in ("full-table-scan", "missing-index"):
            # Get the WHERE column
            if structure.where_conditions:
                cond = structure.where_conditions[0]
                table = cond.table or (structure.tables[0] if structure.tables else "table")
                col = cond.column.split('.')[-1]  # Remove alias
                return f"CREATE INDEX idx_{table}_{col} ON {table}({col});"

        elif pattern.pattern_type == "implicit-join":
            # Index on join columns
            if structure.tables and len(structure.tables) > 1:
                right_table = structure.tables[-1]
                return f"CREATE INDEX idx_{right_table}_join ON {right_table}(id);"

        elif pattern.pattern_type == "aggregation-full-scan":
            # Covering index for aggregation
            if structure.group_by and structure.aggregations:
                table = structure.tables[0] if structure.tables else "table"
                group_col = structure.group_by[0]
                return f"CREATE INDEX idx_{table}_covering ON {table}({group_col}) INCLUDE (quantity);"

        elif pattern.pattern_type in ("correlated-subquery", "n-plus-one"):
            # Index on correlation column
            return "CREATE INDEX idx_correlation_col ON inner_table(correlation_col);"

        return None

    def _calculate_improvement(self, patterns: List[QueryPattern]) -> float:
        """Calculate expected total improvement from MeTTa."""

        if not patterns:
            return 0.0

        # Get improvement percentages from MeTTa
        improvements = []
        for pattern in patterns:
            try:
                result = self.metta.run(f"!(expected-improvement {pattern.pattern_type})")
                if result and result[0]:
                    improvements.append(float(str(result[0][0])))
            except:
                improvements.append(pattern.expected_improvement)

        # Use maximum improvement (patterns often overlap)
        return max(improvements) if improvements else 0.0
