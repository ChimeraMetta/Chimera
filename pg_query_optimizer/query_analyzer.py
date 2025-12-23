"""
SQL Query Analyzer - Parses SQL queries and converts them to MeTTa atoms.

This module provides basic SQL parsing to extract:
- Query type (SELECT, INSERT, UPDATE, DELETE)
- Tables involved
- JOIN conditions
- WHERE clauses
- GROUP BY / ORDER BY columns
- Aggregation functions
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class JoinInfo:
    """Information about a JOIN in a query."""
    join_type: str  # INNER, LEFT, RIGHT, CROSS, implicit
    left_table: str
    right_table: str
    condition: str


@dataclass
class WhereCondition:
    """Information about a WHERE condition."""
    column: str
    operator: str
    value: str
    table: str = ""


@dataclass
class QueryStructure:
    """Parsed structure of a SQL query."""

    query_type: str  # SELECT, INSERT, UPDATE, DELETE
    tables: List[str] = field(default_factory=list)
    table_aliases: Dict[str, str] = field(default_factory=dict)  # alias -> table
    columns: List[str] = field(default_factory=list)
    joins: List[JoinInfo] = field(default_factory=list)
    where_conditions: List[WhereCondition] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    order_by: List[str] = field(default_factory=list)
    aggregations: List[str] = field(default_factory=list)
    has_subquery: bool = False
    has_correlated_subquery: bool = False
    uses_select_star: bool = False
    uses_implicit_join: bool = False

    def to_metta_atoms(self) -> List[str]:
        """Convert query structure to MeTTa atoms for reasoning."""
        atoms = []

        # Query type atom
        atoms.append(f"(query-type {self.query_type.lower()})")

        # Table atoms
        for table in self.tables:
            atoms.append(f"(query-table {table})")

        # Column atoms
        if self.uses_select_star:
            atoms.append("(uses-select-star)")

        for col in self.columns:
            atoms.append(f"(query-column {col})")

        # Join atoms
        for join in self.joins:
            atoms.append(f"(query-join {join.join_type} {join.left_table} {join.right_table})")

        if self.uses_implicit_join:
            atoms.append("(uses-implicit-join)")

        # WHERE condition atoms
        for cond in self.where_conditions:
            table_part = f"{cond.table}." if cond.table else ""
            atoms.append(f"(where-condition {table_part}{cond.column} {cond.operator})")

        # GROUP BY atoms
        for col in self.group_by:
            atoms.append(f"(group-by-column {col})")

        # ORDER BY atoms
        for col in self.order_by:
            atoms.append(f"(order-by-column {col})")

        # Aggregation atoms
        for agg in self.aggregations:
            atoms.append(f"(uses-aggregation {agg})")

        # Subquery atoms
        if self.has_subquery:
            atoms.append("(has-subquery)")
        if self.has_correlated_subquery:
            atoms.append("(has-correlated-subquery)")

        return atoms


class QueryAnalyzer:
    """
    Analyzes SQL queries to extract structural information.

    This is a simplified parser for the PoC demo. It handles common
    SQL patterns but is not a complete SQL parser.
    """

    # Common SQL keywords for parsing
    KEYWORDS = {
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
        'CROSS', 'ON', 'AND', 'OR', 'GROUP', 'BY', 'ORDER', 'HAVING',
        'INSERT', 'UPDATE', 'DELETE', 'INTO', 'SET', 'VALUES', 'AS',
        'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'ASC', 'DESC', 'DISTINCT'
    }

    AGGREGATION_FUNCTIONS = {'COUNT', 'SUM', 'AVG', 'MIN', 'MAX'}

    def __init__(self):
        """Initialize the query analyzer."""
        pass

    def analyze(self, sql: str) -> QueryStructure:
        """
        Analyze a SQL query and return its structure.

        Args:
            sql: The SQL query string to analyze

        Returns:
            QueryStructure with parsed query information
        """
        # Normalize SQL (remove extra whitespace, uppercase keywords)
        normalized = self._normalize_sql(sql)

        # Determine query type
        query_type = self._get_query_type(normalized)

        structure = QueryStructure(query_type=query_type)

        if query_type == "SELECT":
            self._parse_select(normalized, structure)
        elif query_type == "INSERT":
            self._parse_insert(normalized, structure)
        elif query_type == "UPDATE":
            self._parse_update(normalized, structure)
        elif query_type == "DELETE":
            self._parse_delete(normalized, structure)

        return structure

    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL by removing extra whitespace."""
        # Remove newlines and extra spaces
        normalized = ' '.join(sql.split())
        return normalized

    def _get_query_type(self, sql: str) -> str:
        """Determine the type of SQL query."""
        sql_upper = sql.strip().upper()
        if sql_upper.startswith("SELECT"):
            return "SELECT"
        elif sql_upper.startswith("INSERT"):
            return "INSERT"
        elif sql_upper.startswith("UPDATE"):
            return "UPDATE"
        elif sql_upper.startswith("DELETE"):
            return "DELETE"
        return "UNKNOWN"

    def _parse_select(self, sql: str, structure: QueryStructure):
        """Parse a SELECT query."""
        sql_upper = sql.upper()

        # Check for SELECT *
        if re.search(r'SELECT\s+\*', sql_upper) or re.search(r'SELECT\s+\w+\.\*', sql_upper):
            structure.uses_select_star = True

        # Check for subqueries
        if '(' in sql and 'SELECT' in sql_upper[sql_upper.find('('):]:
            structure.has_subquery = True
            # Check for correlated subquery (references outer table)
            if re.search(r'WHERE\s+.*\.\s*\w+\s*=\s*\w+\.\w+', sql_upper):
                structure.has_correlated_subquery = True

        # Extract tables and aliases from FROM clause
        self._extract_tables(sql, structure)

        # Check for implicit joins (comma-separated tables in FROM)
        from_match = re.search(r'FROM\s+(.+?)(?:WHERE|GROUP|ORDER|HAVING|$)', sql_upper)
        if from_match:
            from_clause = from_match.group(1)
            # If there are commas but no JOIN keyword, it's an implicit join
            if ',' in from_clause and 'JOIN' not in from_clause:
                structure.uses_implicit_join = True

        # Extract explicit JOINs
        self._extract_joins(sql, structure)

        # Extract WHERE conditions
        self._extract_where_conditions(sql, structure)

        # Extract GROUP BY
        self._extract_group_by(sql, structure)

        # Extract ORDER BY
        self._extract_order_by(sql, structure)

        # Extract aggregations
        self._extract_aggregations(sql, structure)

        # Extract columns
        self._extract_columns(sql, structure)

    def _extract_tables(self, sql: str, structure: QueryStructure):
        """Extract table names and aliases from the query."""
        sql_upper = sql.upper()

        # Find FROM clause
        from_match = re.search(r'FROM\s+(.+?)(?:WHERE|GROUP|ORDER|HAVING|INNER|LEFT|RIGHT|CROSS|JOIN|$)',
                               sql_upper, re.IGNORECASE)
        if not from_match:
            return

        from_clause = from_match.group(1).strip()

        # Split by comma for multiple tables
        table_parts = from_clause.split(',')

        for part in table_parts:
            part = part.strip()
            # Match "table_name alias" or "table_name AS alias" or just "table_name"
            match = re.match(r'(\w+)(?:\s+(?:AS\s+)?(\w+))?', part, re.IGNORECASE)
            if match:
                table_name = match.group(1).lower()
                alias = match.group(2).lower() if match.group(2) else None

                if table_name not in ['inner', 'left', 'right', 'cross', 'join', 'on']:
                    structure.tables.append(table_name)
                    if alias:
                        structure.table_aliases[alias] = table_name

        # Also extract tables from JOINs
        join_pattern = r'JOIN\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?\s+ON'
        for match in re.finditer(join_pattern, sql, re.IGNORECASE):
            table_name = match.group(1).lower()
            alias = match.group(2).lower() if match.group(2) else None

            if table_name not in structure.tables:
                structure.tables.append(table_name)
            if alias:
                structure.table_aliases[alias] = table_name

    def _extract_joins(self, sql: str, structure: QueryStructure):
        """Extract JOIN information from the query."""
        # Pattern for explicit JOINs
        join_pattern = r'(INNER|LEFT|RIGHT|CROSS)?\s*JOIN\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?\s+ON\s+(.+?)(?=(?:INNER|LEFT|RIGHT|CROSS)?\s*JOIN|WHERE|GROUP|ORDER|HAVING|$)'

        for match in re.finditer(join_pattern, sql, re.IGNORECASE):
            join_type = (match.group(1) or 'INNER').upper()
            right_table = match.group(2).lower()
            condition = match.group(4).strip()

            # Try to determine left table from condition
            left_table = structure.tables[0] if structure.tables else ""

            join_info = JoinInfo(
                join_type=join_type,
                left_table=left_table,
                right_table=right_table,
                condition=condition
            )
            structure.joins.append(join_info)

        # Handle implicit joins (comma syntax)
        if structure.uses_implicit_join and len(structure.tables) > 1:
            for i in range(1, len(structure.tables)):
                join_info = JoinInfo(
                    join_type="implicit",
                    left_table=structure.tables[0],
                    right_table=structure.tables[i],
                    condition="FROM clause"
                )
                structure.joins.append(join_info)

    def _extract_where_conditions(self, sql: str, structure: QueryStructure):
        """Extract WHERE conditions from the query."""
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|HAVING|$)', sql, re.IGNORECASE)
        if not where_match:
            return

        where_clause = where_match.group(1).strip()

        # Split by AND/OR (simplified)
        conditions = re.split(r'\s+AND\s+|\s+OR\s+', where_clause, flags=re.IGNORECASE)

        for cond in conditions:
            cond = cond.strip()
            if not cond:
                continue

            # Skip subquery conditions
            if '(' in cond and 'SELECT' in cond.upper():
                continue

            # Match patterns like "column = value" or "table.column = value"
            match = re.match(r'(?:(\w+)\.)?(\w+)\s*(=|>|<|>=|<=|<>|!=|LIKE|IN)\s*(.+)', cond, re.IGNORECASE)
            if match:
                where_cond = WhereCondition(
                    table=match.group(1) or "",
                    column=match.group(2),
                    operator=match.group(3).upper(),
                    value=match.group(4).strip()
                )
                structure.where_conditions.append(where_cond)

    def _extract_group_by(self, sql: str, structure: QueryStructure):
        """Extract GROUP BY columns from the query."""
        group_match = re.search(r'GROUP\s+BY\s+(.+?)(?:HAVING|ORDER|$)', sql, re.IGNORECASE)
        if not group_match:
            return

        group_clause = group_match.group(1).strip()
        columns = [col.strip() for col in group_clause.split(',')]
        structure.group_by = columns

    def _extract_order_by(self, sql: str, structure: QueryStructure):
        """Extract ORDER BY columns from the query."""
        order_match = re.search(r'ORDER\s+BY\s+(.+?)$', sql, re.IGNORECASE)
        if not order_match:
            return

        order_clause = order_match.group(1).strip()
        # Remove ASC/DESC
        order_clause = re.sub(r'\s+(ASC|DESC)\s*', ' ', order_clause, flags=re.IGNORECASE)
        columns = [col.strip() for col in order_clause.split(',')]
        structure.order_by = columns

    def _extract_aggregations(self, sql: str, structure: QueryStructure):
        """Extract aggregation functions from the query."""
        for agg in self.AGGREGATION_FUNCTIONS:
            if re.search(rf'\b{agg}\s*\(', sql, re.IGNORECASE):
                structure.aggregations.append(agg.lower())

    def _extract_columns(self, sql: str, structure: QueryStructure):
        """Extract selected columns from the query."""
        # Get text between SELECT and FROM
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE)
        if not select_match:
            return

        select_clause = select_match.group(1).strip()

        # Simple column extraction (doesn't handle all cases)
        if select_clause == '*':
            structure.columns = ['*']
            return

        # Split by comma, but be careful with function calls
        depth = 0
        current = ""
        columns = []

        for char in select_clause:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                columns.append(current.strip())
                current = ""
            else:
                current += char

        if current.strip():
            columns.append(current.strip())

        structure.columns = columns

    def _parse_insert(self, sql: str, structure: QueryStructure):
        """Parse an INSERT query."""
        # Extract table name
        match = re.search(r'INSERT\s+INTO\s+(\w+)', sql, re.IGNORECASE)
        if match:
            structure.tables.append(match.group(1).lower())

    def _parse_update(self, sql: str, structure: QueryStructure):
        """Parse an UPDATE query."""
        # Extract table name
        match = re.search(r'UPDATE\s+(\w+)', sql, re.IGNORECASE)
        if match:
            structure.tables.append(match.group(1).lower())

        # Extract WHERE conditions
        self._extract_where_conditions(sql, structure)

    def _parse_delete(self, sql: str, structure: QueryStructure):
        """Parse a DELETE query."""
        # Extract table name
        match = re.search(r'DELETE\s+FROM\s+(\w+)', sql, re.IGNORECASE)
        if match:
            structure.tables.append(match.group(1).lower())

        # Extract WHERE conditions
        self._extract_where_conditions(sql, structure)


def analyze_query_with_metadata(sql: str, table_metadata: Dict[str, Any]) -> Tuple[QueryStructure, List[str]]:
    """
    Analyze a query and generate enhanced MeTTa atoms including table metadata.

    Args:
        sql: The SQL query to analyze
        table_metadata: Dictionary of table information

    Returns:
        Tuple of (QueryStructure, List of MeTTa atoms)
    """
    analyzer = QueryAnalyzer()
    structure = analyzer.analyze(sql)

    atoms = structure.to_metta_atoms()

    # Add table metadata atoms
    for table_name, meta in table_metadata.items():
        row_count = meta.get('row_count', 0)
        atoms.append(f"(table-row-count {table_name} {row_count})")

        indexes = meta.get('indexes', [])
        for idx in indexes:
            atoms.append(f"(table-has-index {table_name} \"{idx}\")")

        # Check for specific index conditions
        if not meta.get('has_index_on_email', True):
            atoms.append(f"(table-missing-index {table_name} email)")
        if not meta.get('has_index_on_user_id', True):
            atoms.append(f"(table-missing-index {table_name} user_id)")
        if not meta.get('has_covering_index', True):
            atoms.append(f"(table-missing-covering-index {table_name})")

    return structure, atoms
