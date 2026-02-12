"""Fluent SDK API for building cybersecurity queries programmatically.

Usage:
    result = (QueryBuilder(engine)
              .with_intent("threat_lookup")
              .for_threat("ransomware")
              .execute())
"""

from typing import Optional, TYPE_CHECKING

from cybersecurity_query.models import QueryIntent, QueryResult

if TYPE_CHECKING:
    from cybersecurity_query.engine import CyberSecurityQueryEngine


class QueryBuilder:
    """Fluent API for constructing and executing cybersecurity queries."""

    def __init__(self, engine: "CyberSecurityQueryEngine"):
        self._engine = engine
        self._intent: Optional[QueryIntent] = None
        self._threat: Optional[str] = None
        self._cve: Optional[str] = None
        self._software: Optional[str] = None
        self._severity: Optional[str] = None
        self._vector: Optional[str] = None

    def with_intent(self, intent: str) -> "QueryBuilder":
        """Set the query intent explicitly."""
        intent_map = {
            "threat_lookup": QueryIntent.THREAT_LOOKUP,
            "mitigation_advice": QueryIntent.MITIGATION_ADVICE,
            "vulnerability_check": QueryIntent.VULNERABILITY_CHECK,
            "relationship_query": QueryIntent.RELATIONSHIP_QUERY,
            "severity_assessment": QueryIntent.SEVERITY_ASSESSMENT,
        }
        self._intent = intent_map.get(intent, QueryIntent.UNKNOWN)
        return self

    def for_threat(self, threat: str) -> "QueryBuilder":
        """Set the target threat."""
        self._threat = threat
        return self

    def for_cve(self, cve: str) -> "QueryBuilder":
        """Set the target CVE."""
        self._cve = cve
        return self

    def for_software(self, software: str) -> "QueryBuilder":
        """Set the target software."""
        self._software = software
        return self

    def with_severity(self, severity: str) -> "QueryBuilder":
        """Set severity filter."""
        self._severity = severity
        return self

    def with_vector(self, vector: str) -> "QueryBuilder":
        """Set attack vector filter."""
        self._vector = vector
        return self

    def execute(self) -> QueryResult:
        """Execute the constructed query."""
        # Build a natural language query string from the builder state
        query_text = self._build_query_text()

        # If intent is explicitly set, use query_with_intent
        if self._intent and self._intent != QueryIntent.UNKNOWN:
            entities = {}
            if self._threat:
                entities["threats"] = [self._threat]
            if self._cve:
                entities["cves"] = [self._cve]
            if self._software:
                entities["software"] = [self._software]
            if self._severity:
                entities["severity"] = [self._severity]
            if self._vector:
                entities["vectors"] = [self._vector]
            return self._engine.query_with_intent(self._intent, entities)

        # Otherwise use NL query
        return self._engine.query(query_text)

    def _build_query_text(self) -> str:
        """Build a NL query string from builder state."""
        if self._intent == QueryIntent.THREAT_LOOKUP and self._threat:
            return f"What is {self._threat}?"
        elif self._intent == QueryIntent.MITIGATION_ADVICE:
            if self._threat:
                return f"How to protect against {self._threat}?"
            elif self._software:
                return f"What mitigations for {self._software}?"
            elif self._cve:
                return f"What mitigations for {self._cve}?"
        elif self._intent == QueryIntent.VULNERABILITY_CHECK and self._cve:
            return f"What is {self._cve}?"
        elif self._intent == QueryIntent.RELATIONSHIP_QUERY and self._threat:
            return f"What can {self._threat} lead to?"
        elif self._intent == QueryIntent.SEVERITY_ASSESSMENT:
            if self._threat:
                return f"How severe is {self._threat}?"
            elif self._severity:
                return f"What are the {self._severity} threats?"

        # Fallback: describe what we have
        parts = []
        if self._threat:
            parts.append(self._threat)
        if self._cve:
            parts.append(self._cve)
        if self._software:
            parts.append(self._software)
        return " ".join(parts) if parts else "list threats"
