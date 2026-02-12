"""Formats query results into human-readable output with reasoning chains.

Uses colored output via ColoredFormatter patterns from common/logging_utils.py.
"""

from typing import List
from cybersecurity_query.models import (
    QueryResult, QueryIntent, ThreatInfo, VulnerabilityInfo,
    MitigationInfo, SeverityAssessment, ReasoningTrace
)


class ResponseFormatter:
    """Formats QueryResult into structured text output."""

    def format(self, result: QueryResult) -> str:
        """Format a QueryResult into a displayable string."""
        sections = []

        # Header
        sections.append(self._format_header(result))

        # Main content based on intent
        content = self._format_content(result)
        if content:
            sections.append(content)

        # Reasoning trace
        if result.reasoning_trace.steps:
            sections.append(self._format_reasoning_trace(result.reasoning_trace))

        # MeTTa queries used
        if result.metta_queries:
            sections.append(self._format_metta_queries(result.metta_queries))

        # Confidence
        sections.append(f"Confidence: {result.confidence:.0%}")

        return "\n".join(sections)

    def _format_header(self, result: QueryResult) -> str:
        """Format the result header."""
        intent_labels = {
            QueryIntent.THREAT_LOOKUP: "Threat Information",
            QueryIntent.MITIGATION_ADVICE: "Mitigation Advice",
            QueryIntent.VULNERABILITY_CHECK: "Vulnerability Details",
            QueryIntent.RELATIONSHIP_QUERY: "Threat Relationships",
            QueryIntent.SEVERITY_ASSESSMENT: "Severity Assessment",
            QueryIntent.UNKNOWN: "Query Result",
        }
        label = intent_labels.get(result.intent, "Query Result")
        line = "=" * 60
        return f"\n{line}\n  {label}\n{line}"

    def _format_content(self, result: QueryResult) -> str:
        """Format the main content based on intent and result types."""
        if not result.results:
            return "No results found for this query."

        intent = result.intent

        if intent == QueryIntent.THREAT_LOOKUP:
            return self._format_threat_info(result.results)
        elif intent == QueryIntent.VULNERABILITY_CHECK:
            return self._format_vulnerability_info(result.results)
        elif intent == QueryIntent.MITIGATION_ADVICE:
            return self._format_mitigation_list(result.results)
        elif intent == QueryIntent.RELATIONSHIP_QUERY:
            return self._format_relationships(result.results, result.entities)
        elif intent == QueryIntent.SEVERITY_ASSESSMENT:
            return self._format_severity(result.results)
        else:
            return self._format_generic(result.results)

    def _format_threat_info(self, results: List) -> str:
        """Format ThreatInfo objects."""
        lines = []
        for item in results:
            if isinstance(item, ThreatInfo):
                lines.append(f"\n  Threat: {item.name}")
                if item.description:
                    lines.append(f"  Description: {item.description}")
                if item.severity:
                    severity_icon = {"critical": "[!]", "high": "[!]", "medium": "[~]", "low": "[-]"}.get(item.severity, "")
                    lines.append(f"  Severity: {severity_icon} {item.severity.upper()}")
                if item.attack_vectors:
                    lines.append(f"  Attack Vectors: {', '.join(item.attack_vectors)}")
                if item.targets:
                    lines.append(f"  Targets: {', '.join(item.targets)}")
                if item.mitigations:
                    lines.append(f"  Recommended Mitigations:")
                    for m in item.mitigations:
                        lines.append(f"    - {m}")
                if item.kill_chain_next:
                    lines.append(f"  Can Lead To: {', '.join(item.kill_chain_next)}")
                if item.related_cves:
                    lines.append(f"  Related CVEs: {', '.join(item.related_cves)}")
            else:
                lines.append(f"  {item}")
        return "\n".join(lines)

    def _format_vulnerability_info(self, results: List) -> str:
        """Format VulnerabilityInfo objects."""
        lines = []
        for item in results:
            if isinstance(item, VulnerabilityInfo):
                lines.append(f"\n  CVE: {item.cve_id}")
                if item.name:
                    lines.append(f"  Name: {item.name}")
                if item.description:
                    lines.append(f"  Description: {item.description}")
                if item.severity:
                    lines.append(f"  Severity: {item.severity.upper()}")
                if item.affected_software:
                    lines.append(f"  Affected Software: {', '.join(item.affected_software)}")
                if item.enabled_threats:
                    lines.append(f"  Enables Threats: {', '.join(item.enabled_threats)}")
                if item.mitigations:
                    lines.append(f"  Mitigations:")
                    for m in item.mitigations:
                        lines.append(f"    - {m}")
            else:
                lines.append(f"  {item}")
        return "\n".join(lines)

    def _format_mitigation_list(self, results: List) -> str:
        """Format MitigationInfo objects."""
        lines = ["\n  Recommended Mitigations:"]
        for i, item in enumerate(results, 1):
            if isinstance(item, MitigationInfo):
                lines.append(f"\n  {i}. {item.name}")
                if item.description:
                    lines.append(f"     {item.description}")
                if item.addresses_threats:
                    lines.append(f"     Addresses: {', '.join(item.addresses_threats)}")
            else:
                lines.append(f"  {i}. {item}")
        return "\n".join(lines)

    def _format_relationships(self, results: List, entities: dict) -> str:
        """Format relationship query results."""
        lines = []
        threats = entities.get("threats", [])
        if threats:
            lines.append(f"\n  Relationships for: {threats[0]}")

        for item in results:
            if isinstance(item, str) and item.strip():
                lines.append(f"    -> {item}")
        return "\n".join(lines) if lines else "  No relationships found."

    def _format_severity(self, results: List) -> str:
        """Format severity assessment results."""
        lines = []
        for item in results:
            if isinstance(item, SeverityAssessment):
                if item.threats_at_level:
                    lines.append(f"\n  {item.severity.upper()} severity threats:")
                    for t in item.threats_at_level:
                        lines.append(f"    - {t}")
                else:
                    lines.append(f"\n  {item.entity_name}: {item.severity.upper()}")
            else:
                lines.append(f"  {item}")
        return "\n".join(lines)

    def _format_generic(self, results: List) -> str:
        """Format generic results."""
        lines = []
        for item in results:
            lines.append(f"  - {item}")
        return "\n".join(lines)

    def _format_reasoning_trace(self, trace: ReasoningTrace) -> str:
        """Format the reasoning chain with evidence citations."""
        lines = ["\n  Reasoning Chain:"]
        for i, step in enumerate(trace.steps, 1):
            lines.append(f"  Step {i} ({step.reasoning_type}):")
            lines.append(f"    Query: {step.query_executed}")
            if step.facts_used:
                lines.append(f"    Evidence: {', '.join(step.facts_used[:3])}")
            if step.results:
                display = step.results[:5]
                lines.append(f"    Results: {', '.join(display)}")
                if len(step.results) > 5:
                    lines.append(f"    ... and {len(step.results) - 5} more")
        return "\n".join(lines)

    def _format_metta_queries(self, queries: List[str]) -> str:
        """Format the MeTTa queries used."""
        lines = ["\n  MeTTa Queries Executed:"]
        for q in queries:
            lines.append(f"    {q}")
        return "\n".join(lines)
