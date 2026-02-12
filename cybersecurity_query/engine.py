"""Main orchestrator for the cybersecurity threat NL query system.

CyberSecurityQueryEngine is the primary SDK entry point. It accepts a MeTTa
space/instance (injected from CLI or tests) and coordinates parsing, query
generation, reasoning, formatting, and rating.

A shared ``OntologyReader`` is created once and passed to both the NL parser
(for KB-based entity resolution) and the reasoning engine (for query
evaluation fallback), keeping the ``.metta`` file as the single source of
truth.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from cybersecurity_query.models import (
    QueryIntent, QueryResult, QueryMetrics, ParsedQuery,
    ThreatInfo, VulnerabilityInfo, MitigationInfo, SeverityAssessment,
)
from cybersecurity_query.ontology_reader import OntologyReader, _ONTOLOGY_PATH
from cybersecurity_query.nl_parser import NLParser
from cybersecurity_query.metta_query_generator import MeTTaQueryGenerator
from cybersecurity_query.reasoning_engine import ReasoningEngine
from cybersecurity_query.response_formatter import ResponseFormatter
from cybersecurity_query.rating_system import RatingSystem

logger = logging.getLogger("engine")


class CyberSecurityQueryEngine:
    """Main SDK entry point for the cybersecurity NL query system."""

    def __init__(self, metta_instance=None, metta_space=None):
        """Initialize with optional MeTTa instance/space for symbolic reasoning.

        Args:
            metta_instance: A hyperon.MeTTa instance (for .run() calls).
            metta_space: A hyperon GroundingSpace (for atom storage).
        """
        ontology_reader = self._init_ontology_reader()
        self.parser = NLParser(ontology_reader=ontology_reader)
        self.generator = MeTTaQueryGenerator()
        self.reasoning = ReasoningEngine(
            metta_instance=metta_instance,
            metta_space=metta_space,
            ontology_reader=ontology_reader,
        )
        self.formatter = ResponseFormatter()
        self.rating_system = RatingSystem()

    @staticmethod
    def _init_ontology_reader() -> Optional[OntologyReader]:
        """Create a shared OntologyReader if the ontology file exists."""
        if os.path.exists(_ONTOLOGY_PATH):
            try:
                return OntologyReader(_ONTOLOGY_PATH)
            except Exception as e:
                logger.warning("Failed to load ontology reader: %s", e)
        return None

    def query(self, nl_text: str) -> QueryResult:
        """Process a natural language query end-to-end.

        Pipeline: parse -> generate MeTTa -> execute -> format
        """
        # Step 1: Parse NL
        parsed = self.parser.parse(nl_text)

        # Step 2: Generate MeTTa query plan
        plan = self.generator.generate(parsed)

        # Step 3: Execute reasoning
        result = self.reasoning.execute_plan(plan, parsed)

        # Step 4: Format response
        result.formatted_response = self.formatter.format(result)

        return result

    def query_with_intent(self, intent: QueryIntent,
                          entities: Dict[str, List[str]]) -> QueryResult:
        """Query with explicit intent and entities (bypasses NL parsing)."""
        parsed = ParsedQuery(
            intent=intent,
            entities=entities,
            raw_query=f"[SDK] intent={intent.value} entities={entities}",
            confidence=1.0,
        )
        plan = self.generator.generate(parsed)
        result = self.reasoning.execute_plan(plan, parsed)
        result.formatted_response = self.formatter.format(result)
        return result

    def get_threat_info(self, threat_name: str) -> QueryResult:
        """Convenience: look up a specific threat."""
        return self.query_with_intent(
            QueryIntent.THREAT_LOOKUP,
            {"threats": [threat_name]},
        )

    def get_mitigations(self, threat_name: str) -> QueryResult:
        """Convenience: get mitigations for a threat."""
        return self.query_with_intent(
            QueryIntent.MITIGATION_ADVICE,
            {"threats": [threat_name]},
        )

    def get_vulnerability(self, cve_id: str) -> QueryResult:
        """Convenience: look up a CVE."""
        return self.query_with_intent(
            QueryIntent.VULNERABILITY_CHECK,
            {"cves": [cve_id]},
        )

    def assess_severity(self, threat_name: str) -> QueryResult:
        """Convenience: assess severity of a threat."""
        return self.query_with_intent(
            QueryIntent.SEVERITY_ASSESSMENT,
            {"threats": [threat_name]},
        )

    def trace_attack_chain(self, threat_name: str) -> QueryResult:
        """Convenience: trace attack chain relationships for a threat."""
        return self.query_with_intent(
            QueryIntent.RELATIONSHIP_QUERY,
            {"threats": [threat_name]},
        )

    def rate_response(self, result: QueryResult) -> Optional[Any]:
        """Collect interactive rating for a query result."""
        return self.rating_system.collect_rating_interactive(result)

    def get_metrics(self) -> QueryMetrics:
        """Get accuracy and rating metrics."""
        translation_accuracy = self.generator.get_translation_accuracy()
        return self.rating_system.get_metrics(translation_accuracy)
