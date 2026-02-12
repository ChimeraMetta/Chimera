"""Natural language parser for cybersecurity threat queries.

Thin orchestrator that delegates to:
- **CVEResolver** + **KBEntityResolver** for entity extraction (from MeTTa KB)
- **EmbeddingIntentClassifier** for intent classification (sentence embeddings)

Entity vocabulary comes from the MeTTa ontology -- adding a new threat to the
``.metta`` file makes it automatically discoverable. Intent classification uses
sentence-transformer embeddings to handle arbitrary paraphrased queries.

Requires: ``sentence-transformers>=2.2.0``
"""

import logging
from typing import Dict, List

from cybersecurity_query.models import ParsedQuery, QueryIntent
from cybersecurity_query.entity_resolver import CVEResolver, KBEntityResolver
from cybersecurity_query.intent_classifier import EmbeddingIntentClassifier

logger = logging.getLogger("nl_parser")


class NLParser:
    """Orchestrates entity extraction and intent classification."""

    def __init__(self, ontology_reader):
        self._cve_resolver = CVEResolver()
        self._entity_resolver = KBEntityResolver(ontology_reader)
        self._intent_classifier = EmbeddingIntentClassifier()

    def parse(self, query: str) -> ParsedQuery:
        """Parse a natural language query into structured form."""
        # Extract entities (CVEs always + threats/software/severity/vectors)
        entities = self._entity_resolver.extract_entities(query)
        entities.setdefault("cves", [])
        entities["cves"] = self._cve_resolver.extract(query) + entities.get("cves", [])
        # Deduplicate CVEs while preserving order
        seen = set()
        entities["cves"] = [c for c in entities["cves"] if not (c in seen or seen.add(c))]

        # Classify intent
        intent, confidence = self._intent_classifier.classify(query, entities)

        return ParsedQuery(
            intent=intent,
            entities=entities,
            raw_query=query,
            confidence=confidence,
        )
