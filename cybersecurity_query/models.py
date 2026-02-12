"""Data classes for the cybersecurity threat NL query system."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import uuid
import time


class QueryIntent(Enum):
    """Supported query intents."""
    THREAT_LOOKUP = "threat_lookup"
    MITIGATION_ADVICE = "mitigation_advice"
    VULNERABILITY_CHECK = "vulnerability_check"
    RELATIONSHIP_QUERY = "relationship_query"
    SEVERITY_ASSESSMENT = "severity_assessment"
    UNKNOWN = "unknown"


@dataclass
class ParsedQuery:
    """Result of NL parsing: intent + extracted entities."""
    intent: QueryIntent
    entities: Dict[str, List[str]]
    raw_query: str
    confidence: float


@dataclass
class QueryStep:
    """A single step in a multi-hop query plan."""
    template: str
    bindings: Dict[str, str]
    collect_variable: str
    description: str = ""
    result_key: str = ""  # identifies result type: "description", "severity", "vectors", etc.


@dataclass
class QueryPlan:
    """Multi-step query plan for complex queries."""
    steps: List[QueryStep]
    intent: QueryIntent
    description: str = ""


@dataclass
class ReasoningStep:
    """A single step in the reasoning trace."""
    query_executed: str
    facts_used: List[str]
    results: List[str]
    reasoning_type: str  # "direct", "multi-hop", "fallback"


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a query."""
    steps: List[ReasoningStep] = field(default_factory=list)

    def add_step(self, query: str, facts: List[str], results: List[str], reasoning_type: str):
        self.steps.append(ReasoningStep(
            query_executed=query,
            facts_used=facts,
            results=results,
            reasoning_type=reasoning_type
        ))


@dataclass
class ThreatInfo:
    """Information about a specific threat."""
    name: str
    description: str = ""
    severity: str = ""
    attack_vectors: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    kill_chain_next: List[str] = field(default_factory=list)
    related_cves: List[str] = field(default_factory=list)


@dataclass
class VulnerabilityInfo:
    """Information about a specific vulnerability."""
    cve_id: str
    name: str = ""
    description: str = ""
    severity: str = ""
    affected_software: List[str] = field(default_factory=list)
    enabled_threats: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)


@dataclass
class MitigationInfo:
    """Information about a specific mitigation."""
    name: str
    description: str = ""
    addresses_threats: List[str] = field(default_factory=list)
    addresses_cves: List[str] = field(default_factory=list)


@dataclass
class SeverityAssessment:
    """Severity assessment result."""
    entity_name: str
    severity: str
    threats_at_level: List[str] = field(default_factory=list)


@dataclass
class QueryResult:
    """Complete result of a cybersecurity query."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    intent: QueryIntent = QueryIntent.UNKNOWN
    entities: Dict[str, List[str]] = field(default_factory=dict)
    results: List[Any] = field(default_factory=list)
    reasoning_trace: ReasoningTrace = field(default_factory=ReasoningTrace)
    confidence: float = 0.0
    formatted_response: str = ""
    metta_queries: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class QueryRating:
    """User rating for a query response."""
    query_id: str
    rating: int  # 1-5
    query_text: str
    intent: str
    timestamp: float = field(default_factory=time.time)
    feedback: str = ""


@dataclass
class QueryMetrics:
    """Accuracy metrics for the query system."""
    total_queries: int = 0
    intent_accuracy: float = 0.0
    translation_accuracy: float = 0.0
    retrieval_accuracy: float = 0.0
    average_rating: float = 0.0
    rated_queries: int = 0
    low_rated_queries: int = 0  # rating <= 2
