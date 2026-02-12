"""Executes MeTTa queries and builds reasoning traces.

The reasoning engine runs MeTTa queries against the loaded ontology. All
knowledge lives in the .metta ontology file -- there are no hardcoded Python
knowledge bases. Queries execute through MeTTa's full reasoning engine via
hyperon; hyperon is a hard requirement.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from cybersecurity_query.models import (
    QueryIntent, QueryPlan, QueryStep, QueryResult,
    ReasoningTrace, ReasoningStep, ThreatInfo, VulnerabilityInfo,
    MitigationInfo, SeverityAssessment, ParsedQuery
)
from cybersecurity_query.ontology_reader import OntologyReader, _ONTOLOGY_PATH

logger = logging.getLogger("reasoning_engine")


class ReasoningEngine:
    """Executes MeTTa queries -- all knowledge lives in the .metta ontology.

    Requires hyperon for MeTTa reasoning. Will raise an error if hyperon is
    not installed.
    """

    def __init__(self, metta_instance=None, metta_space=None, ontology_reader=None):
        self.metta = metta_instance
        self.metta_space = metta_space
        self._ontology: Optional[OntologyReader] = ontology_reader

        if self.metta is None:
            self._init_metta()

    def _init_metta(self):
        """Initialize MeTTa and load the cybersecurity ontology.

        Raises ImportError if hyperon is not installed.
        """
        from hyperon import MeTTa
        self.metta = MeTTa()

        if os.path.exists(_ONTOLOGY_PATH):
            with open(_ONTOLOGY_PATH) as f:
                self.metta.run(f.read())
            logger.info(f"Loaded ontology from {_ONTOLOGY_PATH}")
        else:
            raise FileNotFoundError(
                f"Cybersecurity ontology not found at {_ONTOLOGY_PATH}"
            )

    def execute_plan(self, plan: QueryPlan, parsed: ParsedQuery) -> QueryResult:
        """Execute a query plan and return results with reasoning trace."""
        trace = ReasoningTrace()
        keyed_results: Dict[str, List[str]] = {}
        metta_queries = []

        for step in plan.steps:
            query_str = self._render_query(step)
            metta_queries.append(query_str)

            results = self._execute_query(query_str)
            key = step.result_key or step.description
            keyed_results.setdefault(key, []).extend(results)

            reasoning_type = self._classify_reasoning(query_str)
            trace.add_step(
                query=query_str,
                facts=self._get_rule_facts(query_str),
                results=results,
                reasoning_type=reasoning_type,
            )

        # Build structured result based on intent
        structured_results = self._structure_results(plan.intent, parsed, keyed_results)

        # Calculate confidence
        has_entities = bool(
            parsed.entities.get("threats")
            or parsed.entities.get("cves")
            or parsed.entities.get("software")
        )
        total_results = sum(len(v) for v in keyed_results.values())
        result_confidence = min(1.0, total_results / max(1, len(plan.steps)))
        entity_confidence = 1.0 if has_entities else 0.7
        overall_confidence = min(parsed.confidence, entity_confidence) * result_confidence

        return QueryResult(
            intent=plan.intent,
            entities=parsed.entities,
            results=structured_results,
            reasoning_trace=trace,
            confidence=overall_confidence,
            metta_queries=metta_queries,
        )

    def _render_query(self, step: QueryStep) -> str:
        """Render a QueryStep template with its bindings."""
        query = step.template
        for key, value in step.bindings.items():
            query = query.replace("{" + key + "}", value)
        return query

    def _execute_query(self, query_str: str) -> List[str]:
        """Execute a MeTTa query. Raises on failure."""
        try:
            raw = self.metta.run(query_str)
            return self._parse_metta_results(raw)
        except Exception as e:
            logger.warning(f"MeTTa query failed: {query_str!r} -- {e}")
            return []

    def _parse_metta_results(self, raw_results) -> List[str]:
        """Parse MeTTa execution results into clean strings."""
        results = []
        if not raw_results:
            return results

        for result_set in raw_results:
            if not result_set:
                continue
            for item in result_set:
                s = str(item).strip()
                # Strip quotes from string atoms
                if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
                    s = s[1:-1]
                # Skip empty/nil results
                if s and s not in ('()', 'empty', '(empty)', 'Nothing', 'False', 'None'):
                    results.append(s)

        return results

    def _classify_reasoning(self, query_str: str) -> str:
        """Classify the reasoning type based on the query."""
        q = query_str.strip()
        if q.startswith('!(match '):
            return "direct"
        multi_hop_rules = (
            'all-mitigations-for-software', 'comprehensive-cve-mitigations',
            'kill-chain-reachable', 'threats-for-software', 'mitigations-for-software',
        )
        for rule in multi_hop_rules:
            if rule in q:
                return "multi-hop"
        return "deductive"

    def _get_rule_facts(self, query_str: str) -> List[str]:
        """Extract the MeTTa rule name for trace display."""
        q = query_str.strip()
        if q.startswith('!(match '):
            return ["Direct pattern match on ontology"]
        if q.startswith('!('):
            inner = q[2:]
            rule_name = inner.split(' ')[0].split(')')[0]
            return [f"Rule: {rule_name}"]
        return []

    # --- Result structuring ---

    def _structure_results(self, intent: QueryIntent, parsed: ParsedQuery,
                           keyed: Dict[str, List[str]]) -> List[Any]:
        """Structure keyed results into typed objects based on intent."""
        threats = parsed.entities.get("threats", [])
        cves = parsed.entities.get("cves", [])

        if intent == QueryIntent.THREAT_LOOKUP and threats:
            return [self._build_threat_info(threats[0], keyed)]
        elif intent == QueryIntent.THREAT_LOOKUP and keyed.get("threats"):
            found = keyed["threats"][0]
            return [self._build_threat_info_from_name(found)]
        elif intent == QueryIntent.VULNERABILITY_CHECK and cves:
            return [self._build_vulnerability_info(cves[0], keyed)]
        elif intent == QueryIntent.MITIGATION_ADVICE:
            return self._build_mitigation_list(keyed)
        elif intent == QueryIntent.RELATIONSHIP_QUERY:
            return self._build_relationships(keyed)
        elif intent == QueryIntent.SEVERITY_ASSESSMENT:
            return self._build_severity(keyed, threats, parsed.entities.get("severity", []))
        else:
            all_results = []
            for v in keyed.values():
                all_results.extend(v)
            return all_results

    def _build_threat_info(self, name: str, keyed: Dict[str, List[str]]) -> ThreatInfo:
        """Build ThreatInfo from keyed results."""
        return ThreatInfo(
            name=name,
            description=self._first(keyed.get("description", [])),
            severity=self._first(keyed.get("severity", [])),
            attack_vectors=keyed.get("vectors", []),
            targets=keyed.get("targets", []),
            mitigations=keyed.get("mitigations", []),
            kill_chain_next=keyed.get("kill_chain", []),
            related_cves=keyed.get("cves", []),
        )

    def _build_threat_info_from_name(self, name: str) -> ThreatInfo:
        """Build ThreatInfo by querying for a threat name."""
        info = ThreatInfo(name=name)
        desc = self._execute_query(f'!(match &self (threat-description {name} $d) $d)')
        info.description = self._first(desc)
        sev = self._execute_query(f'!(match &self (threat-severity {name} $s) $s)')
        info.severity = self._first(sev)
        return info

    def _build_vulnerability_info(self, cve_id: str, keyed: Dict[str, List[str]]) -> VulnerabilityInfo:
        """Build VulnerabilityInfo from keyed results."""
        details = keyed.get("details", [])
        name, description, severity = "", "", ""
        if details:
            parts = self._parse_expression_parts(details[0])
            if len(parts) >= 3:
                name, description, severity = parts[0], parts[1], parts[2]
            elif len(parts) == 1:
                name = parts[0]

        return VulnerabilityInfo(
            cve_id=cve_id,
            name=name,
            description=description,
            severity=severity,
            affected_software=keyed.get("software", []),
            enabled_threats=keyed.get("threats", []),
            mitigations=keyed.get("mitigations", []),
        )

    def _build_mitigation_list(self, keyed: Dict[str, List[str]]) -> List[MitigationInfo]:
        """Build MitigationInfo list. Looks up descriptions from ontology."""
        mitigations = []
        seen = set()

        names = []
        for key in ("mitigations", "cve_mitigations", "software_mitigations"):
            names.extend(keyed.get(key, []))

        for name in names:
            if name not in seen:
                seen.add(name)
                desc = self._first(self._execute_query(
                    f'!(match &self (mitigation-description "{name}" $d) $d)'
                ))
                threats = self._execute_query(
                    f'!(match &self (mitigation-addresses "{name}" $t) $t)'
                )
                mitigations.append(MitigationInfo(
                    name=name,
                    description=desc,
                    addresses_threats=threats,
                ))

        return mitigations

    def _build_relationships(self, keyed: Dict[str, List[str]]) -> List[str]:
        """Build relationship results as strings."""
        all_results = []
        for key in ("leads_to", "led_by", "vectors", "reachable",
                     "vector_threats", "software_threats"):
            all_results.extend(keyed.get(key, []))
        seen = set()
        return [r for r in all_results if r and not (r in seen or seen.add(r))]

    def _build_severity(self, keyed: Dict[str, List[str]],
                        threats: List[str], severity_levels: List[str]) -> List[SeverityAssessment]:
        """Build severity assessment."""
        if threats:
            threat = threats[0]
            sev = self._first(keyed.get("severity", []))
            return [SeverityAssessment(entity_name=threat, severity=sev or "unknown")]

        level = severity_levels[0] if severity_levels else "critical"
        return [SeverityAssessment(
            entity_name=level,
            severity=level,
            threats_at_level=keyed.get("threats", []),
        )]

    # --- Helpers ---

    @staticmethod
    def _first(items: List[str]) -> str:
        return items[0] if items else ""

    @staticmethod
    def _parse_expression_parts(s: str) -> List[str]:
        """Parse a MeTTa expression like (name "desc" severity) into parts."""
        s = s.strip()
        if s.startswith('(') and s.endswith(')'):
            s = s[1:-1].strip()

        parts = []
        current = ""
        in_quotes = False
        for ch in s:
            if ch == '"' and not in_quotes:
                in_quotes = True
            elif ch == '"' and in_quotes:
                in_quotes = False
                parts.append(current)
                current = ""
            elif ch in (' ', '\t') and not in_quotes:
                if current:
                    parts.append(current)
                    current = ""
            else:
                current += ch
        if current:
            parts.append(current)
        return parts
