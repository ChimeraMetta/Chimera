"""Translates parsed NL queries into MeTTa expressions.

Maps each QueryIntent to MeTTa functional rule calls, builds QueryPlans
with result_key annotations, and tracks translation accuracy.

The generated queries call rules defined in cybersecurity_ontology.metta.
Adding new facts to the ontology automatically makes them queryable —
no changes to this file are needed.
"""

from typing import List, Dict
from cybersecurity_query.models import (
    ParsedQuery, QueryIntent, QueryStep, QueryPlan
)


class MeTTaQueryGenerator:
    """Generates MeTTa queries from parsed NL queries."""

    def __init__(self):
        self.translation_log: List[Dict] = []

    def generate(self, parsed: ParsedQuery) -> QueryPlan:
        """Generate a MeTTa query plan from a parsed query."""
        plan = self._build_plan(parsed)
        self._log_translation(parsed, plan)
        return plan

    def _build_plan(self, parsed: ParsedQuery) -> QueryPlan:
        """Build a query plan based on intent and entities."""
        intent = parsed.intent
        entities = parsed.entities
        threats = entities.get("threats", [])
        cves = entities.get("cves", [])
        software = entities.get("software", [])
        severity = entities.get("severity", [])
        vectors = entities.get("vectors", [])

        if intent == QueryIntent.THREAT_LOOKUP:
            return self._plan_threat_lookup(threats, cves)
        elif intent == QueryIntent.MITIGATION_ADVICE:
            return self._plan_mitigation_advice(threats, cves, software)
        elif intent == QueryIntent.VULNERABILITY_CHECK:
            return self._plan_vulnerability_check(cves, threats)
        elif intent == QueryIntent.RELATIONSHIP_QUERY:
            return self._plan_relationship_query(threats, vectors, software)
        elif intent == QueryIntent.SEVERITY_ASSESSMENT:
            return self._plan_severity_assessment(threats, severity)
        else:
            return self._plan_fallback(parsed)

    # --- Intent-specific plan builders ---

    def _plan_threat_lookup(self, threats: List[str], cves: List[str]) -> QueryPlan:
        """Build plan for threat_lookup intent.

        Uses MeTTa rules: vectors-for-threat, targets-for-threat,
        kill-chain-next, cves-enabling-threat, mitigations-for-threat.
        """
        steps = []
        target = threats[0] if threats else None

        if not target and cves:
            steps.append(QueryStep(
                template='!(threats-from-cve "{cve}")',
                bindings={"cve": cves[0]},
                collect_variable="$threat",
                description=f"Find threats enabled by {cves[0]}",
                result_key="threats",
            ))
            return QueryPlan(steps=steps, intent=QueryIntent.THREAT_LOOKUP,
                             description=f"Look up threats from {cves[0]}")

        if target:
            # Description (direct match on base fact)
            steps.append(QueryStep(
                template='!(match &self (threat-description {threat} $d) $d)',
                bindings={"threat": target},
                collect_variable="$d",
                description=f"Get description of {target}",
                result_key="description",
            ))
            # Severity (direct match on base fact)
            steps.append(QueryStep(
                template='!(match &self (threat-severity {threat} $s) $s)',
                bindings={"threat": target},
                collect_variable="$s",
                description=f"Get severity of {target}",
                result_key="severity",
            ))
            # Attack vectors (calls deductive rule)
            steps.append(QueryStep(
                template='!(vectors-for-threat {threat})',
                bindings={"threat": target},
                collect_variable="$vec",
                description=f"Get attack vectors for {target}",
                result_key="vectors",
            ))
            # Targets (calls deductive rule — returns each target individually)
            steps.append(QueryStep(
                template='!(targets-for-threat {threat})',
                bindings={"threat": target},
                collect_variable="$target",
                description=f"Get targets of {target}",
                result_key="targets",
            ))
            # Kill chain next (calls deductive rule)
            steps.append(QueryStep(
                template='!(kill-chain-next {threat})',
                bindings={"threat": target},
                collect_variable="$next",
                description=f"Get kill chain successors of {target}",
                result_key="kill_chain",
            ))
            # Related CVEs (calls deductive rule)
            steps.append(QueryStep(
                template='!(cves-enabling-threat {threat})',
                bindings={"threat": target},
                collect_variable="$cve",
                description=f"Get CVEs enabling {target}",
                result_key="cves",
            ))
            # Mitigations (calls deductive rule)
            steps.append(QueryStep(
                template='!(mitigations-for-threat {threat})',
                bindings={"threat": target},
                collect_variable="$mit",
                description=f"Get mitigations for {target}",
                result_key="mitigations",
            ))

        return QueryPlan(steps=steps, intent=QueryIntent.THREAT_LOOKUP,
                         description=f"Comprehensive lookup of {target or 'threats'}")

    def _plan_mitigation_advice(self, threats: List[str], cves: List[str],
                                software: List[str]) -> QueryPlan:
        """Build plan for mitigation_advice intent.

        For software: uses all-mitigations-for-software (3-hop deduction in MeTTa).
        For CVEs: uses mitigations-for-cve.
        For threats: uses mitigations-for-threat.
        """
        steps = []

        if software:
            sw = software[0]
            # Single MeTTa rule call does the full multi-hop reasoning:
            # software -> CVE -> (CVE-specific mitigations + threat-type mitigations)
            steps.append(QueryStep(
                template='!(all-mitigations-for-software "{software}")',
                bindings={"software": sw},
                collect_variable="$mitigation",
                description=f"Multi-hop mitigations for {sw}",
                result_key="mitigations",
            ))
        elif cves:
            cve = cves[0]
            # Comprehensive CVE mitigations (CVE-specific + threat-type)
            steps.append(QueryStep(
                template='!(comprehensive-cve-mitigations "{cve}")',
                bindings={"cve": cve},
                collect_variable="$mitigation",
                description=f"Comprehensive mitigations for {cve}",
                result_key="mitigations",
            ))
        elif threats:
            threat = threats[0]
            steps.append(QueryStep(
                template='!(mitigations-for-threat {threat})',
                bindings={"threat": threat},
                collect_variable="$name",
                description=f"Mitigations for {threat}",
                result_key="mitigations",
            ))

        return QueryPlan(steps=steps, intent=QueryIntent.MITIGATION_ADVICE,
                         description=f"Mitigation advice for {threats or cves or software}")

    def _plan_vulnerability_check(self, cves: List[str], threats: List[str]) -> QueryPlan:
        """Build plan for vulnerability_check intent.

        Uses rules: software-from-cve, threats-from-cve, mitigations-for-cve.
        """
        steps = []

        if cves:
            cve = cves[0]
            # Vulnerability details (direct match on base fact)
            steps.append(QueryStep(
                template='!(match &self (vulnerability "{cve}" $name $desc $sev) ($name $desc $sev))',
                bindings={"cve": cve},
                collect_variable="($name $desc $sev)",
                description=f"Get details of {cve}",
                result_key="details",
            ))
            # Affected software (calls deductive rule)
            steps.append(QueryStep(
                template='!(software-from-cve "{cve}")',
                bindings={"cve": cve},
                collect_variable="$software",
                description=f"Get software affected by {cve}",
                result_key="software",
            ))
            # Enabled threats (calls deductive rule)
            steps.append(QueryStep(
                template='!(threats-from-cve "{cve}")',
                bindings={"cve": cve},
                collect_variable="$threat",
                description=f"Get threats enabled by {cve}",
                result_key="threats",
            ))
            # Mitigations (calls deductive rule)
            steps.append(QueryStep(
                template='!(mitigations-for-cve "{cve}")',
                bindings={"cve": cve},
                collect_variable="$mitigation",
                description=f"Get mitigations for {cve}",
                result_key="mitigations",
            ))

        return QueryPlan(steps=steps, intent=QueryIntent.VULNERABILITY_CHECK,
                         description=f"Vulnerability check for {cves}")

    def _plan_relationship_query(self, threats: List[str], vectors: List[str],
                                 software: List[str]) -> QueryPlan:
        """Build plan for relationship_query intent.

        Uses rules: kill-chain-next, kill-chain-previous, kill-chain-reachable,
        vectors-for-threat, threats-using-vector, threats-for-software.
        """
        steps = []

        if threats:
            threat = threats[0]
            # Direct successors in kill chain
            steps.append(QueryStep(
                template='!(kill-chain-next {threat})',
                bindings={"threat": threat},
                collect_variable="$next",
                description=f"What {threat} can directly lead to",
                result_key="leads_to",
            ))
            # Direct predecessors in kill chain
            steps.append(QueryStep(
                template='!(kill-chain-previous {threat})',
                bindings={"threat": threat},
                collect_variable="$prev",
                description=f"What leads to {threat}",
                result_key="led_by",
            ))
            # Transitive reachability (multi-hop deduction in MeTTa)
            steps.append(QueryStep(
                template='!(kill-chain-reachable {threat})',
                bindings={"threat": threat},
                collect_variable="$reachable",
                description=f"Threats transitively reachable from {threat}",
                result_key="reachable",
            ))
            # Attack vectors
            steps.append(QueryStep(
                template='!(vectors-for-threat {threat})',
                bindings={"threat": threat},
                collect_variable="$vec",
                description=f"Attack vectors for {threat}",
                result_key="vectors",
            ))

        if vectors:
            vector = vectors[0]
            steps.append(QueryStep(
                template='!(threats-using-vector {vector})',
                bindings={"vector": vector},
                collect_variable="$threat",
                description=f"Threats using {vector} vector",
                result_key="vector_threats",
            ))

        if software:
            sw = software[0]
            steps.append(QueryStep(
                template='!(threats-for-software "{software}")',
                bindings={"software": sw},
                collect_variable="$threat",
                description=f"Threats affecting {sw}",
                result_key="software_threats",
            ))

        return QueryPlan(steps=steps, intent=QueryIntent.RELATIONSHIP_QUERY,
                         description=f"Relationships for {threats or vectors or software}")

    def _plan_severity_assessment(self, threats: List[str], severity: List[str]) -> QueryPlan:
        """Build plan for severity_assessment intent.

        Uses rule: threats-by-severity.
        """
        steps = []

        if threats:
            threat = threats[0]
            steps.append(QueryStep(
                template='!(match &self (threat-severity {threat} $sev) $sev)',
                bindings={"threat": threat},
                collect_variable="$sev",
                description=f"Get severity of {threat}",
                result_key="severity",
            ))
        elif severity:
            sev = severity[0]
            steps.append(QueryStep(
                template='!(threats-by-severity {severity})',
                bindings={"severity": sev},
                collect_variable="$threat",
                description=f"List {sev} severity threats",
                result_key="threats",
            ))
        else:
            # Default: list critical threats
            steps.append(QueryStep(
                template='!(threats-by-severity critical)',
                bindings={},
                collect_variable="$threat",
                description="List critical severity threats",
                result_key="threats",
            ))

        return QueryPlan(steps=steps, intent=QueryIntent.SEVERITY_ASSESSMENT,
                         description=f"Severity assessment for {threats or severity or ['all']}")

    def _plan_fallback(self, parsed: ParsedQuery) -> QueryPlan:
        """Fallback plan when intent is unknown."""
        entities = parsed.entities

        if entities.get("threats"):
            return self._plan_threat_lookup(entities["threats"], [])
        elif entities.get("cves"):
            return self._plan_vulnerability_check(entities["cves"], [])
        elif entities.get("software"):
            return self._plan_mitigation_advice([], [], entities["software"])

        return QueryPlan(steps=[], intent=QueryIntent.UNKNOWN,
                         description="No recognized query pattern")

    # --- Utility ---

    def render_query(self, step: QueryStep) -> str:
        """Render a QueryStep template with its bindings into a MeTTa query string."""
        query = step.template
        for key, value in step.bindings.items():
            query = query.replace("{" + key + "}", value)
        return query

    def _log_translation(self, parsed: ParsedQuery, plan: QueryPlan):
        """Log the NL -> MeTTa translation for metrics."""
        queries = [self.render_query(step) for step in plan.steps]
        self.translation_log.append({
            "raw_query": parsed.raw_query,
            "intent": parsed.intent.value,
            "entities": parsed.entities,
            "metta_queries": queries,
            "confidence": parsed.confidence,
        })

    def get_translation_accuracy(self) -> float:
        """Calculate translation accuracy based on logged translations."""
        if not self.translation_log:
            return 0.0
        successful = sum(
            1 for t in self.translation_log
            if t["confidence"] > 0.5 and t["metta_queries"]
        )
        return successful / len(self.translation_log)
