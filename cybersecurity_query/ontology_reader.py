"""Reads and indexes the MeTTa cybersecurity ontology file.

OntologyReader is the single-file parser for the .metta ontology. It is shared
by both the NL parser (for entity vocabulary extraction) and the reasoning
engine (for query evaluation). The .metta file remains the single source of
truth -- no hardcoded Python knowledge bases.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ontology_reader")

# Path to cybersecurity ontology relative to this file
_ONTOLOGY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metta", "cybersecurity_ontology.metta"
)


class OntologyReader:
    """Reads facts directly from the .metta ontology file.

    Used as fallback when hyperon is not installed. Parses the same ontology
    file that MeTTa uses, keeping one source of truth -- no hardcoded data.
    Supports basic pattern matching, rule composition, and entity vocabulary
    extraction for the NL parser.
    """

    def __init__(self, path: str):
        self._facts: Dict[str, List[Tuple[str, ...]]] = {}
        self._parse_file(path)

    def _parse_file(self, path: str):
        """Extract fact atoms from the .metta file.

        Parses both regular facts like ``(threat-severity ransomware critical)``
        and type declarations like ``(: ransomware ThreatType)``. Only rule
        definitions starting with ``(=`` are skipped.
        """
        with open(path) as f:
            for line in f:
                line = line.strip()
                # Skip comments, empty lines, and rules (= ...)
                if not line or line.startswith(';;'):
                    continue
                if line.startswith('(='):
                    continue
                # Parse top-level atoms: (pred arg1 arg2 ...)
                if line.startswith('(') and line.endswith(')'):
                    tokens = self._tokenize(line[1:-1].strip())
                    if tokens and len(tokens) >= 2:
                        pred = tokens[0]
                        args = tuple(tokens[1:])
                        self._facts.setdefault(pred, []).append(args)

    @staticmethod
    def _tokenize(s: str) -> List[str]:
        """Tokenize a MeTTa expression, handling quoted strings."""
        tokens = []
        current = ""
        in_quotes = False
        paren_depth = 0
        for ch in s:
            if ch == '"' and not in_quotes:
                in_quotes = True
            elif ch == '"' and in_quotes:
                in_quotes = False
                tokens.append(current)
                current = ""
            elif ch == '(' and not in_quotes:
                paren_depth += 1
                current += ch
            elif ch == ')' and not in_quotes:
                paren_depth -= 1
                current += ch
                if paren_depth == 0 and current:
                    tokens.append(current)
                    current = ""
            elif ch in (' ', '\t') and not in_quotes and paren_depth == 0:
                if current:
                    tokens.append(current)
                    current = ""
            else:
                current += ch
        if current:
            tokens.append(current)
        return tokens

    def match(self, pred: str, *pattern) -> List[Tuple[str, ...]]:
        """Match facts by predicate + arg pattern. None = wildcard."""
        results = []
        for args in self._facts.get(pred, []):
            if len(args) != len(pattern):
                continue
            if all(p is None or p == a for p, a in zip(pattern, args)):
                results.append(args)
        return results

    # ------------------------------------------------------------------
    # Entity vocabulary extraction (used by KBEntityResolver)
    # ------------------------------------------------------------------

    def get_all_threats(self) -> List[str]:
        """Return all threat names from ``(: X ThreatType)`` declarations."""
        return [args[0] for args in self.match(":", None, "ThreatType")]

    def get_all_attack_vectors(self) -> List[str]:
        """Return all attack vector names from ``(: X AttackVector)``."""
        return [args[0] for args in self.match(":", None, "AttackVector")]

    def get_all_severity_levels(self) -> List[str]:
        """Return all severity levels from ``(: X Severity)``."""
        return [args[0] for args in self.match(":", None, "Severity")]

    def get_all_software(self) -> List[str]:
        """Return all software names from ``(vulnerability-affects $cve $sw)``."""
        seen: Set[str] = set()
        result: List[str] = []
        for args in self._facts.get("vulnerability-affects", []):
            sw = args[1]
            if sw not in seen:
                seen.add(sw)
                result.append(sw)
        return result

    def get_all_cve_ids(self) -> List[str]:
        """Return all CVE IDs from ``(vulnerability $cve ...)``."""
        seen: Set[str] = set()
        result: List[str] = []
        for args in self._facts.get("vulnerability", []):
            cve = args[0]
            if cve not in seen:
                seen.add(cve)
                result.append(cve)
        return result

    def get_cve_names(self) -> Dict[str, str]:
        """Return CVE ID -> common name mapping from vulnerability atoms."""
        mapping: Dict[str, str] = {}
        for args in self._facts.get("vulnerability", []):
            if len(args) >= 2:
                mapping[args[0]] = args[1]
        return mapping

    def get_all_mitigations(self) -> List[str]:
        """Return all mitigation names from ``(mitigation-addresses $name ...)``."""
        seen: Set[str] = set()
        result: List[str] = []
        for args in self._facts.get("mitigation-addresses", []):
            name = args[0]
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    # ------------------------------------------------------------------
    # Query evaluation (used by ReasoningEngine fallback)
    # ------------------------------------------------------------------

    def evaluate(self, query_str: str) -> List[str]:
        """Evaluate a MeTTa query string against parsed ontology facts."""
        q = query_str.strip()

        # Pattern 1: !(match &self (pred args...) return_expr)
        m = re.match(r'^!\(match &self \((.+?)\)\s+(.+)\)$', q)
        if m:
            return self._eval_match(m.group(1), m.group(2))

        # Pattern 2: !(rule-name args...)
        m = re.match(r'^!\((\S+)\s*(.*?)\)$', q)
        if m:
            rule_name = m.group(1)
            args_str = m.group(2).strip()
            args = self._tokenize(args_str) if args_str else []
            return self._eval_rule(rule_name, args)

        return []

    def _eval_match(self, pred_str: str, return_expr: str) -> List[str]:
        """Evaluate a match query against fact atoms."""
        tokens = self._tokenize(pred_str)
        if not tokens:
            return []

        pred = tokens[0]
        args = tokens[1:]

        # Build pattern: None for variables ($var)
        pattern = []
        var_positions = {}
        for i, arg in enumerate(args):
            if arg.startswith('$'):
                pattern.append(None)
                var_positions[arg] = i
            else:
                pattern.append(arg)

        matches = self.match(pred, *pattern)
        return_expr = return_expr.strip()

        # Single variable return: $var
        if return_expr.startswith('$') and return_expr in var_positions:
            pos = var_positions[return_expr]
            return [m[pos] for m in matches]

        # Tuple return: ($var1 $var2 ...)
        if return_expr.startswith('(') and return_expr.endswith(')'):
            ret_tokens = self._tokenize(return_expr[1:-1].strip())
            results = []
            for m in matches:
                parts = []
                for rt in ret_tokens:
                    if rt.startswith('$') and rt in var_positions:
                        parts.append(m[var_positions[rt]])
                    else:
                        parts.append(rt)
                # Format as MeTTa expression
                formatted = '(' + ' '.join(
                    f'"{p}"' if ' ' in p else p for p in parts
                ) + ')'
                results.append(formatted)
            return results

        return []

    def _eval_rule(self, rule_name: str, args: List[str]) -> List[str]:
        """Evaluate a named rule against the ontology."""
        dispatch = {
            'mitigations-for-threat': self._rule_mitigations_for_threat,
            'vectors-for-threat': self._rule_vectors_for_threat,
            'targets-for-threat': self._rule_targets_for_threat,
            'kill-chain-next': self._rule_kill_chain_next,
            'kill-chain-previous': self._rule_kill_chain_previous,
            'kill-chain-reachable': self._rule_kill_chain_reachable,
            'cves-enabling-threat': self._rule_cves_enabling_threat,
            'threats-from-cve': self._rule_threats_from_cve,
            'software-from-cve': self._rule_software_from_cve,
            'mitigations-for-cve': self._rule_mitigations_for_cve,
            'threats-by-severity': self._rule_threats_by_severity,
            'threats-using-vector': self._rule_threats_using_vector,
            'mitigations-for-software': self._rule_mitigations_for_software,
            'all-mitigations-for-software': self._rule_all_mitigations_for_software,
            'comprehensive-cve-mitigations': self._rule_comprehensive_cve_mitigations,
            'threats-for-software': self._rule_threats_for_software,
        }
        fn = dispatch.get(rule_name)
        if fn and args:
            return fn(args[0])
        return []

    # --- Rule implementations (mirror the MeTTa rules in the ontology) ---

    def _rule_mitigations_for_threat(self, threat):
        return [m[0] for m in self.match("mitigation-addresses", None, threat)]

    def _rule_vectors_for_threat(self, threat):
        return [m[1] for m in self.match("threat-uses-vector", threat, None)]

    def _rule_targets_for_threat(self, threat):
        results = []
        for m in self.match("threat-targets", threat, None, None):
            results.append(m[1])  # primary
            results.append(m[2])  # secondary
        return results

    def _rule_kill_chain_next(self, threat):
        return [m[1] for m in self.match("threat-precedes", threat, None)]

    def _rule_kill_chain_previous(self, threat):
        return [m[0] for m in self.match("threat-precedes", None, threat)]

    def _rule_kill_chain_reachable(self, threat):
        """Transitive closure of threat-precedes (up to 3 hops)."""
        reachable = set()
        # 1-hop
        hop1 = [m[1] for m in self.match("threat-precedes", threat, None)]
        reachable.update(hop1)
        # 2-hop
        for t in hop1:
            hop2 = [m[1] for m in self.match("threat-precedes", t, None)]
            reachable.update(hop2)
            # 3-hop
            for t2 in hop2:
                hop3 = [m[1] for m in self.match("threat-precedes", t2, None)]
                reachable.update(hop3)
        return list(reachable)

    def _rule_cves_enabling_threat(self, threat):
        return [m[0] for m in self.match("vulnerability-enables", None, threat)]

    def _rule_threats_from_cve(self, cve):
        return [m[1] for m in self.match("vulnerability-enables", cve, None)]

    def _rule_software_from_cve(self, cve):
        return [m[1] for m in self.match("vulnerability-affects", cve, None)]

    def _rule_mitigations_for_cve(self, cve):
        return [m[1] for m in self.match("cve-mitigated-by", cve, None)]

    def _rule_threats_by_severity(self, severity):
        return [m[0] for m in self.match("threat-severity", None, severity)]

    def _rule_threats_using_vector(self, vector):
        return [m[0] for m in self.match("threat-uses-vector", None, vector)]

    def _rule_mitigations_for_software(self, software):
        """2-hop: software -> CVE -> CVE-specific mitigation."""
        results = set()
        for cve_match in self.match("vulnerability-affects", None, software):
            cve = cve_match[0]
            for mit_match in self.match("cve-mitigated-by", cve, None):
                results.add(mit_match[1])
        return list(results)

    def _rule_all_mitigations_for_software(self, software):
        """3-hop: CVE-specific mitigations + threat-type mitigations."""
        results = set()
        # CVE-specific mitigations
        results.update(self._rule_mitigations_for_software(software))
        # Threat-type mitigations (software -> CVE -> threat -> mitigation)
        for cve_match in self.match("vulnerability-affects", None, software):
            cve = cve_match[0]
            for threat_match in self.match("vulnerability-enables", cve, None):
                threat = threat_match[1]
                for mit_match in self.match("mitigation-addresses", None, threat):
                    results.add(mit_match[0])
        return list(results)

    def _rule_comprehensive_cve_mitigations(self, cve):
        """CVE-specific + threat-type mitigations for a CVE."""
        results = set()
        # CVE-specific
        results.update(self._rule_mitigations_for_cve(cve))
        # Threat-type
        for threat_match in self.match("vulnerability-enables", cve, None):
            threat = threat_match[1]
            for mit_match in self.match("mitigation-addresses", None, threat):
                results.add(mit_match[0])
        return list(results)

    def _rule_threats_for_software(self, software):
        """2-hop: software -> CVE -> threats."""
        results = set()
        for cve_match in self.match("vulnerability-affects", None, software):
            cve = cve_match[0]
            for threat_match in self.match("vulnerability-enables", cve, None):
                results.add(threat_match[1])
        return list(results)
