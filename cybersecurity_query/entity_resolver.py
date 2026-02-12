"""Entity resolution for cybersecurity NL queries.

Two resolvers work together:

* **CVEResolver** -- always active.  Structural format matching for CVE IDs
  (``CVE-\\d{4}-\\d{4,}``) plus named aliases (log4shell, heartbleed, ...).

* **KBEntityResolver** -- reads entity vocabularies from the MeTTa ontology via
  ``OntologyReader``.  Adding a new threat/vector/software to the ``.metta``
  file makes it automatically discoverable with no Python changes.
"""

import re
import logging
from typing import Dict, List, Set

logger = logging.getLogger("entity_resolver")


class CVEResolver:
    """Resolves CVE identifiers from query text.

    Always active -- CVE regex and named aliases are structural identifiers
    that don't belong in the KB.
    """

    def __init__(self):
        self.cve_pattern = re.compile(r'(CVE-\d{4}-\d{4,})', re.IGNORECASE)
        self.cve_aliases = {
            "log4shell": "CVE-2021-44228",
            "printnightmare": "CVE-2021-34527",
            "print nightmare": "CVE-2021-34527",
            "proxylogon": "CVE-2021-26855",
            "proxy logon": "CVE-2021-26855",
            "eternalblue": "CVE-2017-0144",
            "eternal blue": "CVE-2017-0144",
            "heartbleed": "CVE-2014-0160",
            "rapid reset": "CVE-2023-44487",
            "http/2 rapid reset": "CVE-2023-44487",
        }

    def extract(self, query: str) -> List[str]:
        """Extract CVE IDs from query text (regex + aliases)."""
        cves: List[str] = []
        query_lower = query.lower().strip()

        # 1. Regex extraction
        for match in self.cve_pattern.findall(query):
            cve = match.upper()
            if cve not in cves:
                cves.append(cve)

        # 2. Named alias extraction
        for alias, cve_id in self.cve_aliases.items():
            if alias in query_lower and cve_id not in cves:
                cves.append(cve_id)

        return cves


class KBEntityResolver:
    """Resolves entities by reading vocabulary from the MeTTa ontology.

    Takes an ``OntologyReader`` instance and extracts all known threats,
    attack vectors, severity levels, and software names at init time.
    New entities added to the ``.metta`` file are automatically discoverable.
    """

    # Small abbreviation dicts validated against KB at init
    _THREAT_ABBREVIATIONS = {
        "sqli": "sql-injection",
        "mitm": "man-in-the-middle",
        "dos": "ddos",
        "privesc": "privilege-escalation",
        "xss": "xss",
        "apt": "apt",
    }

    _THREAT_PHRASE_ALIASES = {
        "sql injection": "sql-injection",
        "cross-site scripting": "xss",
        "cross site scripting": "xss",
        "denial of service": "ddos",
        "man in the middle": "man-in-the-middle",
        "zero day": "zero-day",
        "zeroday": "zero-day",
        "advanced persistent threat": "apt",
        "brute force": "brute-force",
        "insider threat": "insider-threat",
        "supply chain": "supply-chain-attack",
        "supply chain attack": "supply-chain-attack",
        "credential stuffing": "credential-stuffing",
        "buffer overflow": "buffer-overflow",
        "privilege escalation": "privilege-escalation",
        "dns spoofing": "dns-spoofing",
        "social engineering": "social-engineering",
        "crypto jacking": "cryptojacking",
    }

    _SOFTWARE_ABBREVIATIONS = {
        "log4j": "Apache Log4j",
        "apache log4j": "Apache Log4j",
        "exchange": "Microsoft Exchange Server",
        "exchange server": "Microsoft Exchange Server",
        "outlook": "Microsoft Outlook",
        "citrix": "Citrix ADC",
        "openssl": "OpenSSL",
        "smb": "Windows SMB",
        "print spooler": "Windows Print Spooler",
        "http2": "HTTP/2 implementations",
        "http/2": "HTTP/2 implementations",
    }

    _SEVERITY_ALIASES = {
        "severe": "critical",
        "dangerous": "critical",
    }

    def __init__(self, ontology_reader):
        from cybersecurity_query.ontology_reader import OntologyReader
        self._reader: OntologyReader = ontology_reader

        # Build vocabulary sets from the KB
        self._threats: Set[str] = set(self._reader.get_all_threats())
        self._vectors: Set[str] = set(self._reader.get_all_attack_vectors())
        self._severity_levels: Set[str] = set(self._reader.get_all_severity_levels())
        self._software: List[str] = self._reader.get_all_software()

        # Validate abbreviations against KB (drop any that don't resolve)
        self._threat_abbrevs = {
            k: v for k, v in self._THREAT_ABBREVIATIONS.items()
            if v in self._threats
        }
        self._threat_phrases = {
            k: v for k, v in self._THREAT_PHRASE_ALIASES.items()
            if v in self._threats
        }
        self._software_abbrevs = {
            k: v for k, v in self._SOFTWARE_ABBREVIATIONS.items()
            if v in {s for s in self._software}
        }

        logger.debug(
            "KBEntityResolver: %d threats, %d vectors, %d software from ontology",
            len(self._threats), len(self._vectors), len(self._software),
        )

    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract threats, software, severity, and vectors from query text."""
        entities: Dict[str, List[str]] = {
            "threats": [],
            "software": [],
            "severity": [],
            "vectors": [],
        }
        query_lower = query.lower().strip()

        # --- Threats ---
        # 1. Phrase aliases (multi-word before single-word)
        for alias, canonical in self._threat_phrases.items():
            if alias in query_lower and canonical not in entities["threats"]:
                entities["threats"].append(canonical)

        # 2. Abbreviations
        for abbrev, canonical in self._threat_abbrevs.items():
            if abbrev in query_lower and canonical not in entities["threats"]:
                entities["threats"].append(canonical)

        # 3. Direct KB threat names (hyphenated and space forms)
        for threat in self._threats:
            if threat in query_lower and threat not in entities["threats"]:
                entities["threats"].append(threat)
            space_form = threat.replace("-", " ")
            if space_form != threat and space_form in query_lower and threat not in entities["threats"]:
                entities["threats"].append(threat)

        # --- Software ---
        for alias, canonical in self._software_abbrevs.items():
            if alias in query_lower and canonical not in entities["software"]:
                entities["software"].append(canonical)

        for sw in self._software:
            if sw.lower() in query_lower and sw not in entities["software"]:
                entities["software"].append(sw)

        # --- Severity ---
        for level in self._severity_levels:
            if level in query_lower:
                if level not in entities["severity"]:
                    entities["severity"].append(level)

        # Aliases like "severe" -> "critical"
        for alias, canonical in self._SEVERITY_ALIASES.items():
            if alias in query_lower and canonical not in entities["severity"]:
                entities["severity"].append(canonical)

        # --- Attack vectors ---
        for vector in self._vectors:
            if vector.replace("-", " ") in query_lower or vector in query_lower:
                if vector not in entities["vectors"]:
                    entities["vectors"].append(vector)

        return entities


