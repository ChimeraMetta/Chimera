#!/usr/bin/env python3
"""Test suite for the cybersecurity threat NL query system.

Tests NL parser intent classification, entity extraction, MeTTa query
generation, and end-to-end integration with real MeTTa execution.
"""

import os
import sys
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cybersecurity_query.models import QueryIntent, ParsedQuery
from cybersecurity_query.nl_parser import NLParser
from cybersecurity_query.metta_query_generator import MeTTaQueryGenerator
from cybersecurity_query.reasoning_engine import ReasoningEngine
from cybersecurity_query.response_formatter import ResponseFormatter
from cybersecurity_query.engine import CyberSecurityQueryEngine
from cybersecurity_query.query_builder import QueryBuilder
from cybersecurity_query.ontology_reader import OntologyReader, _ONTOLOGY_PATH
from cybersecurity_query.entity_resolver import CVEResolver, KBEntityResolver
from cybersecurity_query.intent_classifier import EmbeddingIntentClassifier


class TestNLParserIntentClassification(unittest.TestCase):
    """Test intent classification for various NL queries."""

    def setUp(self):
        self.engine = CyberSecurityQueryEngine()
        self.parser = self.engine.parser

    def test_threat_lookup_what_is(self):
        result = self.parser.parse("What is ransomware?")
        self.assertEqual(result.intent, QueryIntent.THREAT_LOOKUP)
        self.assertIn("ransomware", result.entities["threats"])

    def test_threat_lookup_tell_me(self):
        result = self.parser.parse("Tell me about SQL injection")
        self.assertEqual(result.intent, QueryIntent.THREAT_LOOKUP)
        self.assertIn("sql-injection", result.entities["threats"])

    def test_mitigation_how_to_protect(self):
        result = self.parser.parse("How to protect against phishing?")
        self.assertEqual(result.intent, QueryIntent.MITIGATION_ADVICE)
        self.assertIn("phishing", result.entities["threats"])

    def test_mitigation_what_mitigates(self):
        result = self.parser.parse("What mitigates DDoS?")
        self.assertEqual(result.intent, QueryIntent.MITIGATION_ADVICE)
        self.assertIn("ddos", result.entities["threats"])

    def test_vulnerability_cve(self):
        result = self.parser.parse("Is CVE-2021-44228 critical?")
        self.assertEqual(result.intent, QueryIntent.VULNERABILITY_CHECK)
        self.assertIn("CVE-2021-44228", result.entities["cves"])

    def test_vulnerability_alias(self):
        result = self.parser.parse("What does Log4Shell affect?")
        self.assertEqual(result.intent, QueryIntent.VULNERABILITY_CHECK)
        self.assertIn("CVE-2021-44228", result.entities["cves"])

    def test_relationship_lead_to(self):
        result = self.parser.parse("What can phishing lead to?")
        self.assertEqual(result.intent, QueryIntent.RELATIONSHIP_QUERY)
        self.assertIn("phishing", result.entities["threats"])

    def test_severity_how_severe(self):
        result = self.parser.parse("How severe is ransomware?")
        self.assertEqual(result.intent, QueryIntent.SEVERITY_ASSESSMENT)
        self.assertIn("ransomware", result.entities["threats"])

    def test_severity_list_critical(self):
        result = self.parser.parse("What are the critical threats?")
        self.assertEqual(result.intent, QueryIntent.SEVERITY_ASSESSMENT)
        self.assertIn("critical", result.entities["severity"])

    def test_mitigation_for_software(self):
        result = self.parser.parse("How to protect Apache Log4j?")
        self.assertEqual(result.intent, QueryIntent.MITIGATION_ADVICE)
        self.assertIn("Apache Log4j", result.entities["software"])


class TestNLParserEntityExtraction(unittest.TestCase):
    """Test entity extraction from queries."""

    def setUp(self):
        self.engine = CyberSecurityQueryEngine()
        self.parser = self.engine.parser

    def test_extract_cve(self):
        result = self.parser.parse("Tell me about CVE-2021-44228")
        self.assertIn("CVE-2021-44228", result.entities["cves"])

    def test_extract_threat_alias(self):
        result = self.parser.parse("What is cross-site scripting?")
        self.assertIn("xss", result.entities["threats"])

    def test_extract_threat_sql_injection(self):
        result = self.parser.parse("How to prevent sql injection?")
        self.assertIn("sql-injection", result.entities["threats"])

    def test_extract_software(self):
        result = self.parser.parse("What vulnerabilities affect OpenSSL?")
        self.assertIn("OpenSSL", result.entities["software"])

    def test_extract_severity(self):
        result = self.parser.parse("List critical threats")
        self.assertIn("critical", result.entities["severity"])

    def test_extract_cve_alias_heartbleed(self):
        result = self.parser.parse("What is Heartbleed?")
        self.assertIn("CVE-2014-0160", result.entities["cves"])

    def test_extract_cve_alias_eternalblue(self):
        result = self.parser.parse("Tell me about EternalBlue")
        self.assertIn("CVE-2017-0144", result.entities["cves"])

    def test_extract_multiple_entities(self):
        result = self.parser.parse("Is CVE-2021-44228 related to ransomware?")
        self.assertIn("CVE-2021-44228", result.entities["cves"])
        self.assertIn("ransomware", result.entities["threats"])

    def test_extract_threat_brute_force(self):
        result = self.parser.parse("How to prevent brute force attacks?")
        self.assertIn("brute-force", result.entities["threats"])


class TestMeTTaQueryGeneration(unittest.TestCase):
    """Test NL to MeTTa query translation."""

    def setUp(self):
        self.engine = CyberSecurityQueryEngine()
        self.parser = self.engine.parser
        self.generator = MeTTaQueryGenerator()

    def test_threat_lookup_generates_queries(self):
        parsed = self.parser.parse("What is ransomware?")
        plan = self.generator.generate(parsed)
        self.assertTrue(len(plan.steps) > 0)
        query = self.generator.render_query(plan.steps[0])
        self.assertIn("ransomware", query)
        self.assertIn("match", query)

    def test_vulnerability_check_generates_queries(self):
        parsed = self.parser.parse("CVE-2021-44228")
        plan = self.generator.generate(parsed)
        self.assertTrue(len(plan.steps) > 0)
        query = self.generator.render_query(plan.steps[0])
        self.assertIn("CVE-2021-44228", query)

    def test_mitigation_generates_queries(self):
        parsed = self.parser.parse("How to protect against phishing?")
        plan = self.generator.generate(parsed)
        self.assertTrue(len(plan.steps) > 0)

    def test_severity_generates_queries(self):
        parsed = self.parser.parse("What are the critical threats?")
        plan = self.generator.generate(parsed)
        self.assertTrue(len(plan.steps) > 0)

    def test_relationship_generates_queries(self):
        parsed = self.parser.parse("What can phishing lead to?")
        plan = self.generator.generate(parsed)
        self.assertTrue(len(plan.steps) > 0)

    def test_multi_hop_software_mitigation(self):
        parsed = self.parser.parse("What mitigations for Apache Log4j?")
        plan = self.generator.generate(parsed)
        self.assertTrue(len(plan.steps) >= 1)


class TestReasoningEngine(unittest.TestCase):
    """Test reasoning engine with fallback mode."""

    def setUp(self):
        self.engine_obj = CyberSecurityQueryEngine()
        self.reasoning = self.engine_obj.reasoning
        self.parser = self.engine_obj.parser
        self.generator = MeTTaQueryGenerator()

    def test_threat_lookup_returns_results(self):
        parsed = self.parser.parse("What is ransomware?")
        plan = self.generator.generate(parsed)
        result = self.reasoning.execute_plan(plan, parsed)
        self.assertTrue(len(result.results) > 0)
        self.assertTrue(result.confidence > 0)

    def test_vulnerability_returns_results(self):
        parsed = self.parser.parse("CVE-2021-44228")
        plan = self.generator.generate(parsed)
        result = self.reasoning.execute_plan(plan, parsed)
        self.assertTrue(len(result.results) > 0)

    def test_mitigation_returns_results(self):
        parsed = self.parser.parse("How to protect against SQL injection?")
        plan = self.generator.generate(parsed)
        result = self.reasoning.execute_plan(plan, parsed)
        self.assertTrue(len(result.results) > 0)

    def test_severity_returns_results(self):
        parsed = self.parser.parse("What are the critical threats?")
        plan = self.generator.generate(parsed)
        result = self.reasoning.execute_plan(plan, parsed)
        self.assertTrue(len(result.results) > 0)

    def test_reasoning_trace_populated(self):
        parsed = self.parser.parse("What is phishing?")
        plan = self.generator.generate(parsed)
        result = self.reasoning.execute_plan(plan, parsed)
        self.assertTrue(len(result.reasoning_trace.steps) > 0)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests with the CyberSecurityQueryEngine."""

    def setUp(self):
        self.engine = CyberSecurityQueryEngine()

    def test_what_is_ransomware(self):
        result = self.engine.query("What is ransomware?")
        self.assertEqual(result.intent, QueryIntent.THREAT_LOOKUP)
        self.assertTrue(len(result.results) > 0)
        self.assertTrue(result.formatted_response)
        self.assertIn("ransomware", result.formatted_response.lower())

    def test_protect_against_sql_injection(self):
        result = self.engine.query("How to protect against SQL injection?")
        self.assertEqual(result.intent, QueryIntent.MITIGATION_ADVICE)
        self.assertTrue(len(result.results) > 0)
        self.assertTrue(result.formatted_response)

    def test_cve_lookup(self):
        result = self.engine.query("CVE-2021-44228")
        self.assertEqual(result.intent, QueryIntent.VULNERABILITY_CHECK)
        self.assertTrue(len(result.results) > 0)
        self.assertIn("Log4Shell", result.formatted_response)

    def test_phishing_leads_to(self):
        result = self.engine.query("What can phishing lead to?")
        self.assertEqual(result.intent, QueryIntent.RELATIONSHIP_QUERY)
        self.assertTrue(len(result.results) > 0)

    def test_critical_threats(self):
        result = self.engine.query("What are the critical threats?")
        self.assertEqual(result.intent, QueryIntent.SEVERITY_ASSESSMENT)
        self.assertTrue(len(result.results) > 0)

    def test_multi_hop_software_mitigation(self):
        result = self.engine.query("What mitigations for Apache Log4j?")
        self.assertEqual(result.intent, QueryIntent.MITIGATION_ADVICE)
        self.assertTrue(len(result.results) > 0)

    def test_get_threat_info_convenience(self):
        result = self.engine.get_threat_info("phishing")
        self.assertEqual(result.intent, QueryIntent.THREAT_LOOKUP)
        self.assertTrue(len(result.results) > 0)

    def test_get_vulnerability_convenience(self):
        result = self.engine.get_vulnerability("CVE-2021-44228")
        self.assertEqual(result.intent, QueryIntent.VULNERABILITY_CHECK)
        self.assertTrue(len(result.results) > 0)

    def test_assess_severity_convenience(self):
        result = self.engine.assess_severity("ransomware")
        self.assertEqual(result.intent, QueryIntent.SEVERITY_ASSESSMENT)
        self.assertTrue(len(result.results) > 0)

    def test_trace_attack_chain_convenience(self):
        result = self.engine.trace_attack_chain("phishing")
        self.assertEqual(result.intent, QueryIntent.RELATIONSHIP_QUERY)
        self.assertTrue(len(result.results) > 0)


class TestQueryBuilder(unittest.TestCase):
    """Test the fluent QueryBuilder API."""

    def setUp(self):
        self.engine = CyberSecurityQueryEngine()

    def test_builder_threat_lookup(self):
        result = (QueryBuilder(self.engine)
                  .with_intent("threat_lookup")
                  .for_threat("ransomware")
                  .execute())
        self.assertEqual(result.intent, QueryIntent.THREAT_LOOKUP)
        self.assertTrue(len(result.results) > 0)

    def test_builder_vulnerability(self):
        result = (QueryBuilder(self.engine)
                  .with_intent("vulnerability_check")
                  .for_cve("CVE-2021-44228")
                  .execute())
        self.assertEqual(result.intent, QueryIntent.VULNERABILITY_CHECK)
        self.assertTrue(len(result.results) > 0)

    def test_builder_mitigation(self):
        result = (QueryBuilder(self.engine)
                  .with_intent("mitigation_advice")
                  .for_threat("phishing")
                  .execute())
        self.assertEqual(result.intent, QueryIntent.MITIGATION_ADVICE)
        self.assertTrue(len(result.results) > 0)

    def test_builder_software_mitigation(self):
        result = (QueryBuilder(self.engine)
                  .with_intent("mitigation_advice")
                  .for_software("Apache Log4j")
                  .execute())
        self.assertEqual(result.intent, QueryIntent.MITIGATION_ADVICE)
        self.assertTrue(len(result.results) > 0)


class TestResponseFormatter(unittest.TestCase):
    """Test response formatting."""

    def setUp(self):
        self.engine = CyberSecurityQueryEngine()
        self.formatter = ResponseFormatter()

    def test_threat_format_has_sections(self):
        result = self.engine.query("What is ransomware?")
        self.assertIn("Threat Information", result.formatted_response)
        self.assertIn("ransomware", result.formatted_response.lower())

    def test_vulnerability_format(self):
        result = self.engine.query("CVE-2021-44228")
        self.assertIn("Vulnerability Details", result.formatted_response)

    def test_mitigation_format(self):
        result = self.engine.query("How to protect against phishing?")
        self.assertIn("Mitigation", result.formatted_response)

    def test_reasoning_chain_displayed(self):
        result = self.engine.query("What is ransomware?")
        self.assertIn("Reasoning Chain", result.formatted_response)


class TestTranslationAccuracy(unittest.TestCase):
    """Measure translation accuracy against a set of test queries."""

    def test_20_query_accuracy(self):
        """Test that at least 80% of test queries produce valid translations."""
        engine = CyberSecurityQueryEngine()
        test_queries = [
            "What is ransomware?",
            "Tell me about SQL injection",
            "How to protect against phishing?",
            "What mitigates DDoS?",
            "Is CVE-2021-44228 critical?",
            "What does Log4Shell affect?",
            "What can phishing lead to?",
            "How severe is ransomware?",
            "What are the critical threats?",
            "What mitigations for Apache Log4j?",
            "Tell me about buffer overflow",
            "How to prevent brute force attacks?",
            "What is cross-site scripting?",
            "CVE-2017-0144 details",
            "How to defend against man in the middle?",
            "What threats use email as attack vector?",
            "Describe social engineering",
            "What is Heartbleed?",
            "How dangerous is APT?",
            "Protect against supply chain attacks",
        ]

        successful = 0
        for q in test_queries:
            result = engine.query(q)
            if result.results and result.confidence > 0:
                successful += 1

        accuracy = successful / len(test_queries)
        print(f"\nTranslation accuracy: {successful}/{len(test_queries)} = {accuracy:.0%}")
        self.assertGreaterEqual(accuracy, 0.8, f"Translation accuracy {accuracy:.0%} below 80% threshold")


class TestSDKImport(unittest.TestCase):
    """Test that the SDK is importable."""

    def test_import_engine(self):
        from cybersecurity_query import CyberSecurityQueryEngine
        self.assertIsNotNone(CyberSecurityQueryEngine)

    def test_import_query_builder(self):
        from cybersecurity_query import QueryBuilder
        self.assertIsNotNone(QueryBuilder)

    def test_import_models(self):
        from cybersecurity_query import (
            QueryIntent, QueryResult, ThreatInfo, VulnerabilityInfo,
            MitigationInfo, SeverityAssessment, QueryMetrics
        )
        self.assertIsNotNone(QueryIntent)
        self.assertIsNotNone(QueryResult)


# =====================================================================
# OntologyReader + KB entity resolution tests
# =====================================================================


class TestOntologyReaderExtraction(unittest.TestCase):
    """Verify entity vocabulary extraction from the MeTTa ontology."""

    def setUp(self):
        self.reader = OntologyReader(_ONTOLOGY_PATH)

    def test_get_all_threats_count(self):
        threats = self.reader.get_all_threats()
        self.assertGreaterEqual(len(threats), 20, f"Expected >=20 threats, got {len(threats)}")

    def test_get_all_threats_contains_known(self):
        threats = self.reader.get_all_threats()
        for expected in ["ransomware", "phishing", "sql-injection", "xss", "ddos",
                         "rootkit", "trojan", "malware", "apt"]:
            self.assertIn(expected, threats, f"Missing threat: {expected}")

    def test_get_all_attack_vectors(self):
        vectors = self.reader.get_all_attack_vectors()
        self.assertGreaterEqual(len(vectors), 12)
        self.assertIn("email", vectors)
        self.assertIn("network", vectors)
        self.assertIn("web-application", vectors)

    def test_get_all_severity_levels(self):
        levels = self.reader.get_all_severity_levels()
        self.assertGreaterEqual(len(levels), 4)
        for expected in ["critical", "high", "medium", "low"]:
            self.assertIn(expected, levels)

    def test_get_all_software(self):
        software = self.reader.get_all_software()
        self.assertGreaterEqual(len(software), 8)
        self.assertIn("Apache Log4j", software)
        self.assertIn("OpenSSL", software)

    def test_get_all_cve_ids(self):
        cves = self.reader.get_all_cve_ids()
        self.assertGreaterEqual(len(cves), 8)
        self.assertIn("CVE-2021-44228", cves)
        self.assertIn("CVE-2014-0160", cves)

    def test_get_cve_names(self):
        names = self.reader.get_cve_names()
        self.assertEqual(names["CVE-2021-44228"], "Log4Shell")
        self.assertEqual(names["CVE-2014-0160"], "Heartbleed")

    def test_get_all_mitigations(self):
        mitigations = self.reader.get_all_mitigations()
        self.assertGreaterEqual(len(mitigations), 15)
        self.assertIn("input-validation", mitigations)
        self.assertIn("patch-management", mitigations)

    def test_type_declarations_parsed(self):
        """Ensure (: X Type) lines are parsed as facts, not skipped."""
        type_facts = self.reader._facts.get(":", [])
        self.assertGreater(len(type_facts), 0, "No type declarations found")


class TestKBEntityResolver(unittest.TestCase):
    """Test KB-driven entity resolution from the MeTTa ontology."""

    def setUp(self):
        reader = OntologyReader(_ONTOLOGY_PATH)
        self.resolver = KBEntityResolver(reader)

    def test_extract_ransomware(self):
        entities = self.resolver.extract_entities("What is ransomware?")
        self.assertIn("ransomware", entities["threats"])

    def test_extract_sql_injection_alias(self):
        entities = self.resolver.extract_entities("How to prevent sql injection?")
        self.assertIn("sql-injection", entities["threats"])

    def test_extract_brute_force_space_form(self):
        entities = self.resolver.extract_entities("How to prevent brute force attacks?")
        self.assertIn("brute-force", entities["threats"])

    def test_extract_cross_site_scripting(self):
        entities = self.resolver.extract_entities("What is cross-site scripting?")
        self.assertIn("xss", entities["threats"])

    def test_extract_man_in_the_middle(self):
        entities = self.resolver.extract_entities("How to defend against man in the middle?")
        self.assertIn("man-in-the-middle", entities["threats"])

    def test_extract_severity_critical(self):
        entities = self.resolver.extract_entities("List critical threats")
        self.assertIn("critical", entities["severity"])

    def test_extract_software_openssl(self):
        entities = self.resolver.extract_entities("What vulnerabilities affect OpenSSL?")
        self.assertIn("OpenSSL", entities["software"])

    def test_extract_software_log4j_alias(self):
        entities = self.resolver.extract_entities("What about log4j?")
        self.assertIn("Apache Log4j", entities["software"])


class TestEmbeddingIntentClassifier(unittest.TestCase):
    """Test embedding-based intent classification on paraphrased queries."""

    @classmethod
    def setUpClass(cls):
        cls.clf = EmbeddingIntentClassifier()

    def _classify(self, query):
        # Use empty entities to test pure embedding classification
        intent, conf = self.clf.classify(query, {
            "threats": [], "cves": [], "software": [],
            "severity": [], "vectors": [],
        })
        return intent, conf

    def test_paraphrased_mitigation(self):
        intent, _ = self._classify("What defenses exist for ransomware?")
        self.assertEqual(intent, QueryIntent.MITIGATION_ADVICE)

    def test_paraphrased_threat_lookup(self):
        intent, _ = self._classify("Can you explain what a rootkit does?")
        self.assertEqual(intent, QueryIntent.THREAT_LOOKUP)

    def test_paraphrased_severity(self):
        intent, _ = self._classify("Which threats pose the greatest danger?")
        self.assertEqual(intent, QueryIntent.SEVERITY_ASSESSMENT)

    def test_paraphrased_relationship(self):
        intent, _ = self._classify("What attacks follow after a phishing campaign?")
        self.assertEqual(intent, QueryIntent.RELATIONSHIP_QUERY)

    def test_paraphrased_vulnerability(self):
        intent, _ = self._classify("Tell me about the Log4j security flaw")
        self.assertIn(intent, (QueryIntent.VULNERABILITY_CHECK, QueryIntent.THREAT_LOOKUP))

    def test_confidence_above_threshold(self):
        _, conf = self._classify("What is ransomware?")
        self.assertGreater(conf, 0.5, "Confidence too low for clear query")

    def test_entity_override_cve(self):
        """CVE entity presence should force VULNERABILITY_CHECK."""
        intent, _ = self.clf.classify("Tell me about this thing", {
            "threats": [], "cves": ["CVE-2021-44228"], "software": [],
            "severity": [], "vectors": [],
        })
        self.assertEqual(intent, QueryIntent.VULNERABILITY_CHECK)

    def test_entity_override_mitigation(self):
        """Threat entity + mitigation keyword should force MITIGATION_ADVICE."""
        intent, _ = self.clf.classify("How to protect against phishing?", {
            "threats": ["phishing"], "cves": [], "software": [],
            "severity": [], "vectors": [],
        })
        self.assertEqual(intent, QueryIntent.MITIGATION_ADVICE)


class TestCVEResolver(unittest.TestCase):
    """Test CVE structural identifier extraction."""

    def setUp(self):
        self.resolver = CVEResolver()

    def test_extract_cve_format(self):
        cves = self.resolver.extract("Tell me about CVE-2021-44228")
        self.assertIn("CVE-2021-44228", cves)

    def test_extract_alias_log4shell(self):
        cves = self.resolver.extract("What is Log4Shell?")
        self.assertIn("CVE-2021-44228", cves)

    def test_extract_alias_heartbleed(self):
        cves = self.resolver.extract("Explain Heartbleed")
        self.assertIn("CVE-2014-0160", cves)

    def test_extract_multiple(self):
        cves = self.resolver.extract("CVE-2021-44228 and CVE-2017-0144")
        self.assertIn("CVE-2021-44228", cves)
        self.assertIn("CVE-2017-0144", cves)

    def test_no_false_positives(self):
        cves = self.resolver.extract("What is ransomware?")
        self.assertEqual(cves, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
