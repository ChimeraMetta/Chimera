"""Intent classification for cybersecurity NL queries.

Uses sentence-transformer embeddings to classify query intent by cosine
similarity to pre-embedded exemplar phrases. Handles arbitrary paraphrased
queries without any regex patterns.

Entity-based overrides are preserved as logical disambiguation rules (e.g.
CVE present -> VULNERABILITY_CHECK).

Requires: ``sentence-transformers>=2.2.0``
"""

import logging
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer

from cybersecurity_query.models import QueryIntent

logger = logging.getLogger("intent_classifier")

# Exemplar phrases per intent (pre-embedded at init, used for similarity)
_INTENT_EXEMPLARS: Dict[QueryIntent, List[str]] = {
    QueryIntent.THREAT_LOOKUP: [
        "What is ransomware?",
        "Tell me about SQL injection",
        "Describe phishing attacks",
        "Explain what a rootkit does",
        "Can you explain what a trojan is?",
        "Info on buffer overflow",
        "Define social engineering",
        "What does malware do?",
    ],
    QueryIntent.VULNERABILITY_CHECK: [
        "Tell me about CVE-2021-44228",
        "What does Log4Shell affect?",
        "Is CVE-2017-0144 critical?",
        "What is the Heartbleed vulnerability?",
        "Details on EternalBlue",
        "Which systems does ProxyLogon target?",
        "What software is affected by PrintNightmare?",
    ],
    QueryIntent.MITIGATION_ADVICE: [
        "How to protect against ransomware?",
        "What defenses exist for phishing?",
        "How do I prevent SQL injection?",
        "What mitigations are available for DDoS?",
        "How to defend against man in the middle attacks?",
        "What can I do to stop brute force?",
        "Protect against supply chain attacks",
        "What safeguards work for XSS?",
    ],
    QueryIntent.SEVERITY_ASSESSMENT: [
        "How severe is ransomware?",
        "What are the critical threats?",
        "How dangerous is APT?",
        "Rate the severity of phishing",
        "Which threats pose the greatest danger?",
        "List high severity threats",
        "Risk assessment for zero-day exploits",
    ],
    QueryIntent.RELATIONSHIP_QUERY: [
        "What can phishing lead to?",
        "What attacks follow SQL injection?",
        "What attacks follow after a phishing campaign?",
        "Show the attack chain for ransomware",
        "What threats use email as vector?",
        "How do threats relate to each other?",
        "What is the kill chain for malware?",
        "Which threats enable privilege escalation?",
        "What does phishing lead to next?",
        "What is the attack progression from SQL injection?",
    ],
}


class EmbeddingIntentClassifier:
    """Classify intent using sentence embeddings and cosine similarity.

    Uses ``sentence-transformers`` with model ``all-MiniLM-L6-v2`` (~80 MB,
    CPU, ~5 ms/encode).

    Entity-based overrides handle structural disambiguation (CVE presence ->
    VULNERABILITY_CHECK, mitigation keywords + entity -> MITIGATION_ADVICE).
    """

    SIMILARITY_THRESHOLD = 0.35
    CONFIDENCE_MIN = 0.6
    CONFIDENCE_MAX = 0.95

    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        # Try loading from local cache first; fall back to download on first run.
        try:
            self._model = SentenceTransformer(self._MODEL_NAME, local_files_only=True)
        except Exception:
            logger.info("Downloading model %s (first run only)...", self._MODEL_NAME)
            self._model = SentenceTransformer(self._MODEL_NAME)
        self._intent_embeddings: Dict[QueryIntent, object] = {}

        # Pre-embed all exemplars per intent
        for intent, phrases in _INTENT_EXEMPLARS.items():
            self._intent_embeddings[intent] = self._model.encode(
                phrases, convert_to_numpy=True, normalize_embeddings=True,
            )

        logger.info("EmbeddingIntentClassifier: model loaded")

    def classify(self, query: str, entities: Dict[str, List[str]]) -> Tuple[QueryIntent, float]:
        """Classify by entity overrides first, then embedding similarity."""
        query_lower = query.lower().strip()

        # --- Entity-based overrides (logical disambiguation, not NL) ---
        if entities.get("cves"):
            return QueryIntent.VULNERABILITY_CHECK, 0.9

        mitigation_keywords = [
            "protect", "defend", "prevent", "mitigate", "mitigation",
            "stop", "block", "secure", "safeguard", "countermeasure",
            "defense", "remedy", "remediat",
        ]
        if entities.get("threats") and any(kw in query_lower for kw in mitigation_keywords):
            return QueryIntent.MITIGATION_ADVICE, 0.9
        if entities.get("software") and any(kw in query_lower for kw in mitigation_keywords):
            return QueryIntent.MITIGATION_ADVICE, 0.85

        # --- Embedding similarity ---
        query_emb = self._model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
        )[0]

        best_intent = QueryIntent.UNKNOWN
        best_sim = -1.0

        for intent, embs in self._intent_embeddings.items():
            # Max similarity across exemplars for this intent
            sims = embs @ query_emb  # dot product on normalized vectors = cosine
            max_sim = float(sims.max())
            if max_sim > best_sim:
                best_sim = max_sim
                best_intent = intent

        if best_sim < self.SIMILARITY_THRESHOLD:
            # Below threshold -- try threat entity fallback
            if entities.get("threats"):
                return QueryIntent.THREAT_LOOKUP, 0.6
            return QueryIntent.UNKNOWN, 0.0

        # Map similarity [threshold, 1.0] -> confidence [CONFIDENCE_MIN, CONFIDENCE_MAX]
        norm = (best_sim - self.SIMILARITY_THRESHOLD) / (1.0 - self.SIMILARITY_THRESHOLD)
        confidence = self.CONFIDENCE_MIN + norm * (self.CONFIDENCE_MAX - self.CONFIDENCE_MIN)
        confidence = min(self.CONFIDENCE_MAX, max(self.CONFIDENCE_MIN, confidence))

        return best_intent, confidence
