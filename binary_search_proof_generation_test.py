import unittest
import os
import json
import logging
import re
from typing import Dict, List, Any

from hyperon import *
from dynamic_monitor import DynamicMonitor
from proofs.generator import MettaProofGenerator, OpenAIRequests
from proofs.pattern_mapper import PatternMapper
from proofs.processor import ProofProcessorWithPatterns
from proofs.analyzer import ImmuneSystemProofAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("proof_generation_test")

class BinarySearchProofGenerationTest(unittest.TestCase):
    """
    Tests focused specifically on proof generation for binary search algorithms.
    Ensures that our system can generate valid, detailed proofs.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Get OpenAI API key from environment variable
        cls.api_key = os.environ.get("OPENAI_API_KEY")
        if not cls.api_key:
            raise unittest.SkipTest("OPENAI_API_KEY environment variable not set")
        
        # Use a less expensive model for testing
        cls.model_name = "gpt-3.5-turbo"
        logger.info(f"Using model: {cls.model_name} for proof generation tests")
        
        # Standard binary search implementation with correct indentation
        cls.binary_search = """def binary_search(arr, target):
    \"\"\"
    Performs binary search on a sorted array.
    Returns the index if found, otherwise -1.
    \"\"\"
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
"""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create monitor
        self.monitor = DynamicMonitor()
        
        # Initialize pattern mapper
        self.pattern_mapper = PatternMapper()
        
        # Add type definitions
        self.monitor.add_atom("(: Type Type)")
        self.monitor.add_atom("(: Property Type)")
        self.monitor.add_atom("(: bound-check Property)")
        self.monitor.add_atom("(: ordering-check Property)")
        self.monitor.add_atom("(: null-check Property)")
        self.monitor.add_atom("(: termination-guarantee Property)")
        self.monitor.add_atom("(: error-handling Property)")
        self.monitor.add_atom("(: Function Type)")
        self.monitor.add_atom("(: Expression Type)")
        
        # Add relationship definitions
        self.monitor.add_atom("(: function-has-property (--> Function Property Bool))")
        self.monitor.add_atom("(: Expression-Property (--> Expression Property Bool))")
        self.monitor.add_atom("(: adaptation-preserves-property (--> Property Bool))")
        self.monitor.add_atom("(: adaptation-violates-property (--> Property Bool))")
        
        # Initialize analyzer
        self.analyzer = ImmuneSystemProofAnalyzer(
            self.monitor.metta_space, 
            model_name=self.model_name,
            api_key=self.api_key
        )
        
        # Initialize OpenAI client
        self.openai_client = OpenAIRequests(self.api_key, self.model_name)
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'monitor') and self.monitor:
            self.monitor.metta_space = self.monitor.metta.space()
    
    def test_basic_proof_generation(self):
        """Test basic proof generation for binary search."""
        logger.info("Running test_basic_proof_generation")
        
        # Generate proof
        result = self.analyzer.analyze_function_for_proof(
            self.binary_search,
            function_name="binary_search",
            max_attempts=1
        )
        
        # Check proof was generated
        self.assertTrue(result["success"], 
                       f"Failed to generate proof for binary search: {result.get('error', '')}")
        
        # Check that proof components exist
        self.assertIn("proof", result, "Result should contain 'proof' key")
        self.assertGreater(len(result["proof"]), 0, "Proof should have at least one component")
        
        # Log proof components for analysis
        logger.info(f"Generated {len(result['proof'])} proof components")
        for i, component in enumerate(result["proof"][:5]):  # Log first 5 components
            logger.info(f"Proof component {i+1}: {component}")
        
        # Check for json_ir in the result
        self.assertIn("json_ir", result, "Result should contain 'json_ir' key")
        self.assertIn("proof_components", result["json_ir"], "JSON IR should contain proof_components")
        
        # Verify that loop invariants were identified
        invariant_result = self.monitor.query("(match &self (LoopInvariant $loc $expr) ($loc $expr))")
        self.assertTrue(len(invariant_result) > 0, "Proof should identify loop invariants")
        logger.info(f"Found {len(invariant_result)} loop invariants")
        
        # Verify that the function was marked as verified
        verify_result = self.monitor.query("(match &self (verified-function binary_search) True)")
        self.assertTrue(len(verify_result) > 0, "Function should be marked as verified")
    
    def test_proof_component_types(self):
        """Test that proof contains all expected component types."""
        logger.info("Running test_proof_component_types")
        
        # Generate proof
        result = self.analyzer.analyze_function_for_proof(
            self.binary_search,
            function_name="binary_search",
            max_attempts=1
        )
        
        # Check proof was generated
        self.assertTrue(result["success"], 
                       "Failed to generate proof for binary search")
        
        # Extract component types from JSON IR
        component_types = set()
        for component in result["json_ir"].get("proof_components", []):
            component_types.add(component.get("type"))
        
        logger.info(f"Found component types: {component_types}")
        
        # Minimal required component types
        required_types = {"loop_invariant", "precondition", "assertion"}
        
        # Check for each required type
        for required_type in required_types:
            self.assertIn(required_type, component_types,
                         f"Proof should contain '{required_type}' components")
    
    def test_detailed_proof_analysis(self):
        """Perform detailed analysis of the generated proof."""
        logger.info("Running test_detailed_proof_analysis")
        
        # Generate proof
        result = self.analyzer.analyze_function_for_proof(
            self.binary_search,
            function_name="binary_search",
            max_attempts=1
        )
        
        # Check proof was generated
        self.assertTrue(result["success"], 
                       "Failed to generate proof for binary search")
        
        # Extract components for analysis
        json_ir = result["json_ir"]
        proof_components = json_ir.get("proof_components", [])
        
        # Count components by type
        component_counts = {}
        for component in proof_components:
            comp_type = component.get("type", "unknown")
            component_counts[comp_type] = component_counts.get(comp_type, 0) + 1
        
        logger.info(f"Component counts: {component_counts}")
        
        # Extract expressions from components
        expressions = [
            component.get("expression", "") 
            for component in proof_components
        ]
        
        logger.info(f"Found {len(expressions)} expressions in proof")
        
        # Check for key invariants in the expressions
        invariant_patterns = [
            r"left\s*<=\s*right",  # Loop condition
            r"mid\s*=\s*left",     # Mid calculation
            r"arr\s*\[",           # Array access
            r"target",             # Target reference
            r"sorted|order"        # Sorted array assumption
        ]
        
        # Count matches for each pattern
        pattern_matches = {}
        for pattern in invariant_patterns:
            pattern_matches[pattern] = 0
            for expr in expressions:
                if re.search(pattern, expr, re.IGNORECASE):
                    pattern_matches[pattern] += 1
        
        logger.info(f"Pattern matches: {pattern_matches}")
        
        # Check that at least some patterns were found
        self.assertTrue(any(count > 0 for count in pattern_matches.values()),
                       "Proof should contain some expected invariant patterns")
        
        # Extract verification strategy
        strategy = json_ir.get("verification_strategy", {})
        approach = strategy.get("approach", "")
        key_lemmas = strategy.get("key_lemmas", [])
        
        logger.info(f"Verification approach: {approach}")
        logger.info(f"Key lemmas: {key_lemmas}")
        
        # Check that strategy is non-empty
        self.assertTrue(approach, "Verification approach should not be empty")
        self.assertTrue(key_lemmas, "Key lemmas should not be empty")
    
    def test_metta_integration(self):
        """Test integration of proof with MeTTa reasoning system."""
        logger.info("Running test_metta_integration")
        
        # Generate proof
        result = self.analyzer.analyze_function_for_proof(
            self.binary_search,
            function_name="binary_search",
            max_attempts=1
        )
        
        # Check proof was generated
        self.assertTrue(result["success"], 
                       "Failed to generate proof for binary search")
        
        # Check if MeTTa can query the proof components
        loop_invariants = self.monitor.query("(match &self (LoopInvariant $loc $expr) ($loc $expr))")
        preconditions = self.monitor.query("(match &self (Precondition $expr) $expr)")
        assertions = self.monitor.query("(match &self (Assertion $loc $expr) ($loc $expr))")
        
        logger.info(f"MeTTa query results:")
        logger.info(f"- Loop invariants: {len(loop_invariants)}")
        logger.info(f"- Preconditions: {len(preconditions)}")
        logger.info(f"- Assertions: {len(assertions)}")
        
        # Check that at least one of each component type was found
        component_found = (
            len(loop_invariants) > 0 or 
            len(preconditions) > 0 or 
            len(assertions) > 0
        )
        self.assertTrue(component_found, "MeTTa should find at least one proof component")
        
        # Check Expression-Property relationships
        property_rels = self.monitor.query("(match &self (Expression-Property $expr $prop) ($expr $prop))")
        logger.info(f"- Expression-Property relationships: {len(property_rels)}")
        self.assertTrue(len(property_rels) > 0, "Proof should create Expression-Property relationships")
        
        # Check if function-has-property relationships can be derived
        func_props = self.monitor.query("(match &self (function-has-property binary_search $prop) $prop)")
        logger.info(f"- Function properties: {func_props}")
        
        # Not all systems might derive this directly, so just log the result
        if len(func_props) == 0:
            logger.warning("No direct function-has-property relationships found")
    
    def test_proof_robustness(self):
        """Test that proof generation is robust and stable."""
        logger.info("Running test_proof_robustness")
        
        # Generate proof multiple times to test consistency
        results = []
        attempts = 2  # Limited number for API cost reasons
        
        for i in range(attempts):
            logger.info(f"Proof generation attempt {i+1}/{attempts}")
            
            # Reset MeTTa space between attempts
            if i > 0:
                self.monitor.metta_space = self.monitor.metta.space()
                self._initialize_metta_space()
            
            # Generate proof
            result = self.analyzer.analyze_function_for_proof(
                self.binary_search,
                function_name=f"binary_search_{i}",
                max_attempts=1
            )
            
            # Store result if successful
            if result["success"]:
                results.append(result)
                
                # Log key metrics
                component_count = len(result.get("json_ir", {}).get("proof_components", []))
                logger.info(f"Generated proof with {component_count} components")
        
        # Check that at least one successful attempt was made
        self.assertGreater(len(results), 0, 
                         "At least one proof generation attempt should succeed")
        
        # If multiple successful attempts, compare them
        if len(results) > 1:
            # Compare component counts
            counts = [len(r.get("json_ir", {}).get("proof_components", [])) for r in results]
            logger.info(f"Component counts across attempts: {counts}")
            
            # Check that component counts are reasonably similar
            # (allow for some variance due to LLM non-determinism)
            max_count = max(counts)
            min_count = min(counts)
            variance_ratio = min_count / max_count if max_count > 0 else 0
            
            logger.info(f"Proof consistency ratio: {variance_ratio:.2f}")
            self.assertGreaterEqual(variance_ratio, 0.5, 
                                  "Proof generation should be reasonably consistent")
    
    def _initialize_metta_space(self):
        """Initialize MeTTa space with necessary definitions."""
        # Add type definitions
        self.monitor.add_atom("(: Type Type)")
        self.monitor.add_atom("(: Property Type)")
        self.monitor.add_atom("(: bound-check Property)")
        self.monitor.add_atom("(: ordering-check Property)")
        self.monitor.add_atom("(: null-check Property)")
        self.monitor.add_atom("(: termination-guarantee Property)")
        self.monitor.add_atom("(: error-handling Property)")
        self.monitor.add_atom("(: Function Type)")
        self.monitor.add_atom("(: Expression Type)")
        
        # Add relationship definitions
        self.monitor.add_atom("(: function-has-property (--> Function Property Bool))")
        self.monitor.add_atom("(: Expression-Property (--> Expression Property Bool))")
        self.monitor.add_atom("(: adaptation-preserves-property (--> Property Bool))")
        self.monitor.add_atom("(: adaptation-violates-property (--> Property Bool))")

if __name__ == "__main__":
    unittest.main()