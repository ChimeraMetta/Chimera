import unittest
import time
import json
import os
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

from hyperon import *
from dynamic_monitor import DynamicMonitor
from proofs.generator import MettaProofGenerator, OpenAIRequests
from proofs.pattern_mapper import PatternMapper
from proofs.analyzer import ImmuneSystemProofAnalyzer

class MettaProofSystemTests(unittest.TestCase):
    """
    Comprehensive test suite for the MeTTa-based proof generation system.
    Combines unit tests and integration tests.
    """
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create basic monitor for testing
        self.monitor = DynamicMonitor()
        
        # Initialize pattern mapper
        self.pattern_mapper = PatternMapper()
        
        # Load core ontology rules - adjust path as needed
        ontology_path = os.path.join(os.path.dirname(__file__), "metta", "proof_ontology.metta")
        if os.path.exists(ontology_path):
            self.monitor.load_metta_rules(ontology_path)
        
        # Create a mock API key for testing
        self.mock_api_key = "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"
        
        # Initialize the immune system analyzer with test configuration
        self.analyzer = ImmuneSystemProofAnalyzer(
            self.monitor.metta_space, 
            model_name="test-model",
            api_key=self.mock_api_key
        )
        
        # Sample binary search implementation for tests
        self.binary_search = """
        def binary_search(arr, target):
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
        
        # Sample test expressions
        self.test_expressions = {
            "bounds_check": "index < array.length",
            "ordering": "array is sorted in ascending order",
            "null_check": "if value != null then process(value)",
            "termination": "i decreases in each iteration",
            "error_handling": "return -1 if target not found"
        }
        
        # Sample proof JSON for testing conversion
        self.sample_proof_json = {
            "proof_components": [
                {
                    "type": "loop_invariant",
                    "location": "while_loop_1",
                    "expression": "left <= right",
                    "natural_language": "Search range is valid"
                },
                {
                    "type": "assertion",
                    "location": "mid_calculation",
                    "expression": "left <= mid <= right",
                    "natural_language": "Mid point is within the search range"
                }
            ],
            "verification_strategy": {
                "approach": "Prove loop invariants are maintained and imply postconditions",
                "key_lemmas": ["Binary search terminates in logarithmic time", "Array remains sorted"]
            }
        }
        
        # Mock API response for testing
        self.mock_api_response = json.dumps(self.sample_proof_json)
    
    def tearDown(self):
        """Clean up after each test."""
        # Reset MeTTa space
        if hasattr(self, 'monitor') and self.monitor:
            self.monitor.metta_space = self.monitor.metta.space()
    
    # ===================================
    # Unit Tests for Pattern Mapper
    # ===================================
    
    def test_pattern_mapper_bounds(self):
        """Test pattern mapper correctly identifies bounds checking patterns."""
        patterns = self.pattern_mapper.identify_patterns("index < length")
        self.assertTrue(any(prop == 'bound-check' for _, prop in patterns), 
                       "Failed to identify bounds checking pattern")
        
        patterns = self.pattern_mapper.identify_patterns("0 <= i && i < array.length")
        self.assertTrue(any(prop == 'bound-check' for _, prop in patterns),
                       "Failed to identify complex bounds checking pattern")
    
    def test_pattern_mapper_ordering(self):
        """Test pattern mapper correctly identifies ordering patterns."""
        patterns = self.pattern_mapper.identify_patterns("array is sorted")
        self.assertTrue(any(prop == 'ordering-check' for _, prop in patterns),
                       "Failed to identify basic ordering pattern")
        
        patterns = self.pattern_mapper.identify_patterns("elements are in ascending order")
        self.assertTrue(any(prop == 'ordering-check' for _, prop in patterns),
                       "Failed to identify ascending order pattern")
    
    def test_pattern_mapper_null_check(self):
        """Test pattern mapper correctly identifies null checking patterns."""
        patterns = self.pattern_mapper.identify_patterns("value != null")
        self.assertTrue(any(prop == 'null-check' for _, prop in patterns),
                       "Failed to identify null check pattern")
        
        patterns = self.pattern_mapper.identify_patterns("if collection is empty return default")
        self.assertTrue(any(prop == 'null-check' for _, prop in patterns),
                       "Failed to identify empty collection handling pattern")
    
    def test_pattern_mapper_termination(self):
        """Test pattern mapper correctly identifies termination patterns."""
        patterns = self.pattern_mapper.identify_patterns("i decreases in each iteration")
        self.assertTrue(any(prop == 'termination-guarantee' for _, prop in patterns),
                       "Failed to identify decreasing counter pattern")
        
        patterns = self.pattern_mapper.identify_patterns("loop invariant ensures progress")
        self.assertTrue(any(prop == 'termination-guarantee' for _, prop in patterns),
                       "Failed to identify loop invariant progress pattern")
    
    def test_pattern_mapper_error_handling(self):
        """Test pattern mapper correctly identifies error handling patterns."""
        patterns = self.pattern_mapper.identify_patterns("return -1 if not found")
        self.assertTrue(any(prop == 'error-handling' for _, prop in patterns),
                       "Failed to identify not found error pattern")
        
        patterns = self.pattern_mapper.identify_patterns("validate input before processing")
        self.assertTrue(any(prop == 'error-handling' for _, prop in patterns),
                       "Failed to identify input validation pattern")
    
    def test_metta_atom_generation(self):
        """Test generation of MeTTa atoms from expressions."""
        # Create a simplified pattern processor for testing
        processor = self.analyzer.pattern_processor
        
        # Test with a bounds checking expression
        expr_id = "test_expr"
        expression = "index < array.length"
        component = {
            "expression": expression, 
            "natural_language": "Ensures index is within bounds"
        }
        
        atoms = []
        processor._add_property_annotations(atoms, expr_id, expression, component)
        
        # Check that appropriate atoms were generated
        self.assertTrue(any("Expression-Property" in atom and "bound-check" in atom for atom in atoms),
                       "Failed to generate bound-check property atom")
    
    def test_json_to_metta_conversion(self):
        """Test conversion from JSON IR to MeTTa atoms."""
        # Create a simplified pattern processor for testing
        processor = self.analyzer.pattern_processor
        
        # Convert sample proof JSON to MeTTa atoms
        metta_atoms = processor._json_to_metta_proof(self.sample_proof_json)
        
        # Check for proper atom creation
        self.assertTrue(any("Expression" in atom for atom in metta_atoms),
                       "Failed to create Expression atom types")
        self.assertTrue(any("LoopInvariant" in atom for atom in metta_atoms),
                       "Failed to create LoopInvariant atoms")
        
        # Test for property recognition
        property_atoms = [atom for atom in metta_atoms if "Expression-Property" in atom]
        self.assertTrue(len(property_atoms) > 0,
                       "Failed to create Expression-Property relationships")
    
    def test_property_mapping(self):
        """Test mapping of property strings to MeTTa atoms."""
        property_str = "maintains array bounds"
        atom = self.pattern_mapper.map_requirement_to_property(property_str)
        self.assertEqual(atom, "bound-check",
                        f"Failed to map '{property_str}' to 'bound-check'")
        
        property_str = "preserves element order"
        atom = self.pattern_mapper.map_requirement_to_property(property_str)
        self.assertEqual(atom, "ordering-check",
                        f"Failed to map '{property_str}' to 'ordering-check'")
    
    # ===================================
    # Integration Tests with MeTTa
    # ===================================
    
    def test_metta_query_execution(self):
        """Test execution of queries in MeTTa space."""
        # Add test atoms to MeTTa space
        self.monitor.add_atom("(: bound-check Type)")
        self.monitor.add_atom("(: test_func Function)")
        self.monitor.add_atom("(function-has-property test_func bound-check)")
        
        # Test simple query
        result = self.monitor.query("(match &self (function-has-property test_func bound-check) True)")
        self.assertTrue(len(result) > 0 and result[0], 
                       "Failed to execute simple MeTTa query")
    
    def test_pattern_property_integration(self):
        """Test integration of pattern recognition with MeTTa."""
        # Add basic type definitions to MeTTa space
        self.monitor.add_atom("(: bound-check Type)")
        self.monitor.add_atom("(: index-less-than-length Pattern)")
        self.monitor.add_atom("(: test_expr Expression)")
        
        # Add expression definition
        self.monitor.add_atom("(= (test_expr) \"i < length\")")
        
        # Add property relationship
        self.monitor.add_atom("(Expression-Property test_expr bound-check)")
        
        # Test if MeTTa recognizes the relationship
        result = self.monitor.query("(match &self (Expression-Property test_expr bound-check) True)")
        self.assertTrue(len(result) > 0 and result[0],
                       "Failed to recognize Expression-Property relationship in MeTTa")
    
    def test_property_verification_rules(self):
        """Test that property verification rules work in MeTTa."""
        # Define test function with property
        self.monitor.add_atom("(: test_func Function)")
        self.monitor.add_atom("(: test_expr Expression)")
        self.monitor.add_atom("(: bound-check Type)")
        
        # Add relationship between function and expression
        self.monitor.add_atom("(function-invariant test_func test_expr)")
        
        # Add property to expression
        self.monitor.add_atom("(Expression-Property test_expr bound-check)")
        
        # Test if function satisfies property via expression
        result = self.monitor.query("""
            (match &self (function-has-property-via-expr test_func bound-check) True)
        """)
        
        self.assertTrue(len(result) > 0 and result[0],
                       "Failed to verify function property through expression in MeTTa")
    
    def test_add_property_to_metta(self):
        """Test adding property atoms to MeTTa."""
        func_name = "test_binary_search"
        property_str = "maintains array bounds"
        
        # Use pattern processor to add property
        atoms = self.analyzer.pattern_processor._add_property_to_metta(func_name, property_str)
        
        # Check if property was added
        query = f"(match &self (function-has-property {func_name} bound-check) True)"
        result = self.monitor.query(query)
        
        self.assertTrue(len(result) > 0 and result[0],
                       "Failed to add and verify property in MeTTa")
    
    # ===================================
    # API Client Tests
    # ===================================
    
    def test_openai_requests_init(self):
        """Test initialization of OpenAIRequests client."""
        client = OpenAIRequests(self.mock_api_key, "test-model")
        self.assertEqual(client.api_key, self.mock_api_key, "API key not set correctly")
        self.assertEqual(client.model, "test-model", "Model name not set correctly")
        self.assertEqual(client.base_url, "https://api.openai.com/v1", "Base URL not set correctly")
    
    @patch('requests.post')
    def test_openai_requests_chat_completion(self, mock_post):
        """Test chat completion method of OpenAIRequests."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": self.mock_api_response}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Create client and call method
        client = OpenAIRequests(self.mock_api_key, "test-model")
        result = client.chat_completion([{"role": "user", "content": "Test prompt"}])
        
        # Verify request was made correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {self.mock_api_key}")
        self.assertEqual(kwargs["json"]["model"], "test-model")
        
        # Verify response was processed correctly
        self.assertEqual(result["choices"][0]["message"]["content"], self.mock_api_response)
    
    @patch('requests.post')
    def test_openai_requests_get_completion_text(self, mock_post):
        """Test get_completion_text method of OpenAIRequests."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": self.mock_api_response}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Create client and call method
        client = OpenAIRequests(self.mock_api_key, "test-model")
        result = client.get_completion_text([{"role": "user", "content": "Test prompt"}])
        
        # Verify result
        self.assertEqual(result, self.mock_api_response)
    
    # ===================================
    # Mock API Integration Tests
    # ===================================
    
    @patch.object(OpenAIRequests, 'get_completion_text')
    def test_mock_proof_generation(self, mock_get_completion):
        """
        Test proof generation with mock API responses.
        """
        # Configure mock to return JSON string
        mock_get_completion.return_value = self.mock_api_response
        
        # Test analysis with mocked API call
        result = self.analyzer.analyze_function_for_proof(
            self.binary_search, 
            function_name="binary_search",
            max_attempts=1
        )
        
        # Check that proof generation succeeded with our mock
        self.assertTrue(result["success"],
                       "Mock proof generation failed to produce success result")
        
        # Check that some MeTTa atoms were created
        self.assertTrue(len(result.get("proof", [])) > 0,
                       "Mock proof generation failed to create MeTTa atoms")
    
    @patch.object(OpenAIRequests, 'get_completion_text')
    def test_mock_adaptation_verification(self, mock_get_completion):
        """Test adaptation verification with mocked API."""
        # Create a modified binary search that uses a different variable name
        modified_binary_search = self.binary_search.replace("left", "lo").replace("right", "hi")
        
        # Configure mock to return verification result
        mock_get_completion.return_value = json.dumps({
            "preserved_properties": [
                {"property": "maintains array bounds", "explanation": "Both versions check bounds"},
                {"property": "handles target not found case", "explanation": "Both return -1"}
            ],
            "violated_properties": []
        })
        
        # Test adaptation verification
        result = self.analyzer.verify_adaptation(
            self.binary_search,
            modified_binary_search,
            ["maintains array bounds", "handles target not found case"]
        )
        
        # Check that verification succeeded with our mock
        self.assertTrue(result["success"],
                       "Mock adaptation verification failed to produce success result")
        
        # Check that properties were preserved
        self.assertEqual(len(result.get("preserved_properties", [])), 2,
                         "Mock adaptation verification failed to preserve properties")
    
    # ===================================
    # Performance Tests
    # ===================================
    
    def test_pattern_mapping_performance(self):
        """Test performance of pattern mapping."""
        start_time = time.time()
        iterations = 100
        
        for _ in range(iterations):
            for expr_name, expr in self.test_expressions.items():
                self.pattern_mapper.identify_patterns(expr)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / (iterations * len(self.test_expressions))
        
        self.assertLess(avg_time, 0.01,
                       f"Pattern mapping too slow: {avg_time:.6f}s per pattern")
    
    def test_metta_query_performance(self):
        """Test performance of MeTTa queries."""
        # Set up MeTTa space with some atoms
        for i in range(100):
            self.monitor.add_atom(f"(: expr_{i} Expression)")
            self.monitor.add_atom(f"(= (expr_{i}) \"expression {i}\")")
            self.monitor.add_atom(f"(Expression-Property expr_{i} bound-check)")
        
        # Measure query performance
        start_time = time.time()
        iterations = 10
        
        for _ in range(iterations):
            self.monitor.query("(match &self (Expression-Property $expr bound-check) $expr)")
        
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        
        self.assertLess(avg_time, 0.1,
                       f"MeTTa query too slow: {avg_time:.6f}s per query")

    # ===================================
    # Helper Methods
    # ===================================
    
    def load_test_functions(self, filename):
        """Load test functions from a JSON file."""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return []


if __name__ == "__main__":
    unittest.main()