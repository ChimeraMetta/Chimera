import unittest
import os
import json
import time
import logging
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
logger = logging.getLogger("proof_system_test")

class RealAPIProofSystemTests(unittest.TestCase):
    """
    Tests for the MeTTa-based proof generation system using real OpenAI API calls.
    Integrates with actual API responses for more realistic testing.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Get OpenAI API key from environment variable
        cls.api_key = os.environ.get("OPENAI_API_KEY")
        if not cls.api_key:
            raise unittest.SkipTest("OPENAI_API_KEY environment variable not set. Skipping live API tests.")
        
        # Use a less expensive model for testing
        cls.model_name = "gpt-3.5-turbo"
        logger.info(f"Using model: {cls.model_name} for live API tests")
        
        # Sample binary search implementation for tests
        cls.binary_search = """
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
    
    def setUp(self):
        """Set up test environment before each test."""
        # Skip tests if no API key
        if not hasattr(self, 'api_key') or not self.api_key:
            self.skipTest("OPENAI_API_KEY environment variable not set")
        
        # Create basic monitor for testing
        self.monitor = DynamicMonitor()
        
        # Initialize pattern mapper
        self.pattern_mapper = PatternMapper()
        
        # Add basic type definitions that would normally be in the ontology
        self.monitor.add_atom("(: Type Type)")
        self.monitor.add_atom("(: Property Type)")
        self.monitor.add_atom("(: bound-check Property)")
        self.monitor.add_atom("(: ordering-check Property)")
        self.monitor.add_atom("(: null-check Property)")
        self.monitor.add_atom("(: termination-guarantee Property)")
        self.monitor.add_atom("(: error-handling Property)")
        self.monitor.add_atom("(: Function Type)")
        self.monitor.add_atom("(: Expression Type)")
        
        # Add basic relationship definitions
        self.monitor.add_atom("(: function-has-property (--> Function Property Bool))")
        self.monitor.add_atom("(: Expression-Property (--> Expression Property Bool))")
        self.monitor.add_atom("(: adaptation-preserves-property (--> Property Bool))")
        self.monitor.add_atom("(: adaptation-violates-property (--> Property Bool))")
        
        # Load core ontology rules - adjust path as needed
        ontology_path = os.path.join(os.path.dirname(__file__), "metta", "proof_ontology.metta")
        if os.path.exists(ontology_path):
            self.monitor.load_metta_rules(ontology_path)
        
        # Initialize the immune system analyzer with live API configuration
        self.analyzer = ImmuneSystemProofAnalyzer(
            self.monitor.metta_space, 
            model_name=self.model_name,
            api_key=self.api_key
        )
        
        # Fix the analyzer's pattern processor to ensure it uses our monitor
        self.analyzer.pattern_processor.monitor = self.monitor
        
        # Initialize the proof generator directly
        self.proof_generator = MettaProofGenerator(
            self.monitor, 
            model_name=self.model_name, 
            api_key=self.api_key
        )
        
        # Initialize OpenAI client for direct API calls
        self.openai_client = OpenAIRequests(self.api_key, self.model_name)
    
    def tearDown(self):
        """Clean up after each test."""
        # Reset MeTTa space
        if hasattr(self, 'monitor') and self.monitor:
            self.monitor.metta_space = self.monitor.metta.space()
    
    def test_live_property_detection(self):
        """Test property detection using live API for pattern analysis."""
        # Define a test expression to analyze
        test_expression = {
            "expression": "index < array.length", 
            "natural_language": "Ensures index is within bounds of the array"
        }
        
        # Create a prompt to get property analysis from the API
        prompt = f"""
        Analyze this code expression and its description for software properties:
        
        Expression: {test_expression['expression']}
        Description: {test_expression['natural_language']}
        
        Return a list of software properties this expression enforces, such as:
        1. Bound checking
        2. Ordering preservation
        3. Null checking
        4. Termination guarantee
        5. Error handling
        
        Format your response as a JSON object with this structure:
        {{
            "detected_properties": [
                {{
                    "property_type": "property name",
                    "explanation": "why this property is present"
                }}
            ]
        }}
        
        ONLY RETURN THE JSON OBJECT, NO OTHER TEXT.
        """
        
        try:
            # Get real API response
            response = self.openai_client.get_completion_text([
                {"role": "system", "content": "You are a helpful assistant that analyzes code patterns."},
                {"role": "user", "content": prompt}
            ])
            
            # Extract JSON from response
            json_match = re.search(r'(\{[\s\S]*\})', response)
            if json_match:
                json_content = json_match.group(1)
            else:
                json_content = response
                
            # Parse the API response
            properties_data = json.loads(json_content)
            
            # Check that bound-check was detected by the API
            api_property_types = [p.get("property_type", "").lower() for p in properties_data.get("detected_properties", [])]
            self.assertTrue(any("bound" in p for p in api_property_types),
                           f"API did not detect bounds checking in the expression. Found: {api_property_types}")
            
            # Now test our pattern mapper with the same expression
            expr_id = "test_expr"
            self.monitor.add_atom(f"(: {expr_id} Expression)")
            
            # Add the expression to MeTTa
            metta_atoms = []
            self.analyzer.pattern_processor._add_property_annotations(
                metta_atoms, expr_id, test_expression["expression"], test_expression
            )
            
            # Add generated atoms to MeTTa space
            for atom in metta_atoms:
                self.monitor.add_atom(atom)
            
            # Check if our system detected the same property
            bound_check_result = self.monitor.query(f"(match &self (Expression-Property {expr_id} bound-check) True)")
            self.assertTrue(len(bound_check_result) > 0,
                           f"Pattern mapper failed to detect bound-check but API did")
            
            # Log success
            logger.info(f"Successfully detected bounds checking property with both API and pattern mapper")
            
        except Exception as e:
            self.fail(f"Error during live API property detection test: {str(e)}")
    
    def test_live_json_to_metta_conversion(self):
        """Test JSON to MeTTa conversion with real API-generated proof JSON."""
        # Create a prompt to get proof JSON from the API
        prompt = f"""
        Generate a formal proof for this Python function:
        
        ```python
        {self.binary_search}
        ```
        
        I need you to generate proof components in a structured JSON format with this schema:
        {{
            "proof_components": [
                {{
                    "type": "precondition",
                    "expression": "logical expression",
                    "natural_language": "explanation in natural language"
                }},
                {{
                    "type": "loop_invariant",
                    "location": "while_loop",
                    "expression": "invariant expression",
                    "natural_language": "explanation in natural language"
                }},
                {{
                    "type": "assertion",
                    "location": "code location identifier",
                    "expression": "assertion expression",
                    "natural_language": "explanation in natural language"
                }}
            ],
            "verification_strategy": {{
                "approach": "description of proof approach",
                "key_lemmas": ["list of key lemmas or principles used"]
            }}
        }}
        
        Focus on identifying loop invariants and bounds checking.
        ONLY RETURN THE JSON OBJECT, NO OTHER TEXT.
        """
        
        try:
            # Get real API response
            response = self.openai_client.get_completion_text([
                {"role": "system", "content": "You are a helpful assistant that generates formal proofs."},
                {"role": "user", "content": prompt}
            ])
            
            # Extract JSON from response
            json_match = re.search(r'(\{[\s\S]*\})', response)
            if json_match:
                json_content = json_match.group(1)
            else:
                json_content = response
                
            # Parse the API response
            proof_json = json.loads(json_content)
            
            # Verify the structure
            self.assertIn("proof_components", proof_json, "API response missing proof_components")
            self.assertIn("verification_strategy", proof_json, "API response missing verification_strategy")
            
            # Convert to MeTTa atoms
            metta_atoms = self.analyzer.pattern_processor._json_to_metta_proof(proof_json)
            
            # Add atoms to MeTTa space
            for atom in metta_atoms:
                self.monitor.add_atom(atom)
            
            # Check for loop invariants
            loop_inv_result = self.monitor.query("(match &self (LoopInvariant $loc $expr) ($loc $expr))")
            self.assertTrue(len(loop_inv_result) > 0,
                           f"Failed to create and query LoopInvariant atoms from API-generated JSON")
            
            # Check for property annotations - bounds checking should be detected
            property_result = self.monitor.query("(match &self (Expression-Property $expr $prop) ($expr $prop))")
            self.assertTrue(len(property_result) > 0,
                           f"Failed to create Expression-Property relationships from API-generated JSON")
            
            # Log success
            logger.info(f"Successfully converted API-generated JSON to MeTTa atoms")
            logger.info(f"Found {len(loop_inv_result)} loop invariants and {len(property_result)} property relationships")
            
        except Exception as e:
            self.fail(f"Error during live JSON to MeTTa conversion test: {str(e)}")
    
    def test_live_proof_generation(self):
        """Test full proof generation with real API calls."""
        # Define test function name
        func_name = "binary_search"
        
        try:
            # Generate proof using real API calls
            result = self.analyzer.analyze_function_for_proof(
                self.binary_search, 
                function_name=func_name,
                max_attempts=1
            )
            
            # Check that proof generation succeeded
            self.assertTrue(result["success"],
                           f"Live proof generation failed: {result.get('error', 'Unknown error')}")
            
            # Check that proof was added to MeTTa space
            verify_result = self.monitor.query(f"(match &self (verified-function {func_name}) True)")
            self.assertTrue(len(verify_result) > 0,
                           f"Failed to add verified-function atom to MeTTa space")
            
            # Check that at least one loop invariant was found
            invariant_result = self.monitor.query("(match &self (LoopInvariant $loc $expr) $loc)")
            self.assertTrue(len(invariant_result) > 0,
                           "No loop invariants found in generated proof")
            
            # Check that at least one property was identified
            property_result = self.monitor.query("(match &self (Expression-Property $expr $prop) $prop)")
            self.assertTrue(len(property_result) > 0,
                           "No properties found in generated proof")
            
            # Log success with detailed information
            logger.info(f"Successfully generated proof for {func_name}")
            logger.info(f"Found {len(invariant_result)} loop invariants")
            logger.info(f"Identified properties: {property_result}")
            
        except Exception as e:
            self.fail(f"Error during live proof generation test: {str(e)}")
    
    def test_live_adaptation_verification(self):
        """Test adaptation verification with real API calls."""
        # Create a modified binary search that uses different variable names
        modified_binary_search = self.binary_search.replace("left", "lo").replace("right", "hi")
        
        try:
            # Verify adaptation using real API calls
            result = self.analyzer.verify_adaptation(
                self.binary_search,
                modified_binary_search,
                ["maintains array bounds", "handles target not found case"]
            )
            
            # Check if verification succeeded
            self.assertTrue(result["success"],
                           f"Live adaptation verification failed: {result.get('error', 'Unknown error')}")
            
            # Check that properties were preserved
            self.assertGreaterEqual(len(result.get("preserved_properties", [])), 1,
                                 "No properties were preserved in adaptation verification")
            
            # Verify that the adaptation-preserves-property atoms were added to MeTTa
            preserve_result = self.monitor.query("(match &self (adaptation-preserves-property $prop) $prop)")
            self.assertTrue(len(preserve_result) > 0,
                           "Failed to add adaptation-preserves-property atoms to MeTTa space")
            
            # Log success
            logger.info(f"Successfully verified adaptation with real API")
            logger.info(f"Preserved properties: {result.get('preserved_properties', [])}")
            
        except Exception as e:
            self.fail(f"Error during live adaptation verification test: {str(e)}")
    
    def test_live_property_mapping(self):
        """Test mapping of property descriptions to MeTTa atoms with real API."""
        # List of property descriptions to test
        property_descriptions = [
            "maintains array bounds",
            "preserves element ordering",
            "handles null inputs",
            "ensures termination",
            "handles errors properly"
        ]
        
        try:
            # Create a prompt to get property mappings from the API
            properties_str = "\n".join([f"- {p}" for p in property_descriptions])
            prompt = f"""
            Map these software property descriptions to formal property types:
            
            {properties_str}
            
            For each property, determine the most appropriate formal property type from this list:
            - bound-check
            - ordering-check
            - null-check
            - termination-guarantee
            - error-handling
            
            Return your mappings as a JSON object with this structure:
            {{
                "property_mappings": [
                    {{
                        "description": "property description",
                        "formal_type": "formal property type"
                    }}
                ]
            }}
            
            ONLY RETURN THE JSON OBJECT, NO OTHER TEXT.
            """
            
            # Get real API response
            response = self.openai_client.get_completion_text([
                {"role": "system", "content": "You are a helpful assistant that maps software properties."},
                {"role": "user", "content": prompt}
            ])
            
            # Extract JSON from response
            json_match = re.search(r'(\{[\s\S]*\})', response)
            if json_match:
                json_content = json_match.group(1)
            else:
                json_content = response
                
            # Parse the API response
            mappings_data = json.loads(json_content)
            
            # Compare API mappings with our pattern mapper
            for mapping in mappings_data.get("property_mappings", []):
                description = mapping.get("description")
                api_type = mapping.get("formal_type")
                
                if description and api_type:
                    # Get our pattern mapper's mapping
                    mapper_type = self.pattern_mapper.map_requirement_to_property(description)
                    
                    # Log comparison
                    logger.info(f"Property: '{description}'")
                    logger.info(f"  API mapping: {api_type}")
                    logger.info(f"  Pattern mapper: {mapper_type}")
                    
                    # For the bound-check case specifically, verify agreement
                    if "bound" in description.lower():
                        self.assertEqual(api_type, "bound-check",
                                      f"API did not map '{description}' to bound-check")
                        self.assertEqual(mapper_type, "bound-check",
                                      f"Pattern mapper did not map '{description}' to bound-check")
            
            # Overall success
            logger.info("Successfully compared API property mappings with pattern mapper")
            
        except Exception as e:
            self.fail(f"Error during live property mapping test: {str(e)}")


if __name__ == "__main__":
    import re  # Import re here for the test
    unittest.main()