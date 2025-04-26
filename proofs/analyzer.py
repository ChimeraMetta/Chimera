import os
import logging
import requests
import json
import re
from typing import Dict, List, Any

from hyperon import *
from dynamic_monitor import DynamicMonitor, monitor
from static_analyzer import decompose_function
from proofs.generator import MettaProofGenerator
from proofs.processor import ProofProcessorWithPatterns

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("proof_system")

class OpenAIRequests:
    """
    A minimal OpenAI API client using the requests library instead of the official SDK.
    This avoids the dependency issues with PyPy 3.8.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: Your OpenAI API key
            model: The model to use for completions
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        
    def chat_completion(self, messages: list, temperature: float = 0.3, max_tokens: int = 2048) -> Dict[str, Any]:
        """
        Send a chat completion request to the OpenAI API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness (0 to 1, lower is more deterministic)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The API response as a dictionary
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
        except Exception as e:
            logging.error(f"OpenAI API request failed: {e}")
            return {"error": str(e)}
    
    def get_completion_text(self, messages: list, temperature: float = 0.3, max_tokens: int = 2048) -> str:
        """
        Get just the completion text from a chat completion request.
        
        Args:
            messages: List of message dictionaries
            temperature: Controls randomness
            max_tokens: Maximum tokens to generate
            
        Returns:
            The text of the completion, or an error message
        """
        response = self.chat_completion(messages, temperature, max_tokens)
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            error_msg = response.get("error", {}).get("message", "Unknown error")
            logging.error(f"Failed to extract completion text: {error_msg}")
            return f"Error: {error_msg}"

class ImmuneSystemProofAnalyzer:
    """
    Extends the codebase analyzer with proof generation capabilities.
    Acts as a core component of the software immune system for donor identification.
    """
    
    def __init__(self, metta_space=None, model_name="gpt-4", api_key=None):
        """Initialize the proof analyzer."""
        self.monitor = monitor if monitor else DynamicMonitor(metta_space)
        self.model_name = model_name
        self.api_key = api_key
        
        # Initialize OpenAI client if API key is provided
        self.openai_client = OpenAIRequests(api_key, model_name) if api_key else None
        
        # Initialize components with OpenAI client
        self.proof_generator = MettaProofGenerator(self.monitor, model_name, api_key)
        self.pattern_processor = ProofProcessorWithPatterns(self.monitor, model_name, api_key)
        
        # Load ontology rules for proof verification
        self._load_proof_ontology()
        
    def _load_proof_ontology(self):
        """Load MeTTa ontology rules for proof reasoning."""
        ontology_file = os.path.join(os.path.dirname(__file__), "metta", "proof_ontology.metta")
        if os.path.exists(ontology_file):
            success = self.monitor.load_metta_rules(ontology_file)
            if success:
                logger.info(f"Successfully loaded proof ontology from {ontology_file}")
            else:
                logger.warning(f"Failed to load proof ontology from {ontology_file}")
        else:
            logger.warning(f"Proof ontology file not found: {ontology_file}")
    
    def _call_openai_api(self, prompt: str) -> str:
        """
        Call OpenAI API with the given prompt using requests-based client.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            The response text from the API
        """
        if not self.openai_client:
            logging.error("OpenAI client not initialized. API key may be missing.")
            return "Error: OpenAI client not initialized"
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in formal verification and software analysis."},
            {"role": "user", "content": prompt}
        ]
        
        return self.openai_client.get_completion_text(messages)
            
    def analyze_function_for_proof(self, function_code: str, function_name: str = None,
                                 context: str = None, max_attempts: int = 3) -> Dict[str, Any]:
        """
        Analyze a function and generate formal proof of its correctness.
        
        Args:
            function_code: Source code of the function
            function_name: Name of the function (optional)
            context: Domain context for the function
            max_attempts: Maximum number of proof generation attempts
            
        Returns:
            Dictionary with analysis results including proof if successful
        """
        if not self.api_key:
            return {"success": False, "error": "OpenAI API key not provided. Cannot analyze function."}
            
        logger.info(f"Analyzing function{' ' + function_name if function_name else ''} for proof generation")
        
        # Run static analysis
        analysis = decompose_function(function_code)
        if "error" in analysis and analysis["error"]:
            logger.error(f"Static analysis failed: {analysis['error']}")
            return {"success": False, "error": f"Static analysis failed: {analysis['error']}"}
        
        # Add static analysis to MeTTa
        for atom in analysis.get("metta_atoms", []):
            self.monitor.add_atom(atom)
        
        # Generate proof
        proof_result = self.proof_generator.generate_proof(
            function_code, context, max_attempts
        )
        
        # Add proof-specific metrics to the result
        if proof_result["success"]:
            # Add the proof to MeTTa space
            logger.info(f"Successfully generated proof")
            
            # Mark the function as verified in MeTTa space
            func_name = function_name or "unnamed_function"
            self.monitor.add_atom(f"(verified-function {func_name})")
            
            # Add generated invariants to MeTTa space
            for proof_component in proof_result.get("proof", []):
                if proof_component.startswith("(LoopInvariant"):
                    self.monitor.add_atom(f"(function-invariant {func_name} {proof_component})")
            
            # Analyze proof complexity
            proof_complexity = len(proof_result.get("proof", []))
            proof_result["complexity"] = proof_complexity
            self.monitor.add_atom(f"(proof-complexity {func_name} {proof_complexity})")
        else:
            logger.warning(f"Failed to generate proof after {max_attempts} attempts")
        
        return proof_result
    
    def identify_potential_donors(self, target_properties: List[str], 
                                candidate_functions: List[str],
                                context: str = None) -> List[Dict[str, Any]]:
        """
        Identify suitable donor functions based on proof properties.
        
        Args:
            target_properties: List of required properties for donors
            candidate_functions: List of candidate function code
            context: Domain context for the functions
            
        Returns:
            List of suitable donors with their proofs and compatibility scores
        """
        if not self.api_key:
            return []
            
        logger.info(f"Identifying potential donors from {len(candidate_functions)} candidates")
        
        suitable_donors = []
        
        for i, func_code in enumerate(candidate_functions):
            logger.info(f"Analyzing candidate {i+1}/{len(candidate_functions)}")
            
            # Generate proof for candidate function
            proof_result = self.analyze_function_for_proof(
                func_code, f"candidate_{i}", context
            )
            
            if not proof_result["success"]:
                continue
            
            # Check if the proof satisfies target properties
            compatibility_score = self._calculate_compatibility(proof_result, target_properties)
            
            if compatibility_score > 0.7:  # Threshold for suitability
                suitable_donors.append({
                    "function": func_code,
                    "proof": proof_result.get("proof", []),
                    "compatibility_score": compatibility_score,
                    "properties_satisfied": self._get_satisfied_properties(proof_result, target_properties)
                })
        
        # Sort by compatibility score
        suitable_donors.sort(key=lambda x: x["compatibility_score"], reverse=True)
        
        logger.info(f"Found {len(suitable_donors)} suitable donors")
        return suitable_donors

    def _get_satisfied_properties(self, proof_result: Dict[str, Any], 
                                target_properties: List[str]) -> List[str]:
        """Get the list of target properties satisfied by this proof."""
        satisfied = []
        
        # Extract JSON IR for easier property checking
        json_ir = proof_result.get("json_ir", {})
        
        # Check each property
        for prop in target_properties:
            if self._check_property_satisfied(json_ir, prop):
                satisfied.append(prop)
                
                # Also add to MeTTa space with proper atoms
                if "function" in proof_result:
                    func_name = proof_result.get("function_name", "unnamed_function")
                    self._add_property_to_metta(func_name, prop)
        
        return satisfied
        
    def _add_property_to_metta(self, func_name: str, property: str) -> None:
        """Add a property to MeTTa space."""
        self.pattern_processor._add_property_to_metta(func_name, property)
            
    def _python_to_metta_value(self, value) -> str:
        """
        Convert Python values to MeTTa atom representations.
        Avoid string matching by creating proper atoms.
        """
        if value is None:
            return "None"
            
        if isinstance(value, (int, float)):
            return str(value)
            
        if isinstance(value, bool):
            return "True" if value else "False"
            
        if isinstance(value, str):
            # Create a string atom with proper escaping
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
            
        if isinstance(value, list):
            # Create a proper list atom
            if not value:
                return "Empty"  # Empty list atom
                
            items = [self._python_to_metta_value(item) for item in value]
            return f"(List {' '.join(items)})"
            
        if isinstance(value, dict):
            # Create a proper dict atom 
            if not value:
                return "(Dict)"  # Empty dict atom
                
            entries = []
            for k, v in value.items():
                key_atom = self._python_to_metta_value(k)
                value_atom = self._python_to_metta_value(v)
                entries.append(f"(KeyValue {key_atom} {value_atom})")
                
            return f"(Dict {' '.join(entries)})"
            
        # Default for other types - create an opaque atom
        type_name = type(value).__name__
        return f"({type_name} \"{str(value)}\")"
    
    def _calculate_compatibility(self, proof_result: Dict[str, Any], target_properties: List[str]) -> float:
        """Calculate compatibility score between a proof and target properties."""
        satisfied = 0
        json_ir = proof_result.get("json_ir", {})
        
        for prop in target_properties:
            if self._check_property_satisfied(json_ir, prop):
                satisfied += 1
        
        return satisfied / len(target_properties) if target_properties else 0.0
    
    def _check_property_satisfied(self, json_ir: Dict[str, Any], property: str) -> bool:
        """
        Check if a specific property is satisfied by the proof.
        Instead of string matching, we map properties to specific patterns to look for.
        """
        components = json_ir.get("proof_components", [])
        property_lower = property.lower()
        
        # Map properties to specific patterns to check for
        property_patterns = {
            "maintains array bounds": ["<=", ">=", "<", ">", "index", "bound", "range", "within"],
            "preserves order": ["sort", "order", "increas", "decreas", "ascend", "descend"],
            "terminates": ["terminat", "halt", "progress", "decrease", "invariant", "loop"],
            "handles empty arrays": ["empty", "null", "none", "length", "size", "zero"],
            "returns correct index": ["index", "position", "return", "correct", "find"],
            "handles target not found": ["not found", "-1", "error", "exception", "target", "miss"]
        }
        
        # Get patterns to look for based on the requested property
        patterns = []
        for key, pattern_list in property_patterns.items():
            if any(term in property_lower for term in key.split()):
                patterns.extend(pattern_list)
        
        # If no specific pattern found, use the property terms themselves
        if not patterns:
            patterns = property_lower.split()
        
        # Look for patterns in expressions and natural language descriptions
        for component in components:
            expr = component.get("expression", "").lower()
            desc = component.get("natural_language", "").lower()
            
            # Check if any pattern is found in the expression or description
            if any(pattern in expr for pattern in patterns) or any(pattern in desc for pattern in patterns):
                return True
        
        # Also check verification strategy
        strategy = json_ir.get("verification_strategy", {})
        approach = strategy.get("approach", "").lower()
        
        if any(pattern in approach for pattern in patterns):
            return True
        
        # Check lemmas
        for lemma in strategy.get("key_lemmas", []):
            lemma_lower = lemma.lower()
            if any(pattern in lemma_lower for pattern in patterns):
                return True
        
        return False
    
    def _map_property_to_metta_atom(self, property: str) -> str:
        """
        Map a property string to a corresponding MeTTa atom.
        This avoids string-based pattern matching in MeTTa.
        """
        property_lower = property.lower()
        
        if "bound" in property_lower or "index" in property_lower:
            return "(bound-checking)"
        elif "order" in property_lower or "sort" in property_lower:
            return "(ordering-preservation)"
        elif "null" in property_lower or "empty" in property_lower:
            return "(null-check)"
        elif "terminat" in property_lower or "loop" in property_lower:
            return "(termination)"
        elif "error" in property_lower or "not found" in property_lower:
            return "(error-handling)"
        else:
            # Create a custom property atom
            safe_name = property_lower.replace(" ", "-").replace(":", "")
            return f"(property-{safe_name})"
    
    def verify_adaptation(self, original_func: str, adapted_func: str, essential_properties: List[str]) -> Dict[str, Any]:
        """Verify that an adaptation preserves essential properties."""
        if not self.api_key:
            return {"success": False, "error": "OpenAI API key not provided. Cannot verify adaptation."}
            
        # Delegate to pattern processor for adaptation verification
        return self.pattern_processor.verify_adaptation(
            original_func, adapted_func, essential_properties
        )
    
    def incorporate_execution_evidence(self, function_code: str, inputs: List[Any],
                                     outputs: List[Any], states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Incorporate runtime evidence to enhance proof generation.
        
        Args:
            function_code: Function source code
            inputs: List of function inputs from executions
            outputs: List of function outputs from executions
            states: List of execution state snapshots
            
        Returns:
            Enhanced proof result
        """
        if not self.api_key:
            return {"success": False, "error": "OpenAI API key not provided. Cannot incorporate execution evidence."}
            
        logger.info(f"Incorporating execution evidence from {len(inputs)} executions")
        
        # Add execution evidence to MeTTa
        for i, (input_val, output_val, state) in enumerate(zip(inputs, outputs, states)):
            # Convert Python values to MeTTa representations
            input_metta = self._python_to_metta_value(input_val)
            output_metta = self._python_to_metta_value(output_val)
            
            # Add execution record
            execution_atom = f"(execution-record {i} {input_metta} {output_metta})"
            self.monitor.add_atom(execution_atom)
            
            # Add key state variables
            for var_name, value in state.items():
                value_metta = self._python_to_metta_value(value)
                state_atom = f"(execution-state {i} {var_name} {value_metta})"
                self.monitor.add_atom(state_atom)
        
        # Generate proof with execution evidence
        # We can provide a hint to the LLM about the existence of execution evidence
        context = f"Function has {len(inputs)} execution traces available"
        
        proof_result = self.analyze_function_for_proof(
            function_code, 
            context=context, 
            max_attempts=5  # More attempts since we have more information
        )
        
        if proof_result["success"]:
            proof_result["used_execution_evidence"] = True
            
        return proof_result