import json
import logging
from typing import Dict, List, Any, Optional

from hyperon import *
from reflectors.dynamic_monitor import DynamicMonitor
from reflectors.static_analyzer import decompose_function
from proofs.processor import ProofProcessorWithPatterns
from proofs.requester import OpenAIRequests

class MettaProofGenerator:
    """
    A generator for formal proofs using LLMs with MeTTa for verification.
    Uses a structured JSON IR as an intermediate representation between LLMs and MeTTa.
    """
    
    def __init__(self, monitor: Optional[DynamicMonitor] = None, model_name: str = "gpt-4o-mini", api_key: str = None):
        """
        Initialize the proof generator.
        
        Args:
            monitor: Existing DynamicMonitor instance (creates new one if None)
            model_name: Name of the LLM model to use for generation
            api_key: OpenAI API key (required for API calls)
        """
        self.monitor = monitor or DynamicMonitor()
        self.pattern_processor = ProofProcessorWithPatterns(self.monitor)
        self.model_name = model_name
        self.api_key = api_key
        self.openai_client = OpenAIRequests(api_key, model_name) if api_key else None
        self.inference_attempts = 0
        self.debugging_attempts = 0
        
        # Verification statistics
        self.verification_stats = {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "debugging_successes": 0
        }
    
    def generate_proof(self, function_code: str, context: Optional[str] = None, 
                  max_attempts: int = 3, debug_failed_proofs: bool = True) -> Dict[str, Any]:
        """
        Generate a formal proof for a Python function.
        
        Args:
            function_code: Source code of the function (as string)
            context: Optional domain context information
            max_attempts: Maximum number of generation attempts
            debug_failed_proofs: Whether to attempt debugging failed proofs
            
        Returns:
            Dictionary containing the result of proof generation
        """
        if not self.api_key or not self.openai_client:
            return {"success": False, "error": "OpenAI API key not provided. Cannot generate proof."}
            
        # Run static analysis on the function
        analysis = decompose_function(function_code)
        if "error" in analysis and analysis["error"]:
            return {"success": False, "error": f"Static analysis failed: {analysis['error']}"}
        
        # Extract specifications
        specs = self._extract_specifications(analysis)
        if not specs:
            # Generate specifications using LLM if not found in analysis
            specs = self._generate_specifications(function_code)
        
        # Add specifications to MeTTa space
        self._add_specs_to_metta(specs)
        
        # Extract function name if available
        function_name = analysis.get("function_name", "unnamed_function")
        
        # Generate proof using JSON IR
        for attempt in range(max_attempts):
            self.inference_attempts += 1
            self.verification_stats["attempts"] += 1
            
            # Generate proof components using LLM with JSON IR
            proof_json = self._generate_proof_json(function_code, specs, context)
            
            # Convert JSON IR to MeTTa representation
            metta_proof = self._json_to_metta_proof(proof_json)
            
            # Try to verify the proof
            verification_result = self._verify_proof(function_code, specs, metta_proof)
            
            if verification_result["success"]:
                self.verification_stats["successes"] += 1
                return {
                    "success": True,
                    "proof": metta_proof,
                    "function": function_code,
                    "function_name": function_name,
                    "specifications": specs,
                    "json_ir": proof_json,
                    "attempts": attempt + 1
                }
            
            # If verification failed and we should debug
            if debug_failed_proofs and verification_result.get("error_message"):
                self.debugging_attempts += 1
                debug_result = self._debug_proof(
                    function_code, specs, proof_json, verification_result["error_message"]
                )
                
                if debug_result["success"]:
                    self.verification_stats["debugging_successes"] += 1
                    return {
                        "success": True,
                        "proof": debug_result["proof"],
                        "function": function_code,
                        "function_name": function_name,
                        "specifications": specs,
                        "json_ir": debug_result["json_ir"],
                        "debug_message": verification_result["error_message"],
                        "was_debugged": True
                    }
        
        # If all attempts failed
        self.verification_stats["failures"] += 1
        return {
            "success": False,
            "function": function_code,
            "function_name": function_name if "function_name" in analysis else "unnamed_function",
            "specifications": specs,
            "attempts": max_attempts,
            "error": "Maximum proof generation attempts reached"
        }
    
    def _extract_specifications(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract specifications from static analysis result."""
        # This is simplified - a real implementation would extract preconditions, 
        # postconditions, and invariants from the static analysis
        specs = {
            "preconditions": [],
            "postconditions": [],
            "invariants": []
        }
        
        for atom in analysis.get("metta_atoms", []):
            if "requires" in atom:
                specs["preconditions"].append(atom)
            elif "ensures" in atom:
                specs["postconditions"].append(atom)
            elif "invariant" in atom:
                specs["invariants"].append(atom)
                
        return specs
    
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
            {"role": "system", "content": "You are a helpful assistant specialized in formal verification and proof generation."},
            {"role": "user", "content": prompt}
        ]
        
        return self.openai_client.get_completion_text(messages)
    
    def _generate_specifications(self, function_code: str) -> Dict[str, Any]:
        """Generate specifications using LLM when not found in code."""
        prompt = f"""
        Please analyze this Python function and generate formal specifications:
        
        ```python
        {function_code}
        ```
        
        Return specifications as a JSON object with the following structure:
        {{
            "preconditions": [list of conditions that must be true before function execution],
            "postconditions": [list of conditions that must be true after function execution],
            "invariants": [list of conditions that must be maintained]
        }}

        ONLY RETURN THE JSON OBJECT, NO OTHER TEXT.
        """
        
        try:
            # Call OpenAI API directly
            llm_response = self._call_openai_api(prompt)
            
            # Extract JSON from response
            json_content = self._extract_json_from_llm_response(llm_response)
            specs = json.loads(json_content)
            
            # Ensure the expected structure
            if not isinstance(specs, dict):
                specs = {}
            specs.setdefault("preconditions", [])
            specs.setdefault("postconditions", [])
            specs.setdefault("invariants", [])
            
            return specs
            
        except Exception as e:
            logging.error(f"Failed to generate specifications: {e}")
            return {"preconditions": [], "postconditions": [], "invariants": []}
    
    def _add_specs_to_metta(self, specs: Dict[str, Any]) -> None:
        """Add specifications to MeTTa space."""
        for precond in specs.get("preconditions", []):
            if isinstance(precond, str):
                self.monitor.add_atom(f"(specification-precondition {precond})")
        
        for postcond in specs.get("postconditions", []):
            if isinstance(postcond, str):
                self.monitor.add_atom(f"(specification-postcondition {postcond})")
        
        for inv in specs.get("invariants", []):
            if isinstance(inv, str):
                self.monitor.add_atom(f"(specification-invariant {inv})")
    
    def _generate_proof_json(self, function_code: str, specs: Dict[str, Any], 
                            context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate proof components using LLM with structured JSON IR.
        """
        # Prepare prompt with specifications
        preconditions_str = "\n".join([f"- {p}" for p in specs.get("preconditions", [])])
        postconditions_str = "\n".join([f"- {p}" for p in specs.get("postconditions", [])])
        
        context_info = f"Domain context: {context}\n\n" if context else ""
        
        prompt = f"""
        {context_info}
        Please generate a formal proof for this Python function:
        
        ```python
        {function_code}
        ```
        
        SPECIFICATIONS:
        Preconditions:
        {preconditions_str or "None specified"}
        
        Postconditions:
        {postconditions_str or "None specified"}
        
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
                    "location": "loop identifier",
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
        
        The approach should focus on identifying necessary loop invariants, key assertions, and the logical sequencing of proof elements.
        Please ensure all expressions are logically precise and all components are properly typed. ONLY RETURN THE JSON OBJECT, NO OTHER TEXT.
        """
        
        try:
            # Call OpenAI API directly
            llm_response = self._call_openai_api(prompt)
            
            # Extract JSON from response
            json_content = self._extract_json_from_llm_response(llm_response)
            proof_json = json.loads(json_content)
            
            # Validate the structure
            if not isinstance(proof_json, dict) or "proof_components" not in proof_json:
                raise ValueError("Invalid proof JSON structure")
            
            return proof_json
            
        except Exception as e:
            logging.error(f"Failed to generate proof JSON: {e}")
            # Return a minimal valid structure
            return {
                "proof_components": [],
                "verification_strategy": {
                    "approach": "Failed to generate",
                    "key_lemmas": []
                },
                "error": str(e)
            }
    
    def _json_to_metta_proof(self, proof_json: Dict[str, Any]) -> List[str]:
        """Convert proof component JSON to MeTTa representation."""
        return self.pattern_processor._json_to_metta_proof(proof_json)
    
    def _verify_proof(self, function_code: str, specs: Dict[str, Any], 
                     metta_proof: List[str]) -> Dict[str, Any]:
        """
        Verify the proof using MeTTa.
        """
        verification_result = {
            "success": False,
            "error_message": None
        }
        
        try:
            # Add proof components to MeTTa space
            for component in metta_proof:
                self.monitor.add_atom(component)
            
            # Create a temporary verification script that would use MeTTa's symbolic execution
            # This is a simplified placeholder - real implementation would do actual formal verification
            
            # Check if proof satisfies specifications
            # 1. Check if all loop invariants are valid
            invariant_check = self.monitor.query("(match &self (LoopInvariant $loc $expr) $expr)")
            if not invariant_check:
                verification_result["error_message"] = "No loop invariants found"
                return verification_result
            
            # 2. Check if invariants imply postconditions
            # This is a simplified check - would be more complex in real implementation
            postcondition_check = self.monitor.query("""
                (match &self (LoopInvariant $loc $inv) 
                       (specification-postcondition $post)
                       (implies $inv $post))
            """)
            
            # 3. Check if preconditions establish invariants
            precondition_check = self.monitor.query("""
                (match &self (specification-precondition $pre) 
                       (LoopInvariant $loc $inv)
                       (implies $pre $inv))
            """)
            
            # Simple verification logic for demonstration
            # Real verification would be much more sophisticated
            if invariant_check and (postcondition_check or precondition_check):
                verification_result["success"] = True
            else:
                verification_result["error_message"] = "Verification failed: invariants don't satisfy specifications"
            
            return verification_result
            
        except Exception as e:
            logging.error(f"Verification error: {e}")
            verification_result["error_message"] = f"Verification error: {str(e)}"
            return verification_result
    
    def _debug_proof(self, function_code: str, specs: Dict[str, Any], 
                    proof_json: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """
        Debug a failed proof attempt using LLM.
        """
        # Serialize the proof JSON for inclusion in prompt
        proof_json_str = json.dumps(proof_json, indent=2)
        
        prompt = f"""
        I need you to fix this proof that failed verification. Here's the function:
        
        ```python
        {function_code}
        ```
        
        The original proof attempt (in JSON format):
        ```json
        {proof_json_str}
        ```
        
        The verification error was:
        {error_message}
        
        Please fix the proof by modifying the JSON. Focus on:
        1. Correcting any logical errors in the expressions
        2. Adding missing loop invariants or assertions
        3. Ensuring the proof components are properly connected
        
        Return the corrected proof in the same JSON format. ONLY RETURN THE JSON OBJECT, NO OTHER TEXT.
        """
        
        try:
            # Call OpenAI API directly
            llm_response = self._call_openai_api(prompt)
            
            # Extract JSON from response
            json_content = self._extract_json_from_llm_response(llm_response)
            fixed_proof_json = json.loads(json_content)
            
            # Convert to MeTTa and verify again
            metta_proof = self._json_to_metta_proof(fixed_proof_json)
            verification_result = self._verify_proof(function_code, specs, metta_proof)
            
            return {
                "success": verification_result["success"],
                "proof": metta_proof,
                "json_ir": fixed_proof_json,
                "debug_error": verification_result.get("error_message") if not verification_result["success"] else None
            }
            
        except Exception as e:
            logging.error(f"Debugging failed: {e}")
            return {
                "success": False,
                "error": f"Debugging failed: {str(e)}"
            }
    
    def _extract_json_from_llm_response(self, llm_response: Any) -> str:
        """
        Extract JSON content from LLM response.
        """
        if not llm_response:
            return "{}"
            
        response_str = str(llm_response)
        
        # Look for JSON content between triple backticks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_str)
        if json_match:
            return json_match.group(1)
        
        # Look for JSON content between curly braces
        json_match = re.search(r'(\{[\s\S]*\})', response_str)
        if json_match:
            return json_match.group(1)
        
        # If no JSON-like content found, return the whole response
        # (this might fail when parsing but at least we tried)
        return response_str
    
    def _escape_for_metta(self, expr: str) -> str:
        """
        Escape a string for inclusion in MeTTa.
        """
        # Replace characters that might cause issues in MeTTa
        escaped = expr.replace('"', '\\"')
        
        # If expression contains spaces or special characters, wrap in quotes
        if ' ' in escaped or any(c in escaped for c in '()[]{}'):
            escaped = f'"{escaped}"'
        
        return escaped
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about proof generation attempts.
        """
        stats = self.verification_stats.copy()
        stats["inference_attempts"] = self.inference_attempts
        stats["debugging_attempts"] = self.debugging_attempts
        
        if stats["attempts"] > 0:
            stats["success_rate"] = stats["successes"] / stats["attempts"]
            stats["debug_success_rate"] = stats["debugging_successes"] / stats["debugging_attempts"] if stats["debugging_attempts"] > 0 else 0
        else:
            stats["success_rate"] = 0
            stats["debug_success_rate"] = 0
            
        return stats