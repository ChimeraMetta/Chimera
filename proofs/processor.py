from typing import Dict, List, Any
import logging
import json
import re
from proofs.pattern_mapper import PatternMapper
from proofs.requester import OpenAIRequests

class ProofProcessorWithPatterns:
    """Integration code for the proof generation system with improved pattern mapping."""
    
    def __init__(self, monitor=None, model_name="gpt-4o-mini", api_key=None):
        """Initialize with the pattern mapper and monitor."""
        self.monitor = monitor
        self.pattern_mapper = PatternMapper()
        self.model_name = model_name
        self.api_key = api_key
        self.openai_client = OpenAIRequests(api_key, model_name) if api_key else None
        
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
            {"role": "system", "content": "You are a helpful assistant specialized in formal verification and pattern analysis."},
            {"role": "user", "content": prompt}
        ]

        logging.info("Messages: " + str(messages))
        logging.info("API key: " + self.api_key)
        
        return self.openai_client.get_completion_text(messages)
    
    def _add_property_annotations(self, metta_components: List[str], expr_id: str, 
                                expr_content: str, component: Dict[str, Any]) -> None:
        """
        Add property annotations for expressions based on content and metadata.
        Uses the pattern mapper to identify patterns and create proper MeTTa atoms.
        """
        # Get natural language description if available
        description = component.get("natural_language", "")
        
        # Explicitly create the merged content for pattern search - this ensures both
        # expression content and description are considered together
        content_for_pattern_search = f"{expr_content} {description}"
        
        # Use pattern mapper to identify patterns on the combined content
        patterns = self.pattern_mapper.identify_patterns(content_for_pattern_search)
        
        # If no patterns found with the combined approach, try separate searches
        if not patterns:
            # Try just the expression
            patterns = self.pattern_mapper.identify_patterns(expr_content)
            
            # If still no patterns, try just the description
            if not patterns and description:
                patterns = self.pattern_mapper.identify_patterns(description)
        
        # Add pattern atoms to MeTTa components
        for pattern_atom, property_type in patterns:
            # First ensure the property type exists
            metta_components.append(f"(: {property_type} Property)")
            
            # Relate the expression to the property
            expression_property = f"(Expression-Property {expr_id} {property_type})"
            if expression_property not in metta_components:  # Avoid duplicates
                metta_components.append(expression_property)
            
            # Add pattern-property relationship
            pattern_property = f"(pattern-property {pattern_atom} {property_type})"
            if pattern_property not in metta_components:  # Avoid duplicates
                metta_components.append(pattern_property)
        
        # If no patterns were found, add a generic property based on the component type
        if not patterns:
            # Map component type to a property
            component_type = component.get("type", "unknown")
            if component_type == "loop_invariant":
                property_type = "termination-guarantee"
            elif "bound" in expr_content.lower() or "index" in expr_content.lower():
                property_type = "bound-check"
            elif "order" in expr_content.lower() or "sort" in expr_content.lower():
                property_type = "ordering-check"
            elif "null" in expr_content.lower() or "empty" in expr_content.lower():
                property_type = "null-check"
            elif "error" in expr_content.lower() or "exception" in expr_content.lower():
                property_type = "error-handling"
            else:
                property_type = "generic-property"
            
            # Ensure property type exists
            metta_components.append(f"(: {property_type} Property)")
            
            # Add expression-property relationship
            metta_components.append(f"(Expression-Property {expr_id} {property_type})")
    
    def _map_property_to_metta_atom(self, property: str) -> str:
        """
        Map a property string to a corresponding MeTTa atom using the pattern mapper.
        """
        return self.pattern_mapper.map_requirement_to_property(property)
    
    def _check_property_satisfied(self, json_ir: Dict[str, Any], property: str) -> bool:
        """
        Check if a specific property is satisfied by the proof using the pattern mapper.
        """
        components = json_ir.get("proof_components", [])
        property_type = self.pattern_mapper.map_requirement_to_property(property)
        
        # Look for expressions with the required property type
        for component in components:
            expr = component.get("expression", "")
            desc = component.get("natural_language", "")
            
            # Use pattern mapper to identify patterns
            patterns = self.pattern_mapper.identify_patterns(expr, desc)
            
            # Check if any pattern maps to the required property
            for _, pattern_property in patterns:
                if pattern_property == property_type:
                    return True
        
        # Also check verification strategy
        strategy = json_ir.get("verification_strategy", {})
        approach = strategy.get("approach", "")
        
        # Check strategy approach with pattern mapper
        strategy_patterns = self.pattern_mapper.identify_patterns(approach)
        if any(prop == property_type for _, prop in strategy_patterns):
            return True
        
        # Check lemmas
        for lemma in strategy.get("key_lemmas", []):
            lemma_patterns = self.pattern_mapper.identify_patterns(lemma)
            if any(prop == property_type for _, prop in lemma_patterns):
                return True
        
        return False
    
    def _json_to_metta_proof(self, proof_json: Dict[str, Any]) -> List[str]:
        """
        Convert proof component JSON to MeTTa representation.
        Creates proper atoms with descriptive pattern names.
        """
        metta_components = []
        
        try:
            # First, ensure necessary type definitions are included
            metta_components.append("(: Expression Type)")
            metta_components.append("(: Property Type)")
            metta_components.append("(: bound-check Property)")
            metta_components.append("(: ordering-check Property)")
            metta_components.append("(: null-check Property)")
            metta_components.append("(: termination-guarantee Property)")
            metta_components.append("(: error-handling Property)")
            metta_components.append("(: Expression-Property (--> Expression Property Bool))")
            
            # Process each proof component
            for component in proof_json.get("proof_components", []):
                component_type = component.get("type", "unknown")
                
                if component_type == "precondition":
                    # Create a unique atom ID for the expression
                    expr_id = f"precond_{len(metta_components)}"
                    expr_content = component.get("expression", "")
                    
                    # Add the expression as a proper atom definition
                    metta_components.append(f"(: {expr_id} Expression)")
                    metta_components.append(f"(= ({expr_id}) \"{self._escape_for_metta(expr_content)}\")")
                    
                    # Reference the atom in the Precondition
                    metta_components.append(f"(Precondition {expr_id})")
                    
                    # Add property annotations using pattern mapper
                    self._add_property_annotations(metta_components, expr_id, expr_content, component)
                
                elif component_type == "loop_invariant":
                    location = component.get("location", "unknown_loop")
                    expr_id = f"inv_{location}_{len(metta_components)}"
                    expr_content = component.get("expression", "")
                    
                    # Create atoms for the expression
                    metta_components.append(f"(: {expr_id} Expression)")
                    metta_components.append(f"(= ({expr_id}) \"{self._escape_for_metta(expr_content)}\")")
                    
                    # Reference the atom in the loop invariant
                    metta_components.append(f"(LoopInvariant {location} {expr_id})")
                    
                    # Add property annotations using pattern mapper
                    self._add_property_annotations(metta_components, expr_id, expr_content, component)
                
                elif component_type == "assertion":
                    location = component.get("location", "unknown_location")
                    expr_id = f"assert_{location}_{len(metta_components)}"
                    expr_content = component.get("expression", "")
                    
                    # Create atoms for the expression
                    metta_components.append(f"(: {expr_id} Expression)")
                    metta_components.append(f"(= ({expr_id}) \"{self._escape_for_metta(expr_content)}\")")
                    
                    # Reference the atom in the assertion
                    metta_components.append(f"(Assertion {location} {expr_id})")
                    
                    # Add property annotations using pattern mapper
                    self._add_property_annotations(metta_components, expr_id, expr_content, component)
            
            # Add verification strategy as atoms with descriptive names
            strategy = proof_json.get("verification_strategy", {})
            approach = strategy.get("approach", "No approach specified")
            strategy_id = f"strategy_{len(metta_components)}"
            
            metta_components.append(f"(: {strategy_id} Strategy)")
            metta_components.append(f"(= ({strategy_id}) \"{self._escape_for_metta(approach)}\")")
            metta_components.append(f"(VerificationStrategy {strategy_id})")
            
            # Add property annotations for strategy using pattern mapper
            self._add_property_annotations(metta_components, strategy_id, approach, {"natural_language": approach})
            
            # Add lemmas as atoms with descriptive names
            for i, lemma in enumerate(strategy.get("key_lemmas", [])):
                lemma_id = f"lemma_{len(metta_components)}_{i}"
                metta_components.append(f"(: {lemma_id} Lemma)")
                metta_components.append(f"(= ({lemma_id}) \"{self._escape_for_metta(lemma)}\")")
                metta_components.append(f"(KeyLemma {lemma_id})")
                
                # Add property annotations for lemma using pattern mapper
                self._add_property_annotations(metta_components, lemma_id, lemma, {"natural_language": lemma})
                
        except Exception as e:
            logging.error(f"Error converting JSON to MeTTa proof: {e}", exc_info=True)
            error_id = f"error_{len(metta_components)}"
            metta_components.append(f"(: {error_id} Error)")
            metta_components.append(f"(= ({error_id}) \"Error converting to MeTTa: {str(e)}\")")
            metta_components.append(f"(ProofError {error_id})")
        
        return metta_components
    
    def _escape_for_metta(self, expr: str) -> str:
        """Escape a string for inclusion in MeTTa."""
        return expr.replace('"', '\\"').replace('\\', '\\\\')

    def verify_adaptation(self, original_func: str, adapted_func: str, 
                        essential_properties: List[str]) -> Dict[str, Any]:
        """
        Verify that an adaptation preserves essential properties.
        Uses pattern mapper for property identification and OpenAI for verification.
        """
        if not self.api_key or not self.openai_client:
            return {"success": False, "error": "OpenAI API key not provided. Cannot verify adaptation."}
            
        logging.info("Verifying adaptation preserves essential properties")
        
        # Analyze both functions with OpenAI
        prompt = f"""
        I need to verify that an adapted function preserves essential properties from the original function.
        
        ORIGINAL FUNCTION:
        ```python
        {original_func}
        ```
        
        ADAPTED FUNCTION:
        ```python
        {adapted_func}
        ```
        
        ESSENTIAL PROPERTIES TO PRESERVE:
        {', '.join(essential_properties)}
        
        Please analyze both functions and determine if the adaptation preserves the essential properties.
        For each property, indicate whether it is PRESERVED or VIOLATED in the adapted function.
        Provide a brief explanation for each property.
        
        Return your analysis in JSON format:
        {{
            "preserved_properties": [
                {{
                    "property": "property name",
                    "explanation": "why it's preserved"
                }}
            ],
            "violated_properties": [
                {{
                    "property": "property name",
                    "explanation": "why it's violated"
                }}
            ]
        }}
        
        ONLY RETURN THE JSON OBJECT, NO OTHER TEXT.
        """
        
        # Call OpenAI API
        llm_response = self._call_openai_api(prompt)
        
        # Extract JSON from response
        # Look for JSON content
        json_match = re.search(r'(\{[\s\S]*\})', llm_response)
        if json_match:
            json_content = json_match.group(1)
        else:
            json_content = llm_response
            
        try:
            result = json.loads(json_content)
            
            # Add MeTTa atoms for verification results if monitor is available
            if hasattr(self, 'monitor') and self.monitor:
                # Add preserved properties
                for prop_info in result.get("preserved_properties", []):
                    prop = prop_info.get("property")
                    property_type = self.pattern_mapper.map_requirement_to_property(prop)
                    self.monitor.add_atom(f"(adaptation-preserves-property {property_type})")
                
                # Add violated properties
                for prop_info in result.get("violated_properties", []):
                    prop = prop_info.get("property")
                    property_type = self.pattern_mapper.map_requirement_to_property(prop)
                    self.monitor.add_atom(f"(adaptation-violates-property {property_type})")
            
            # Check if all essential properties are preserved
            preserved = [p.get("property") for p in result.get("preserved_properties", [])]
            violated = [p.get("property") for p in result.get("violated_properties", [])]
            
            return {
                "success": all(prop in preserved for prop in essential_properties),
                "preserved_properties": preserved,
                "violated_properties": violated,
                "details": result
            }
            
        except Exception as e:
            logging.error(f"Error parsing verification result: {e}")
            return {
                "success": False,
                "error": f"Failed to verify adaptation: {str(e)}",
                "raw_response": llm_response
            }
    
    def _add_property_to_metta(self, func_name: str, property: str) -> List[str]:
        """
        Add a property to MeTTa space using proper descriptive pattern atoms.
        Returns the list of added atoms.
        """
        if not hasattr(self, 'monitor') or not self.monitor:
            return []
            
        added_atoms = []
        
        # Ensure type definitions exist
        type_atoms = [
            "(: Type Type)",
            "(: Property Type)",
            f"(: {func_name} Function)",
            "(: Function Type)"
        ]
        
        for atom in type_atoms:
            self.monitor.add_atom(atom)
            added_atoms.append(atom)
        
        # Convert property to MeTTa atom
        property_atom = self.pattern_mapper.map_requirement_to_property(property)
        
        # Ensure the property type is defined
        property_type_atom = f"(: {property_atom} Property)"
        self.monitor.add_atom(property_type_atom)
        added_atoms.append(property_type_atom)
        
        # Add property relationship to MeTTa space
        property_atom_expr = f"(function-has-property {func_name} {property_atom})"
        self.monitor.add_atom(property_atom_expr)
        added_atoms.append(property_atom_expr)
        
        # Add property name definitions if it's a non-standard property
        if property_atom.startswith("property-"):
            property_name = property_atom.strip("()").split("-", 1)[1] if "(" in property_atom else property_atom.split("-", 1)[1]
            type_def = f"(: {property_name} Property)"
            value_def = f"(= ({property_name}) \"{property}\")"
            
            self.monitor.add_atom(type_def)
            self.monitor.add_atom(value_def)
            
            added_atoms.extend([type_def, value_def])
            
        # Ensure the property type has a function-has-property relationship defined
        relationship_atom = "(: function-has-property (--> Function Property Bool))"
        self.monitor.add_atom(relationship_atom)
        added_atoms.append(relationship_atom)
        
        return added_atoms
    
    def analyze_function_for_proof(self, function_code: str, function_name: str = None,
                                context: str = None, max_attempts: int = 3) -> Dict[str, Any]:
        """
        Analyze function and generate proof using OpenAI.
        This is a simplified version that delegates to the OpenAI API directly.
        """
        if not self.api_key or not self.openai_client:
            return {"success": False, "error": "OpenAI API key not provided. Cannot analyze function."}
            
        logging.info(f"Analyzing function{' ' + function_name if function_name else ''} for proof generation")
        
        # Create a prompt for OpenAI
        context_info = f"Domain context: {context}\n\n" if context else ""
        
        prompt = f"""
        {context_info}
        Please generate a formal proof for this Python function:
        
        ```python
        {function_code}
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
        
        # Track attempts
        for attempt in range(max_attempts):
            # Call OpenAI API
            llm_response = self._call_openai_api(prompt)
            
            # Extract JSON from response
            json_match = re.search(r'(\{[\s\S]*\})', llm_response)
            if json_match:
                json_content = json_match.group(1)
            else:
                json_content = llm_response
                
            try:
                # Parse the JSON response
                proof_json = json.loads(json_content)
                
                # Validate the structure
                if not isinstance(proof_json, dict) or "proof_components" not in proof_json:
                    if attempt < max_attempts - 1:
                        # Try again with more specific instructions
                        prompt += "\n\nYour previous response did not have the correct JSON structure. Please ensure your response contains a 'proof_components' array and a 'verification_strategy' object."
                        continue
                    else:
                        return {
                            "success": False,
                            "error": "Failed to generate valid proof JSON after multiple attempts",
                            "raw_response": llm_response
                        }
                
                # Convert JSON to MeTTa proof
                metta_proof = self._json_to_metta_proof(proof_json)
                
                # Add to MeTTa space if monitor is available
                if hasattr(self, 'monitor') and self.monitor:
                    for component in metta_proof:
                        self.monitor.add_atom(component)
                    
                    # Add function verification status
                    func_name = function_name or "unnamed_function"
                    self.monitor.add_atom(f"(verified-function {func_name})")
                
                return {
                    "success": True,
                    "proof": metta_proof,
                    "function": function_code,
                    "function_name": function_name,
                    "json_ir": proof_json,
                    "attempts": attempt + 1
                }
                
            except Exception as e:
                logging.error(f"Failed to process proof on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    # Try again with more specific error feedback
                    prompt += f"\n\nYour previous response had an error: {str(e)}. Please ensure you return a valid JSON object."
                    continue
        
        # If all attempts failed
        return {
            "success": False,
            "function": function_code,
            "function_name": function_name,
            "attempts": max_attempts,
            "error": "Maximum proof generation attempts reached",
            "raw_response": llm_response
        }