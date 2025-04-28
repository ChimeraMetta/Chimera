class ProofGuidedImplementationGenerator:
    """
    Generates alternative implementations of functions guided by proof components.
    Uses formal proof components to constrain LLM-based code generation to produce
    verified alternatives that satisfy the same properties, leveraging MeTTa ontology rules.
    """
    
    def __init__(self, analyzer, model_name="gpt-4o-mini", api_key=None):
        """
        Initialize the implementation generator.
        
        Args:
            analyzer: Reference to the ImmuneSystemProofAnalyzer
            model_name: Name of the LLM model to use
            api_key: OpenAI API key
        """
        self.analyzer = analyzer
        self.model_name = model_name
        self.api_key = api_key
        self.openai_client = self.analyzer.openai_client
        # Use the MeTTa monitor from the analyzer
        self.monitor = self.analyzer.monitor
        
    def generate_alternative_implementation(self, function_code: str, function_name: str, 
                                           proof_components: list = None, 
                                           strategy: str = "semantically_equivalent",
                                           constraints: dict = None) -> dict:
        """
        Generate an alternative implementation of a function guided by proof components.
        
        Args:
            function_code: Original function source code
            function_name: Name of the function
            proof_components: Proof components (if None, will be generated)
            strategy: Generation strategy (semantically_equivalent, optimized, simplified)
            constraints: Additional constraints for generation (e.g., language features to use/avoid)
            
        Returns:
            Dictionary with generated implementation and verification results
        """
        # If proof components not provided, generate them
        if not proof_components:
            proof_result = self.analyzer.analyze_function_for_proof(function_code, function_name)
            if not proof_result["success"]:
                return {
                    "success": False,
                    "error": "Failed to generate proof components for the function"
                }
            proof_components = proof_result.get("proof", [])
            proof_json = proof_result.get("json_ir", {})
        else:
            # Extract JSON IR from proof components if available
            proof_json = self._extract_json_ir_from_components(proof_components)
        
        # Format the proof components for prompting
        formatted_proof = self._format_proof_for_prompt(proof_components, proof_json)
        
        # Extract property atoms from proof components
        # Use the ontology to map proof components to properties
        property_atoms = self._extract_property_atoms_from_components(proof_components)
        
        # Register function and properties in MeTTa space
        self._register_original_function(function_code, function_name, property_atoms)
        
        # Generate prompt based on strategy
        prompt = self._create_implementation_prompt(
            function_code, 
            function_name, 
            formatted_proof, 
            strategy, 
            constraints
        )
        
        # Generate alternative implementation
        alternative_code = self._generate_implementation(prompt)
        
        # Extract function name and function body
        extracted_function = self._extract_function_from_response(alternative_code)
        if not extracted_function:
            return {
                "success": False,
                "error": "Failed to extract function from generated response",
                "generated_text": alternative_code
            }
        
        # Register the alternative function in MeTTa space
        alt_func_id = self._register_alternative_function(extracted_function, function_name)
        
        # Verify the alternative implementation using MeTTa ontology
        verification_result = self._verify_with_metta_ontology(
            function_name,
            alt_func_id,
            property_atoms
        )
        
        # If MeTTa ontology verification failed, use the analyzer's verification as fallback
        if not verification_result["success"]:
            analyzer_verification = self._verify_with_analyzer(
                extracted_function, 
                function_name,
                proof_components
            )
            
            verification_result.update(analyzer_verification)
        
        return {
            "success": verification_result["success"],
            "original_function": function_code,
            "alternative_function": extracted_function,
            "verification_result": verification_result,
            "strategy": strategy,
            "proof_components": proof_components,
            "property_atoms": property_atoms
        }
    
    def _extract_json_ir_from_components(self, proof_components: list) -> dict:
        """
        Extract JSON IR structure from proof components.
        
        Args:
            proof_components: List of proof components
            
        Returns:
            JSON IR structure
        """
        # Initialize the JSON IR structure
        json_ir = {
            "proof_components": [],
            "verification_strategy": {
                "approach": "",
                "key_lemmas": []
            }
        }
        
        # Convert proof components to JSON IR format
        for component in proof_components:
            json_component = {
                "type": component.get("type", "unknown"),
                "expression": component.get("expression", ""),
                "natural_language": component.get("explanation", ""),
                "location": component.get("location", "function")
            }
            json_ir["proof_components"].append(json_component)
        
        return json_ir
    
    def _format_proof_for_prompt(self, proof_components: list, proof_json: dict) -> str:
        """
        Format proof components for inclusion in the prompt.
        
        Args:
            proof_components: List of proof components
            proof_json: JSON IR structure
            
        Returns:
            Formatted proof string
        """
        # Extract components by type
        preconditions = []
        loop_invariants = []
        assertions = []
        postconditions = []
        
        for comp in proof_components:
            comp_type = comp.get("type")
            
            if comp_type == "precondition":
                preconditions.append(f"PRECONDITION: {comp.get('expression')}\n   Meaning: {comp.get('explanation')}")
            elif comp_type == "loop_invariant":
                loop_invariants.append(f"LOOP INVARIANT at {comp.get('location')}: {comp.get('expression')}\n   Meaning: {comp.get('explanation')}")
            elif comp_type == "assertion":
                assertions.append(f"ASSERTION at {comp.get('location')}: {comp.get('expression')}\n   Meaning: {comp.get('explanation')}")
            elif comp_type == "postcondition":
                postconditions.append(f"POSTCONDITION: {comp.get('expression')}\n   Meaning: {comp.get('explanation')}")
        
        # Get verification strategy
        strategy = proof_json.get("verification_strategy", {})
        approach = strategy.get("approach", "")
        lemmas = strategy.get("key_lemmas", [])
        
        # Format the complete proof
        formatted_proof = "FORMAL PROOF COMPONENTS:\n\n"
        
        if preconditions:
            formatted_proof += "PRECONDITIONS:\n" + "\n".join(preconditions) + "\n\n"
        
        if loop_invariants:
            formatted_proof += "LOOP INVARIANTS:\n" + "\n".join(loop_invariants) + "\n\n"
        
        if assertions:
            formatted_proof += "ASSERTIONS:\n" + "\n".join(assertions) + "\n\n"
        
        if postconditions:
            formatted_proof += "POSTCONDITIONS:\n" + "\n".join(postconditions) + "\n\n"
        
        if approach:
            formatted_proof += f"VERIFICATION APPROACH:\n{approach}\n\n"
        
        if lemmas:
            formatted_proof += "KEY LEMMAS:\n" + "\n".join([f"- {lemma}" for lemma in lemmas])
        
        return formatted_proof
    
    def _extract_property_atoms_from_components(self, proof_components: list) -> list:
        """
        Extract property atoms from proof components using the MeTTa ontology.
        
        Args:
            proof_components: List of proof components
            
        Returns:
            List of property atoms
        """
        # Use the property verifier if available from the analyzer
        if hasattr(self.analyzer, "property_verifier"):
            return self.analyzer.property_verifier.extract_properties_from_components(proof_components)
        
        # Fallback implementation if property verifier is not available
        properties = set()
        
        # First, add component expressions to MeTTa
        for i, comp in enumerate(proof_components):
            comp_type = comp.get("type", "unknown")
            expr = comp.get("expression", "")
            loc = comp.get("location", "function")
            
            # Skip invalid components
            if not expr:
                continue
                
            # Create a unique expression ID
            expr_id = f"expr_{i}"
            
            # Register the expression in MeTTa
            self.monitor.add_atom(f"(: {expr_id} Expression)")
            safe_expr = expr.replace('"', '\\"')
            self.monitor.add_atom(f"(= {expr_id} \"{safe_expr}\")")
            
            # Add the appropriate type of component
            if comp_type == "precondition":
                self.monitor.add_atom(f"(Precondition {expr_id})")
            elif comp_type == "loop_invariant":
                self.monitor.add_atom(f"(LoopInvariant {loc} {expr_id})")
            elif comp_type == "assertion":
                self.monitor.add_atom(f"(Assertion {loc} {expr_id})")
            elif comp_type == "postcondition":
                self.monitor.add_atom(f"(Postcondition {expr_id})")
            
            # Query the MeTTa ontology for patterns in this expression
            combined_text = (expr + " " + comp.get("explanation", "")).lower()
            patterns = self._identify_patterns_from_text(combined_text)
            
            # For each pattern, query the property
            for pattern in patterns:
                query = f"""
                (match &self 
                   (pattern-property {pattern} $property)
                   $property)
                """
                
                results = self.monitor.query(query)
                for property_atom in results:
                    properties.add(property_atom)
                    
                    # Add the property to the expression in MeTTa
                    self.monitor.add_atom(f"(Expression-Property {expr_id} {property_atom})")
        
        return list(properties)
    
    def _identify_patterns_from_text(self, text: str) -> list:
        """
        Identify patterns from text based on keywords.
        Maps to the atomic patterns defined in the ontology.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of pattern atoms
        """
        patterns = []
        
        # These mappings should match the pattern atoms in your ontology
        
        # Bounds checking patterns
        if any(term in text for term in ["bound", "index", "range", "within"]):
            if "<=" in text:
                patterns.append("index-less-equal-length")
            elif "<" in text:
                patterns.append("index-less-than-length")
            if ">=" in text and "0" in text:
                patterns.append("index-greater-equal-zero")
            patterns.append("index-within-bounds")
        
        # Ordering patterns
        if "sort" in text:
            patterns.append("array-is-sorted")
        if "ascend" in text:
            patterns.append("ascending-order")
        if "descend" in text:
            patterns.append("descending-order")
        if "preserves" in text and "order" in text:
            patterns.append("preserves-element-order")
        
        # Null checking patterns
        if "null" in text or "none" in text:
            patterns.append("value-not-null")
            patterns.append("checks-for-null")
        if "empty" in text:
            patterns.append("handles-empty-collection")
        
        # Termination patterns
        if "decreas" in text or "decrements" in text:
            patterns.append("decreasing-loop-variable")
        if "increas" in text or "increments" in text:
            patterns.append("increasing-towards-bound")
        if "invariant" in text:
            patterns.append("loop-invariant-progress")
        if "finite" in text or "count" in text:
            patterns.append("finite-iteration-count")
        
        # Error handling patterns
        if "-1" in text or "not found" in text:
            patterns.append("checks-for-not-found")
        if "valid" in text and "input" in text:
            patterns.append("validates-input")
        if "edge" in text or "case" in text:
            patterns.append("handles-edge-cases")
        if "error" in text or "exception" in text:
            patterns.append("error-code-return")
        
        return patterns
    
    def _register_original_function(self, function_code: str, function_name: str, property_atoms: list) -> str:
        """
        Register the original function in MeTTa space with its properties.
        
        Args:
            function_code: Function source code
            function_name: Function name
            property_atoms: List of property atoms
            
        Returns:
            Function ID in MeTTa space
        """
        import hashlib
        func_hash = hashlib.md5(function_code.encode()).hexdigest()[:8]
        func_id = f"func_{func_hash}"
        
        # Add function to MeTTa space
        self.monitor.add_atom("(: Function Type)")
        self.monitor.add_atom(f"(: {func_id} Function)")
        
        safe_code = function_code.replace('"', '\\"').replace('\n', '\\n')
        self.monitor.add_atom(f"(= ({func_id}) \"{safe_code}\")")
        
        # Register function properties
        for prop in property_atoms:
            self.monitor.add_atom(f"(function-has-property {func_id} {prop})")
        
        return func_id
    
    def _register_alternative_function(self, function_code: str, original_name: str) -> str:
        """
        Register an alternative function in MeTTa space.
        
        Args:
            function_code: Function source code
            original_name: Original function name
            
        Returns:
            Alternative function ID in MeTTa space
        """
        import hashlib
        func_hash = hashlib.md5(function_code.encode()).hexdigest()[:8]
        alt_func_id = f"alt_{func_hash}"
        
        # Add function to MeTTa space
        self.monitor.add_atom("(: Function Type)")
        self.monitor.add_atom(f"(: {alt_func_id} Function)")
        
        safe_code = function_code.replace('"', '\\"').replace('\n', '\\n')
        self.monitor.add_atom(f"(= ({alt_func_id}) \"{safe_code}\")")
        
        # Register alternative function marker
        self.monitor.add_atom(f"(alternative-function {alt_func_id} {original_name})")
        
        return alt_func_id
    
    def _verify_with_metta_ontology(self, original_name: str, alt_func_id: str, property_atoms: list) -> dict:
        """
        Verify the alternative implementation using MeTTa ontology rules.
        
        Args:
            original_name: Original function name
            alt_func_id: Alternative function ID
            property_atoms: List of property atoms
            
        Returns:
            Verification result dictionary
        """
        # Format the property list for the query
        prop_list = f"({' '.join(property_atoms)})" if property_atoms else "()"
        
        # Use the suitable-donor-candidate predicate to check if all properties are preserved
        query = f"""
        (match &self 
           (suitable-donor-candidate {alt_func_id} {prop_list})
           True)
        """
        
        results = self.monitor.query(query)
        all_preserved = len(results) > 0
        
        # If verification succeeded, return success
        if all_preserved:
            return {
                "success": True,
                "properties_preserved": True,
                "verification_method": "metta_ontology"
            }
        
        # If verification failed, check which properties are preserved
        preserved_properties = []
        violated_properties = []
        
        for prop in property_atoms:
            # Use satisfies-property predicate to check individual properties
            property_query = f"""
            (match &self 
               (satisfies-property {alt_func_id} {prop})
               True)
            """
            
            property_results = self.monitor.query(property_query)
            if property_results:
                preserved_properties.append(prop)
            else:
                violated_properties.append(prop)
        
        return {
            "success": len(preserved_properties) == len(property_atoms),
            "properties_preserved": len(preserved_properties) == len(property_atoms),
            "preserved_properties": preserved_properties,
            "violated_properties": violated_properties,
            "verification_method": "metta_ontology"
        }
    
    def _verify_with_analyzer(self, alternative_function: str, function_name: str, proof_components: list) -> dict:
        """
        Verify the alternative implementation using the analyzer as fallback.
        
        Args:
            alternative_function: Alternative function code
            function_name: Original function name
            proof_components: Original proof components
            
        Returns:
            Verification result dictionary
        """
        # Use the analyzer to generate proof for the alternative
        alt_proof_result = self.analyzer.analyze_function_for_proof(
            alternative_function, 
            function_name
        )
        
        if not alt_proof_result["success"]:
            return {
                "success": False,
                "error": "Failed to generate proof for alternative implementation",
                "verification_method": "analyzer_fallback"
            }
        
        # Extract property atoms from both sets of proof components
        if hasattr(self.analyzer, "property_verifier"):
            original_properties = set(self.analyzer.property_verifier.extract_properties_from_components(proof_components))
            alt_properties = set(self.analyzer.property_verifier.extract_properties_from_components(alt_proof_result["proof"]))
            
            # Check if all original properties are preserved
            preserved_properties = list(original_properties.intersection(alt_properties))
            violated_properties = list(original_properties - alt_properties)
            
            return {
                "success": len(violated_properties) == 0,
                "properties_preserved": len(violated_properties) == 0,
                "preserved_properties": preserved_properties,
                "violated_properties": violated_properties,
                "verification_method": "analyzer_property_verification"
            }
        
        # If property verifier not available, use a simple component type comparison
        original_types = {comp["type"] for comp in proof_components}
        alt_types = {comp["type"] for comp in alt_proof_result.get("proof", [])}
        
        return {
            "success": original_types.issubset(alt_types),
            "properties_preserved": original_types.issubset(alt_types),
            "verification_method": "component_type_comparison"
        }
    
    def _create_implementation_prompt(self, function_code: str, function_name: str,
                                    formatted_proof: str, strategy: str,
                                    constraints: dict = None) -> str:
        """
        Create a prompt for generating an alternative implementation.
        
        Args:
            function_code: Original function source code
            function_name: Name of the function
            formatted_proof: Formatted proof components
            strategy: Generation strategy
            constraints: Additional constraints
            
        Returns:
            Prompt string
        """
        # Handle constraints
        constraints_text = ""
        if constraints:
            constraints_text = "IMPLEMENTATION CONSTRAINTS:\n"
            for key, value in constraints.items():
                constraints_text += f"- {key}: {value}\n"
            constraints_text += "\n"
        
        # Create strategy-specific instructions
        if strategy == "optimized":
            strategy_text = """OPTIMIZATION GOAL:
Generate an optimized version of the function that improves time or space complexity while satisfying the same proof components.
Focus on algorithmic improvements that maintain the same correctness guarantees.
"""
        elif strategy == "simplified":
            strategy_text = """SIMPLIFICATION GOAL:
Generate a simplified version of the function with improved readability and maintainability.
Reduce complexity and make the code more straightforward while preserving the same behavior.
"""
        else:  # semantically_equivalent
            strategy_text = """SEMANTIC EQUIVALENCE GOAL:
Generate an alternative implementation that is semantically equivalent to the original function.
The implementation should satisfy the same proof components but may use different approaches.
"""
        
        # Create the prompt
        prompt = f"""
I need you to generate an alternative implementation of this function that satisfies all the formal proof components:

ORIGINAL FUNCTION:
```python
{function_code}
```

{formatted_proof}

{strategy_text}

{constraints_text}
INSTRUCTIONS:
1. Generate a complete, working Python function named {function_name} 
2. Your implementation MUST satisfy all PRECONDITIONS and POSTCONDITIONS
3. Any loops MUST maintain their respective LOOP INVARIANTS
4. The key assertions MUST hold at their respective points in the code
5. Include brief comments explaining how your implementation satisfies the proof components

ONLY RETURN THE FUNCTION CODE WITH COMMENTS, NO EXPLANATIONS OUTSIDE THE CODE.
DO NOT include markdown formatting, explanation text, or any other content before or after the function.
"""
        
        return prompt
    
    def _generate_implementation(self, prompt: str) -> str:
        """
        Generate alternative implementation using the LLM.
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            Generated code as string
        """
        if not self.openai_client:
            return "Error: OpenAI client not initialized"
        
        messages = [
            {"role": "system", "content": "You are an expert programmer specializing in algorithm design and formal verification."},
            {"role": "user", "content": prompt}
        ]
        
        return self.openai_client.get_completion_text(messages)
    
    def _extract_function_from_response(self, response: str) -> str:
        """
        Extract the function code from the LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Extracted function code
        """
        import re
        
        # Look for code blocks
        code_match = re.search(r'```(?:python)?\s*([\s\S]*?)\s*```', response)
        if code_match:
            return code_match.group(1).strip()
        
        # If no code blocks, look for function definition
        function_match = re.search(r'(def\s+[^:]+:[\s\S]*)', response)
        if function_match:
            return function_match.group(1).strip()
        
        # If not found, return the whole response (it might be just the function)
        if response.strip().startswith("def "):
            return response.strip()
        
        return None
    
    def batch_generate_alternatives(self, function_code: str, function_name: str, 
                                   count: int = 3, strategies: list = None) -> list:
        """
        Generate multiple alternative implementations with different strategies.
        
        Args:
            function_code: Original function source code
            function_name: Name of the function
            count: Number of alternatives to generate
            strategies: List of strategies to use (default: all available)
            
        Returns:
            List of alternative implementations with verification results
        """
        # Generate proof components once
        proof_result = self.analyzer.analyze_function_for_proof(function_code, function_name)
        if not proof_result["success"]:
            return [{
                "success": False,
                "error": "Failed to generate proof components for the function"
            }]
        
        proof_components = proof_result.get("proof", [])
        
        # Default strategies
        if not strategies:
            strategies = ["semantically_equivalent", "optimized", "simplified"]
        
        # Generate alternatives with different strategies
        alternatives = []
        
        for i in range(count):
            # Cycle through strategies
            strategy = strategies[i % len(strategies)]
            
            # Use different constraints for diversity
            constraints = self._generate_diverse_constraints(i)
            
            # Generate alternative
            alt_result = self.generate_alternative_implementation(
                function_code,
                function_name,
                proof_components,
                strategy,
                constraints
            )
            
            alternatives.append(alt_result)
        
        return alternatives
    
    def _generate_diverse_constraints(self, index: int) -> dict:
        """
        Generate diverse constraints for alternative implementations.
        
        Args:
            index: Index of the alternative
            
        Returns:
            Constraints dictionary
        """
        constraints_options = [
            {
                "style": "Functional programming style with minimal mutable state",
                "features": "Use list comprehensions, map/filter, and avoid explicit loops where possible"
            },
            {
                "style": "Imperative programming with explicit control flow",
                "features": "Use explicit loops and conditional statements for clarity"
            },
            {
                "style": "Optimized for readability and maintainability",
                "features": "Use descriptive variable names and add explanatory comments"
            },
            {
                "style": "Optimized for performance",
                "features": "Minimize memory allocations and function calls"
            },
            {
                "style": "Minimal implementation",
                "features": "Use the most concise implementation possible without sacrificing clarity"
            }
        ]
        
        return constraints_options[index % len(constraints_options)]