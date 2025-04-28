class MeTTaPropertyVerifier:
    """
    Verifies that alternative implementations preserve required properties using MeTTa.
    This implementation specifically leverages the existing proof ontology rules.
    """
    
    def __init__(self, monitor):
        """
        Initialize the property verifier.
        
        Args:
            monitor: MeTTa monitor instance
        """
        self.monitor = monitor
    
    def verify_property_preservation(self, original_function: str, 
                                   alternative_function: str,
                                   property_atoms: list) -> dict:
        """
        Verify that the alternative implementation preserves required properties
        using the existing MeTTa ontology rules.
        
        Args:
            original_function: Original function code
            alternative_function: Alternative function code  
            property_atoms: List of MeTTa property atoms to verify
            
        Returns:
            Verification result dictionary
        """
        # Create unique identifiers for the functions
        import hashlib
        orig_hash = hashlib.md5(original_function.encode()).hexdigest()[:8]
        alt_hash = hashlib.md5(alternative_function.encode()).hexdigest()[:8]
        
        orig_func_id = f"orig_{orig_hash}"
        alt_func_id = f"alt_{alt_hash}"
        
        # Add functions to MeTTa space
        self._add_function_to_metta(orig_func_id, original_function)
        self._add_function_to_metta(alt_func_id, alternative_function)
        
        # Add property atoms for the original function
        for prop in property_atoms:
            # Use the existing pattern-property mappings in the ontology
            self.monitor.add_atom(f"(function-has-property {orig_func_id} {prop})")
        
        # Use the valid-adaptation predicate from the ontology to verify preservation
        query = f"""
        (match &self 
           (valid-adaptation {orig_func_id} {alt_func_id} {self._format_property_list(property_atoms)})
           True)
        """
        
        adaptation_results = self.monitor.query(query)
        all_preserved = len(adaptation_results) > 0
        
        # If the high-level check failed, check individual properties
        if not all_preserved:
            # Get preserved and violated properties using the ontology predicates
            preserved_query = f"""
            (match &self 
               (preserved-properties {orig_func_id} {alt_func_id} {self._format_property_list(property_atoms)})
               $preserved)
            """
            
            preserved_results = self.monitor.query(preserved_query)
            preserved_properties = preserved_results[0] if preserved_results else []
            
            violated_query = f"""
            (match &self 
               (violated-properties {orig_func_id} {alt_func_id} {self._format_property_list(property_atoms)})
               $violated)
            """
            
            violated_results = self.monitor.query(violated_query)
            violated_properties = violated_results[0] if violated_results else []
            
            # Format the results
            preservation_results = {
                prop: prop in preserved_properties for prop in property_atoms
            }
            
            return {
                "all_preserved": False,
                "property_results": preservation_results,
                "preserved_properties": preserved_properties,
                "violated_properties": violated_properties,
                "preserved_count": len(preserved_properties),
                "violated_count": len(violated_properties)
            }
        
        # All properties were preserved
        return {
            "all_preserved": True,
            "property_results": {prop: True for prop in property_atoms},
            "preserved_properties": property_atoms,
            "violated_properties": [],
            "preserved_count": len(property_atoms),
            "violated_count": 0
        }
    
    def _format_property_list(self, properties: list) -> str:
        """
        Format a list of properties for use in MeTTa queries.
        
        Args:
            properties: List of property atoms
            
        Returns:
            Formatted property list string
        """
        if not properties:
            return "()"
        
        return f"({' '.join(properties)})"
    
    def _add_function_to_metta(self, func_id: str, function_code: str) -> None:
        """
        Add a function to the MeTTa space.
        
        Args:
            func_id: Function identifier
            function_code: Function source code
        """
        # Add function type definition if not already present
        self.monitor.add_atom("(: Function Type)")
        
        # Add function declaration
        self.monitor.add_atom(f"(: {func_id} Function)")
        
        # Add function code as a string value
        safe_code = function_code.replace('"', '\\"').replace('\n', '\\n')
        self.monitor.add_atom(f"(= ({func_id}) \"{safe_code}\")")
    
    def verify_with_execution_traces(self, original_function: str, alternative_function: str,
                                   inputs: list, expected_outputs: list) -> dict:
        """
        Verify alternative implementation using execution traces,
        leveraging the execution evidence rules in the ontology.
        
        Args:
            original_function: Original function code
            alternative_function: Alternative function code
            inputs: List of test inputs
            expected_outputs: List of expected outputs
            
        Returns:
            Verification result dictionary
        """
        import re
        
        # Extract function name
        func_name_match = re.search(r'def\s+([a-zA-Z0-9_]+)', original_function)
        if not func_name_match:
            return {"success": False, "error": "Could not extract function name"}
        
        func_name = func_name_match.group(1)
        
        # Create execution context
        exec_context = {}
        
        # Execute the alternative function
        try:
            exec(alternative_function, exec_context)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to execute alternative function: {str(e)}"
            }
        
        # Run tests
        test_results = []
        all_passed = True
        
        for i, input_value in enumerate(inputs):
            expected = expected_outputs[i] if i < len(expected_outputs) else None
            
            try:
                # Handle different input types (single value, list, dict)
                if isinstance(input_value, list):
                    actual = exec_context[func_name](*input_value)
                elif isinstance(input_value, dict):
                    actual = exec_context[func_name](**input_value)
                else:
                    actual = exec_context[func_name](input_value)
                
                # Check if the actual output matches the expected output
                if expected is not None:
                    passed = actual == expected
                    all_passed = all_passed and passed
                else:
                    # If no expected output provided, just record the actual output
                    passed = None
                
                test_results.append({
                    "input": input_value,
                    "expected": expected,
                    "actual": actual,
                    "passed": passed
                })
                
            except Exception as e:
                test_results.append({
                    "input": input_value,
                    "expected": expected,
                    "error": str(e),
                    "passed": False
                })
                all_passed = False
        
        # Record the execution traces in MeTTa
        self._add_execution_traces_to_metta(func_name, test_results)
        
        # Use the MeTTa ontology to check if execution evidence supports properties
        supporting_properties = self._get_properties_supported_by_execution(func_name)
        
        return {
            "success": True,
            "all_passed": all_passed,
            "test_results": test_results,
            "pass_count": sum(1 for r in test_results if r.get("passed") == True),
            "fail_count": sum(1 for r in test_results if r.get("passed") == False),
            "supported_properties": supporting_properties
        }
    
    def _add_execution_traces_to_metta(self, func_name: str, test_results: list) -> None:
        """
        Add execution traces to MeTTa for verification.
        
        Args:
            func_name: Function name
            test_results: Test execution results
        """
        for i, result in enumerate(test_results):
            # Convert input and output to MeTTa-friendly strings
            input_str = self._convert_to_metta_string(result.get("input"))
            
            if "error" in result:
                # Record execution error
                self.monitor.add_atom(f"(execution-error {func_name} {i} \"{result['error']}\")")
            else:
                # Record successful execution using the ontology's execution-record predicate
                output_str = self._convert_to_metta_string(result.get("actual"))
                self.monitor.add_atom(f"(execution-record {i} {input_str} {output_str})")
                
                # Add execution state information that can be used by the ontology rules
                if isinstance(result.get("input"), list) and len(result.get("input")) > 0:
                    # For array inputs, add information about ordering
                    if self._is_ordered_array(result.get("input")[0]):
                        self.monitor.add_atom(f"(ordered-collection {input_str})")
                
                # Add information about the result for error handling checks
                actual = result.get("actual")
                if actual == -1:
                    self.monitor.add_atom(f"(= {output_str} negative-one-value)")
                
                # Record test result
                if result.get("passed") is not None:
                    passed_str = "True" if result.get("passed") else "False"
                    self.monitor.add_atom(f"(test-result {func_name} {i} {passed_str})")
    
    def _is_ordered_array(self, arr):
        """Check if an array is ordered."""
        if not isinstance(arr, (list, tuple)) or len(arr) <= 1:
            return True
            
        return all(arr[i] <= arr[i+1] for i in range(len(arr)-1)) or \
               all(arr[i] >= arr[i+1] for i in range(len(arr)-1))
    
    def _get_properties_supported_by_execution(self, func_name: str) -> list:
        """
        Use the ontology to determine which properties are supported by execution evidence.
        
        Args:
            func_name: Function name
            
        Returns:
            List of supported properties
        """
        # Query for properties supported by execution evidence
        query = f"""
        (match &self 
           (execution-evidence-supports {func_name} $property)
           $property)
        """
        
        results = self.monitor.query(query)
        return results
    
    def _convert_to_metta_string(self, value) -> str:
        """
        Convert a Python value to a MeTTa-friendly string representation.
        
        Args:
            value: Python value to convert
            
        Returns:
            MeTTa-friendly string representation
        """
        if value is None:
            return "None"
            
        if isinstance(value, (int, float, bool)):
            return str(value)
            
        if isinstance(value, str):
            # Escape quotes for MeTTa
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
            
        if isinstance(value, (list, tuple)):
            # Format as a MeTTa list
            items = [self._convert_to_metta_string(item) for item in value]
            return f"(List {' '.join(items)})"
            
        if isinstance(value, dict):
            # Format as a MeTTa dict structure
            items = []
            for k, v in value.items():
                key_str = self._convert_to_metta_string(k)
                val_str = self._convert_to_metta_string(v)
                items.append(f"(KeyValue {key_str} {val_str})")
            return f"(Dict {' '.join(items)})"
            
        # For other types, convert to string
        return f'"{str(value)}"'
    
    def extract_properties_from_components(self, proof_components: list) -> list:
        """
        Extract property atoms from proof components,
        using the pattern-property mappings in the ontology.
        
        Args:
            proof_components: List of proof components
            
        Returns:
            List of property atoms
        """
        properties = []
        
        # Map components to patterns and then to properties
        for component in proof_components:
            comp_type = component.get("type")
            expression = component.get("expression", "")
            explanation = component.get("explanation", "")
            
            # Combined text for pattern detection
            combined_text = (expression + " " + explanation).lower()
            
            # Attempt to identify patterns from the text
            identified_patterns = self._identify_patterns_from_text(combined_text)
            
            # For each identified pattern, query the ontology for the corresponding property
            for pattern in identified_patterns:
                pattern_property_query = f"""
                (match &self 
                   (pattern-property {pattern} $property)
                   $property)
                """
                
                property_results = self.monitor.query(pattern_property_query)
                properties.extend(property_results)
        
        # Remove duplicates and return
        return list(set(properties))
    
    def _identify_patterns_from_text(self, text: str) -> list:
        """
        Identify patterns from text based on keywords.
        This function maps to the atomic patterns defined in the ontology.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of pattern atoms
        """
        patterns = []
        
        # Bounds checking patterns
        if "index" in text and "length" in text:
            if "<=" in text:
                patterns.append("index-less-equal-length")
            elif "<" in text:
                patterns.append("index-less-than-length")
                
        if "index" in text and "0" in text and ">=" in text:
            patterns.append("index-greater-equal-zero")
            
        if "bound" in text or ("0" in text and "<=" in text and "index" in text and "length" in text):
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
        if "null" in text and "!=" in text:
            patterns.append("value-not-null")
            
        if "if" in text and ("null" in text or "none" in text):
            patterns.append("checks-for-null")
            
        if "empty" in text:
            patterns.append("handles-empty-collection")
        
        # Termination patterns
        if "decrease" in text or "decrement" in text:
            patterns.append("decreasing-loop-variable")
            
        if "increase" in text or "increment" in text:
            patterns.append("increasing-towards-bound")
            
        if "progress" in text and "invariant" in text:
            patterns.append("loop-invariant-progress")
            
        if "iteration" in text and ("count" in text or "finite" in text):
            patterns.append("finite-iteration-count")
        
        # Error handling patterns
        if "-1" in text or "not found" in text:
            patterns.append("checks-for-not-found")
            
        if "valid" in text and "input" in text:
            patterns.append("validates-input")
            
        if "edge" in text and "case" in text:
            patterns.append("handles-edge-cases")
            
        if "error" in text and "code" in text:
            patterns.append("error-code-return")
        
        return patterns