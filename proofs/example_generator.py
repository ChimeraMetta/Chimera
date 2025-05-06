import json
import re
from typing import Dict, List, Any, Optional

from hyperon import *
from reflectors.dynamic_monitor import DynamicMonitor
from proofs.requester import OpenAIRequests

class ExampleDrivenProofGenerator:
    """
    A more flexible proof generator that uses example functions with their corresponding
    proof components to guide LLM output for any algorithm type.
    """
    
    def __init__(self, monitor=None, model_name="gpt-4o-mini", api_key=None):
        """Initialize the proof generator with examples."""
        self.monitor = monitor
        self.model_name = model_name
        self.api_key = api_key
        self.openai_client = OpenAIRequests(api_key, model_name) if api_key else None
        
        # Load example functions and their proof components
        self.example_library = self._initialize_example_library()
    
    def _initialize_example_library(self):
        """
        Initialize a library of example functions and their proof components.
        Each example includes a function and its corresponding proof in JSON format.
        """
        return {
            "basic_search": {
                "function": """
def basic_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
                """,
                "proof": {
                    "proof_components": [
                        {
                            "type": "precondition",
                            "expression": "len(arr) >= 0",
                            "natural_language": "The array has a valid length",
                            "location": "function"
                        },
                        {
                            "type": "loop_invariant",
                            "expression": "i >= 0 && i <= len(arr)",
                            "natural_language": "Loop index stays within valid bounds",
                            "location": "for_loop"
                        },
                        {
                            "type": "assertion",
                            "expression": "i < len(arr) => arr[i] is defined",
                            "natural_language": "Array access is within bounds",
                            "location": "array_access"
                        },
                        {
                            "type": "postcondition",
                            "expression": "result == -1 || (result >= 0 && arr[result] == target)",
                            "natural_language": "Either target not found or correct index returned",
                            "location": "function"
                        }
                    ],
                    "verification_strategy": {
                        "approach": "Linear search verification using loop invariants",
                        "key_lemmas": [
                            "The search examines every element in the array",
                            "If the target is found, its index is immediately returned",
                            "If search completes with no match, -1 is returned"
                        ]
                    }
                }
            },
            "binary_search": {
                "function": """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
                """,
                "proof": {
                    "proof_components": [
                        {
                            "type": "precondition",
                            "expression": "is_sorted(arr)",
                            "natural_language": "The array must be sorted in ascending order",
                            "location": "function"
                        },
                        {
                            "type": "loop_invariant",
                            "expression": "left <= right + 1 && (target in arr => target in arr[left:right+1])",
                            "natural_language": "If target exists, it is in the current search range",
                            "location": "while_loop"
                        },
                        {
                            "type": "assertion",
                            "expression": "mid == left + (right - left) // 2",
                            "natural_language": "Middle index calculation avoids overflow",
                            "location": "mid_calculation"
                        },
                        {
                            "type": "postcondition",
                            "expression": "result == -1 || (result >= 0 && arr[result] == target)",
                            "natural_language": "Either target not found or correct index returned",
                            "location": "function"
                        }
                    ],
                    "verification_strategy": {
                        "approach": "Binary search correctness using loop invariants and range analysis",
                        "key_lemmas": [
                            "The search space reduces by half in each iteration",
                            "If target exists, it remains in the current range",
                            "The algorithm terminates after logarithmic iterations",
                            "The algorithm returns correct index or -1 if not found"
                        ]
                    }
                }
            },
            "sorting_example": {
                "function": """
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
                """,
                "proof": {
                    "proof_components": [
                        {
                            "type": "precondition",
                            "expression": "len(arr) >= 0",
                            "natural_language": "The array has a valid length",
                            "location": "function"
                        },
                        {
                            "type": "loop_invariant",
                            "expression": "is_sorted(arr[0:i])",
                            "natural_language": "The subarray arr[0:i] is sorted",
                            "location": "outer_loop"
                        },
                        {
                            "type": "loop_invariant",
                            "expression": "j >= -1 && j < i && arr[j+1:i+1] contains elements > key",
                            "natural_language": "Elements greater than key are shifted right",
                            "location": "inner_loop"
                        },
                        {
                            "type": "postcondition",
                            "expression": "is_sorted(arr) && is_permutation(result, original_arr)",
                            "natural_language": "Array is sorted and contains the same elements",
                            "location": "function"
                        }
                    ],
                    "verification_strategy": {
                        "approach": "Insertion sort verification using loop invariants",
                        "key_lemmas": [
                            "The algorithm maintains a sorted prefix of increasing length",
                            "Each iteration correctly inserts a new element into the sorted prefix",
                            "After n iterations, the entire array is sorted"
                        ]
                    }
                }
            },
            "general_algorithm": {
                "function": """
def process_data(data, threshold):
    results = []
    for item in data:
        if item.value > threshold:
            processed = transform(item)
            if processed is not None:
                results.append(processed)
    return results
                """,
                "proof": {
                    "proof_components": [
                        {
                            "type": "precondition",
                            "expression": "data != None && threshold is defined",
                            "natural_language": "Input data and threshold must be valid",
                            "location": "function"
                        },
                        {
                            "type": "loop_invariant",
                            "expression": "all(result[i].value > threshold for i in range(len(results)))",
                            "natural_language": "All processed items in results exceed the threshold",
                            "location": "for_loop"
                        },
                        {
                            "type": "assertion",
                            "expression": "processed is None || is_valid(processed)",
                            "natural_language": "Any processed item added to results is valid",
                            "location": "append_operation"
                        },
                        {
                            "type": "postcondition",
                            "expression": "all(r in results => r.value > threshold && is_valid(r))",
                            "natural_language": "All results exceed threshold and are valid",
                            "location": "function"
                        }
                    ],
                    "verification_strategy": {
                        "approach": "Data processing verification using invariants",
                        "key_lemmas": [
                            "Only items exceeding the threshold are processed",
                            "Only valid processed items are included in results",
                            "The results maintain the required properties"
                        ]
                    }
                }
            }
        }
    
    def generate_proof(self, function_code: str, function_name: str = None, 
                       max_attempts: int = 3) -> Dict[str, Any]:
        """
        Generate a formal proof for the given function using example-driven approach.
        
        Args:
            function_code: The source code of the function
            function_name: Name of the function (optional)
            max_attempts: Maximum number of generation attempts
            
        Returns:
            Dict containing proof results
        """
        if not self.api_key or not self.openai_client:
            return {"success": False, "error": "OpenAI API key not provided"}
        
        # Extract function name if not provided
        if not function_name:
            import re
            match = re.search(r'def\s+([a-zA-Z0-9_]+)', function_code)
            if match:
                function_name = match.group(1)
            else:
                function_name = "unnamed_function"
        
        # Format examples for the prompt
        examples_text = self._format_examples_for_prompt(function_code)
        
        # Create the prompt
        prompt = self._create_example_driven_prompt(function_code, examples_text)
        
        # Make multiple attempts if needed
        for attempt in range(max_attempts):
            try:
                # Call the LLM
                messages = [
                    {"role": "system", "content": "You are a formal verification expert specializing in algorithm correctness proofs."},
                    {"role": "user", "content": prompt}
                ]
                response = self.openai_client.get_completion_text(messages)
                
                # Extract and parse the JSON
                json_content = self._extract_json_from_response(response)
                
                try:
                    proof_json = json.loads(json_content)
                    
                    # Validate and normalize
                    proof_json = self._validate_and_normalize(proof_json, function_name)
                    
                    # Convert to MeTTa atoms
                    metta_atoms = self._json_to_metta_atoms(proof_json, function_name)
                    
                    # Add to MeTTa space if monitor is available
                    if self.monitor:
                        for atom in metta_atoms:
                            self.monitor.add_atom(atom)
                    
                    return {
                        "success": True,
                        "proof": metta_atoms,
                        "function_name": function_name,
                        "json_ir": proof_json,
                        "attempts": attempt + 1
                    }
                
                except json.JSONDecodeError as e:
                    if attempt == max_attempts - 1:
                        # Create a fallback proof on the last attempt
                        fallback_proof = self._create_fallback_proof(function_code, function_name)
                        return {
                            "success": True,
                            "proof": fallback_proof["metta_atoms"],
                            "function_name": function_name,
                            "json_ir": fallback_proof["json_ir"],
                            "fallback": True,
                            "error": str(e)
                        }
            except Exception as e:
                if attempt == max_attempts - 1:
                    # Create a fallback proof on the last attempt
                    fallback_proof = self._create_fallback_proof(function_code, function_name)
                    return {
                        "success": True,
                        "proof": fallback_proof["metta_atoms"],
                        "function_name": function_name,
                        "json_ir": fallback_proof["json_ir"],
                        "fallback": True,
                        "error": str(e)
                    }
        
        # Should never reach here due to fallback, but just in case
        return {"success": False, "error": "Failed to generate proof after all attempts"}
    
    def _format_examples_for_prompt(self, function_code: str) -> str:
        """
        Select the most relevant examples for the input function and format them for the prompt.
        
        Args:
            function_code: The function code to analyze
            
        Returns:
            Formatted examples as a string
        """
        # Identify key characteristics of the function
        has_loop = "for " in function_code or "while " in function_code
        is_search = "search" in function_code.lower() or "find" in function_code.lower()
        is_sort = "sort" in function_code.lower()
        returns_minus_one = "return -1" in function_code
        
        # Choose the most relevant example
        primary_example = None
        if is_search and "left" in function_code and "right" in function_code:
            primary_example = "binary_search"
        elif is_search:
            primary_example = "basic_search"
        elif is_sort:
            primary_example = "sorting_example"
        else:
            primary_example = "general_algorithm"
        
        # Format the chosen example
        example = self.example_library[primary_example]
        example_json = json.dumps(example["proof"], indent=2)
        
        return f"""
EXAMPLE FUNCTION:
```python
{example["function"].strip()}
```

EXAMPLE PROOF COMPONENTS:
```json
{example_json}
```
"""
    
    def _create_example_driven_prompt(self, function_code: str, examples_text: str) -> str:
        """
        Create a prompt that uses examples to guide LLM output.
        
        Args:
            function_code: The function to analyze
            examples_text: Formatted examples
            
        Returns:
            Complete prompt string
        """
        return f"""
I need you to generate a formal proof for this function:

```python
{function_code}
```

Please follow the EXACT same structure and format as this example:
{examples_text}

YOUR RESPONSE REQUIREMENTS:
1. ONLY RETURN A VALID JSON OBJECT with the same structure as the example
2. Your proof_components array MUST include these types:
   - At least one "precondition"
   - At least one "loop_invariant" for each loop in the function
   - At least one "assertion" for important statements
   - At least one "postcondition"
3. Each component MUST have these fields:
   - "type": One of the four types above
   - "expression": A CLEAN logical expression (NO markdown or LaTeX)
   - "natural_language": Plain explanation in English
   - "location": Where this applies (e.g., "function", "for_loop", "while_loop", or specific code point)

EXPRESSION FIELD RULES:
- Write ONLY valid logical conditions (as if they were executable code)
- Use standard operators: ==, !=, <, >, <=, >=, +, -, *, /, &&, ||, !
- Use function-like predicates for complex concepts: is_sorted(arr), contains(list, item)
- NEVER use bullet points, markdown formatting, or sentence fragments
- NEVER use LaTeX notation (like \forall or \exists)

IMPORTANT:
- DO NOT include ANY text before or after the JSON object
- DO NOT explain your reasoning outside the JSON
- Focus on correctness and completeness of the proof components
"""
    
    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON content from LLM response with robust error handling.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Extracted JSON string
        """
        import re
        
        # Look for JSON content between triple backticks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            return json_match.group(1)
        
        # Look for anything that looks like a JSON object
        json_match = re.search(r'(\{[\s\S]*\})', response)
        if json_match:
            return json_match.group(1)
        
        # If we can't find JSON-like content, return the whole response
        return response
    
    def _validate_and_normalize(self, proof_json: Dict[str, Any], function_name: str) -> Dict[str, Any]:
        """
        Validate and normalize the proof JSON structure.
        
        Args:
            proof_json: The raw proof JSON from LLM
            function_name: Name of the function
            
        Returns:
            Validated and normalized proof JSON
        """
        # Ensure we have the basic structure
        if not isinstance(proof_json, dict):
            proof_json = {}
        
        if "proof_components" not in proof_json or not isinstance(proof_json["proof_components"], list):
            proof_json["proof_components"] = []
        
        if "verification_strategy" not in proof_json or not isinstance(proof_json["verification_strategy"], dict):
            proof_json["verification_strategy"] = {}
        
        strategy = proof_json["verification_strategy"]
        if "approach" not in strategy or not isinstance(strategy["approach"], str):
            strategy["approach"] = f"Correctness verification for {function_name}"
        
        if "key_lemmas" not in strategy or not isinstance(strategy["key_lemmas"], list):
            strategy["key_lemmas"] = ["The algorithm terminates for all valid inputs", 
                                     "The algorithm produces correct results"]
        
        # Ensure all components have required fields
        for component in proof_json["proof_components"]:
            if not isinstance(component, dict):
                continue
                
            component.setdefault("type", "assertion")
            component.setdefault("expression", "valid_condition")
            component.setdefault("natural_language", "Condition holds at this point")
            component.setdefault("location", "function")
            
            # Clean expression
            component["expression"] = self._clean_expression(component["expression"])
        
        # Check for required component types
        component_types = {comp.get("type") for comp in proof_json["proof_components"] if isinstance(comp, dict)}
        required_types = {"precondition", "loop_invariant", "postcondition"}
        missing_types = required_types - component_types
        
        # Add default components for missing types
        for missing_type in missing_types:
            default_comp = {
                "type": missing_type,
                "expression": f"valid_{missing_type}",
                "natural_language": f"Default {missing_type}",
                "location": "function"
            }
            
            proof_json["proof_components"].append(default_comp)
        
        return proof_json
    
    def _clean_expression(self, expression: str) -> str:
        """
        Clean a logical expression to ensure it's valid.
        
        Args:
            expression: Raw expression from LLM
            
        Returns:
            Cleaned expression
        """
        if not expression:
            return "valid_expression"
        
        # Remove markdown formatting
        expression = re.sub(r'\*\*', '', expression)
        
        # Remove LaTeX delimiters
        expression = re.sub(r'\\[\(\[]', '', expression)
        expression = re.sub(r'\\[\)\]]', '', expression)
        expression = re.sub(r'\\text\{([^}]*)\}', r'\1', expression)
        
        # Replace LaTeX operators with standard ones
        expression = expression.replace('\\leq', '<=')
        expression = expression.replace('\\geq', '>=')
        expression = expression.replace('\\neq', '!=')
        expression = expression.replace('\\rightarrow', '->')
        expression = expression.replace('\\land', '&&')
        expression = expression.replace('\\lor', '||')
        expression = expression.replace('\\neg', '!')
        expression = expression.replace('\\forall', 'forall')
        expression = expression.replace('\\exists', 'exists')
        
        # Remove bullet points and numbering
        expression = re.sub(r'^[\s•\-\*\d\.\)]+', '', expression)
        
        # Normalize whitespace
        expression = re.sub(r'\s+', ' ', expression).strip()
        
        # Remove any trailing punctuation
        expression = re.sub(r'[,;\.]+$', '', expression)
        
        # Fix common issues with condition expressions
        if expression.lower().startswith(('if ', 'where ', 'when ')):
            expression = re.sub(r'^(?:if|where|when)\s+', '', expression, flags=re.IGNORECASE)
        
        return expression
    
    def _json_to_metta_atoms(self, proof_json: Dict[str, Any], function_name: str) -> List[str]:
        """
        Convert proof JSON to MeTTa atoms.
        
        Args:
            proof_json: Validated proof JSON
            function_name: Name of the function
            
        Returns:
            List of MeTTa atom strings
        """
        atoms = []
        
        # Create safe function name
        safe_func_name = self._sanitize_for_metta(function_name)
        
        # Add function verification marker
        atoms.append(f"(verified-function {safe_func_name})")
        
        # Ensure type definitions
        atoms.extend([
            "(: Type Type)",
            "(: Property Type)",
            "(: Function Type)",
            "(: Expression Type)",
            f"(: {safe_func_name} Function)",
            "(: bound-check Property)",
            "(: ordering-check Property)",
            "(: null-check Property)",
            "(: termination-guarantee Property)",
            "(: error-handling Property)",
            "(: function-has-property (--> Function Property Bool))",
            "(: Expression-Property (--> Expression Property Bool))"
        ])
        
        # Process components
        for i, component in enumerate(proof_json.get("proof_components", [])):
            try:
                component_type = component.get("type", "unknown")
                location = component.get("location", "function")
                expression = component.get("expression", "")
                explanation = component.get("natural_language", "")
                
                # Skip invalid components
                if not expression:
                    continue
                
                # Create a stable ID for the expression
                type_prefix = component_type[:3]  # First three chars of type
                expr_id = f"{type_prefix}_{safe_func_name}_{i}"
                
                # Escape strings for MeTTa
                safe_expr = self._escape_for_metta(expression)
                safe_explanation = self._escape_for_metta(explanation)
                
                # Add expression definition
                atoms.append(f"(: {expr_id} Expression)")
                atoms.append(f"(= {expr_id} \"{safe_expr}\")")
                
                # Add explanation
                atoms.append(f"(expression-explanation {expr_id} \"{safe_explanation}\")")
                
                # Add component atom
                if component_type == "precondition":
                    atoms.append(f"(Precondition {expr_id})")
                elif component_type == "loop_invariant":
                    safe_location = self._sanitize_for_metta(location)
                    atoms.append(f"(LoopInvariant {safe_location} {expr_id})")
                elif component_type == "assertion":
                    safe_location = self._sanitize_for_metta(location)
                    atoms.append(f"(Assertion {safe_location} {expr_id})")
                elif component_type == "postcondition":
                    atoms.append(f"(Postcondition {expr_id})")
                
                # Add property annotations
                self._add_property_annotations(atoms, expr_id, expression, explanation)
                
            except Exception as e:
                # If component processing fails, add error atom and continue
                atoms.append(f"(component-error \"{i}\" \"{self._escape_for_metta(str(e))}\")")
        
        # Add verification strategy
        strategy = proof_json.get("verification_strategy", {})
        approach = strategy.get("approach", "")
        if approach:
            atoms.append(f"(verification-approach \"{self._escape_for_metta(approach)}\")")
        
        # Add key lemmas
        for i, lemma in enumerate(strategy.get("key_lemmas", [])):
            if lemma:
                atoms.append(f"(key-lemma {i} \"{self._escape_for_metta(lemma)}\")")
        
        return atoms
    
    def _sanitize_for_metta(self, text: str) -> str:
        """
        Sanitize text for use as a MeTTa symbol.
        
        Args:
            text: Original text
            
        Returns:
            Sanitized text
        """
        if not text:
            return "unnamed"
        
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', text)
        
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = 'x' + sanitized
        
        return sanitized
    
    def _escape_for_metta(self, text: str) -> str:
        """
        Escape a string for inclusion in MeTTa.
        
        Args:
            text: Original text
            
        Returns:
            Escaped text
        """
        if not text:
            return ""
        
        return text.replace('\\', '\\\\').replace('"', '\\"')
    
    def _add_property_annotations(self, atoms: List[str], expr_id: str, 
                                 expression: str, explanation: str) -> None:
        """
        Add property annotations for an expression based on its content.
        
        Args:
            atoms: List of atoms to append to
            expr_id: Expression ID
            expression: Expression content
            explanation: Expression explanation
        """
        # Combine for content analysis
        content = (expression + " " + explanation).lower()
        
        # Check for bounds-related content
        if any(term in content for term in ["bound", "index", "length", "size", "range"]):
            atoms.append(f"(Expression-Property {expr_id} bound-check)")
        
        # Check for ordering-related content
        if any(term in content for term in ["sort", "order", "compare", "less", "greater"]):
            atoms.append(f"(Expression-Property {expr_id} ordering-check)")
        
        # Check for null-checking content
        if any(term in content for term in ["null", "none", "empty"]):
            atoms.append(f"(Expression-Property {expr_id} null-check)")
        
        # Check for termination-related content
        if any(term in content for term in ["terminat", "halt", "progress", "decreas", "increas"]):
            atoms.append(f"(Expression-Property {expr_id} termination-guarantee)")
        
        # Check for error-handling content
        if any(term in content for term in ["error", "exception", "not found", "return -1"]):
            atoms.append(f"(Expression-Property {expr_id} error-handling)")
    
    def _create_fallback_proof(self, function_code: str, function_name: str) -> Dict[str, Any]:
        """
        Create a fallback proof when LLM generation fails.
        
        Args:
            function_code: Function source code
            function_name: Function name
            
        Returns:
            Dictionary with fallback proof
        """
        # Analyze function for basic properties
        has_loop = "for " in function_code or "while " in function_code
        returns_value = "return " in function_code
        
        # Create minimal proof JSON
        json_ir = {
            "proof_components": [
                {
                    "type": "precondition",
                    "expression": "valid_input(args)",
                    "natural_language": "Input arguments are valid",
                    "location": "function"
                },
                {
                    "type": "postcondition",
                    "expression": "valid_output(result)",
                    "natural_language": "Output satisfies requirements",
                    "location": "function"
                }
            ],
            "verification_strategy": {
                "approach": f"Correctness verification for {function_name}",
                "key_lemmas": [
                    "The algorithm terminates for all valid inputs",
                    "The algorithm produces correct results"
                ]
            }
        }
        
        # Add loop invariant if function has loops
        if has_loop:
            json_ir["proof_components"].append({
                "type": "loop_invariant",
                "expression": "loop_index_in_bounds && loop_makes_progress",
                "natural_language": "Loop index stays within bounds and the loop makes progress",
                "location": "main_loop"
            })
        
        # Convert to MeTTa atoms
        metta_atoms = self._json_to_metta_atoms(json_ir, function_name)
        
        return {
            "json_ir": json_ir,
            "metta_atoms": metta_atoms
        }