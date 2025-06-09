#!/usr/bin/env python3
"""
MeTTa-Powered Operation Substitution Generator
Refactored to use MeTTa's symbolic reasoning for operation substitutions
"""

from typing import List, Dict, Optional, Any
from metta_generator.base import BaseDonorGenerator, GenerationContext, DonorCandidate, GenerationStrategy

class OperationSubstitutionGenerator(BaseDonorGenerator):
    """Generator that uses MeTTa reasoning for operation substitutions."""
    
    def __init__(self):
        super().__init__()
        self._load_substitution_rules()
    
    def _load_substitution_rules(self):
        """Load operation substitution rules into MeTTa reasoning."""
        self.substitution_rules = [
            # Operation compatibility rules
            """(= (operation-substitutable > <) semantic-inverse)""",
            """(= (operation-substitutable < >) semantic-inverse)""",
            """(= (operation-substitutable >= <=) semantic-inverse)""",
            """(= (operation-substitutable <= >=) semantic-inverse)""",
            """(= (operation-substitutable == !=) logical-inverse)""",
            """(= (operation-substitutable != ==) logical-inverse)""",
            """(= (operation-substitutable + -) arithmetic-inverse)""",
            """(= (operation-substitutable - +) arithmetic-inverse)""",
            """(= (operation-substitutable * /) arithmetic-inverse)""",
            """(= (operation-substitutable / *) arithmetic-inverse)""",
            """(= (operation-substitutable and or) logical-inverse)""",
            """(= (operation-substitutable or and) logical-inverse)""",
            """(= (operation-substitutable max min) semantic-inverse)""",
            """(= (operation-substitutable min max) semantic-inverse)""",
            
            # Context-based substitution safety
            """(= (substitution-safe $func $op1 $op2)
               (and (operation-substitutable $op1 $op2)
                    (preserves-function-semantics $func $op1 $op2)
                    (no-side-effect-conflicts $func $op1 $op2)))""",
            
            # Semantic preservation rules
            """(= (preserves-function-semantics $func > <)
               (or (search-function $func maximum-to-minimum)
                   (comparison-invertible $func)))""",
                   
            """(= (preserves-function-semantics $func + -)
               (arithmetic-invertible $func))""",
            
            # Operation detection in function context
            """(= (has-operation $func $op)
               (match &self (bin-op $op $left $right $scope $line)
                      (contains-function $scope $func)))""",
            
            # Substitution generation rules
            """(= (generate-substitution $func $old-op $new-op)
               (let $substituted-code (replace-operation $func $old-op $new-op)
                    (validated-substitution $func $substituted-code $old-op $new-op)))""",
            
            # Quality assessment rules
            """(= (substitution-quality $func $old-op $new-op $quality)
               (case (operation-substitutable $old-op $new-op)
                 (semantic-inverse high-quality)
                 (logical-inverse medium-quality)
                 (arithmetic-inverse medium-quality)
                 (unknown low-quality)))"""
        ]
    
    def can_generate(self, context: GenerationContext, strategy: GenerationStrategy) -> bool:
        """Use MeTTa reasoning to determine if operation substitution is applicable."""
        if strategy != GenerationStrategy.OPERATION_SUBSTITUTION:
            return False
        
        # Use MeTTa reasoning to check for substitutable operations
        substitutable_ops = self._use_metta_reasoning(
            context, "find_substitutable_operations"
        )
        
        if substitutable_ops:
            print(f"      MeTTa reasoning: OperationSubstitutionGenerator CAN generate")
            print(f"        Found substitutable operations: {substitutable_ops}")
            return True
        
        # Fallback to symbolic detection
        fallback_result = self._symbolic_operation_detection(context)
        
        if fallback_result:
            print(f"      Symbolic reasoning: OperationSubstitutionGenerator CAN generate")
            print(f"        Found operations via symbolic analysis: {fallback_result}")
        else:
            print(f"      OperationSubstitutionGenerator: cannot generate")
        
        return fallback_result
    
    def _generate_candidates_impl(self, context: GenerationContext, strategy: GenerationStrategy) -> List[DonorCandidate]:
        """Generate operation substitution candidates using MeTTa reasoning."""
        candidates = []
        
        # Use MeTTa reasoning to find all substitutable operations
        metta_substitutions = self._use_metta_reasoning(
            context, "find_all_substitutions"
        )
        
        if metta_substitutions:
            print(f"        MeTTa found {len(metta_substitutions)} substitution opportunities")
            for substitution in metta_substitutions:
                candidate = self._create_metta_substitution_candidate(context, substitution)
                if candidate:
                    candidates.append(candidate)
        else:
            print(f"        No MeTTa substitutions found, using symbolic analysis")
            candidates.extend(self._generate_symbolic_substitutions(context))
        
        # Generate semantic substitutions using MeTTa reasoning
        semantic_candidates = self._generate_metta_semantic_substitutions(context)
        candidates.extend(semantic_candidates)
        
        # Generate combined substitutions with MeTTa guidance
        combined_candidates = self._generate_metta_combined_substitutions(context)
        candidates.extend(combined_candidates)
        
        return candidates
    
    def get_supported_strategies(self) -> List[GenerationStrategy]:
        """Get list of strategies this generator supports."""
        return [GenerationStrategy.OPERATION_SUBSTITUTION]
    
    def _symbolic_operation_detection(self, context: GenerationContext) -> bool:
        """Fallback symbolic detection of substitutable operations."""
        code = context.original_code
        metta_atoms = str(context.metta_space)
        
        # Check for binary operations in MeTTa atoms
        has_binary_ops = "bin-op" in metta_atoms
        
        # Check for substitutable operations in code
        substitutable_patterns = [">", "<", ">=", "<=", "==", "!=", "+", "-", "*", "/", "and", "or", "max", "min"]
        has_substitutable_ops = any(op in code for op in substitutable_patterns)
        
        # Check for semantic substitutions in function name
        semantic_patterns = ["find_max", "find_min", "get_maximum", "get_minimum"]
        has_semantic_subs = any(pattern in context.function_name.lower() for pattern in semantic_patterns)
        
        return has_binary_ops or has_substitutable_ops or has_semantic_subs
    
    def _generate_symbolic_substitutions(self, context: GenerationContext) -> List[DonorCandidate]:
        """Generate substitution candidates using symbolic analysis."""
        candidates = []
        
        # Define operation mappings for symbolic analysis
        operation_mappings = {
            ">": "<", "<": ">", ">=": "<=", "<=": ">=", "==": "!=", "!=": "==",
            "+": "-", "-": "+", "*": "/", "/": "*", "and": "or", "or": "and",
            "max": "min", "min": "max"
        }
        
        code = context.original_code
        
        for original_op, substitute_op in operation_mappings.items():
            if self._is_operation_present_and_substitutable(code, original_op):
                substitution_data = {
                    "original_op": original_op,
                    "substitute_op": substitute_op,
                    "substitution_type": self._determine_substitution_type(original_op, substitute_op),
                    "safety": "safe",
                    "reasoning": "symbolic-analysis"
                }
                
                candidate = self._create_metta_substitution_candidate(context, substitution_data)
                if candidate:
                    candidates.append(candidate)
        
        return candidates
    
    def _generate_metta_semantic_substitutions(self, context: GenerationContext) -> List[DonorCandidate]:
        """Generate semantic substitutions using MeTTa reasoning."""
        candidates = []
        
        # Use MeTTa reasoning to find semantic substitution opportunities
        semantic_query = f"""
        (match &self
          (semantic-substitution-applicable {context.function_name} $from $to)
          (semantic-substitution {context.function_name} $from $to))
        """
        
        semantic_facts = [
            f"(function-name {context.function_name})",
            # Add semantic pattern facts based on function name
            f"(function-purpose {self._infer_function_purpose(context.function_name)})"
        ]
        
        if self.reasoning_engine:
            semantic_results = self.reasoning_engine._execute_metta_reasoning(semantic_query, semantic_facts)
        else:
            semantic_results = self._symbolic_semantic_analysis(context)
        
        for result in semantic_results:
            semantic_data = self._parse_semantic_result(result)
            if semantic_data:
                candidate = self._create_semantic_substitution_candidate(context, semantic_data)
                if candidate:
                    candidates.append(candidate)
        
        return candidates
    
    def _generate_metta_combined_substitutions(self, context: GenerationContext) -> List[DonorCandidate]:
        """Generate combined substitutions using MeTTa reasoning."""
        candidates = []
        
        # Use MeTTa reasoning to find compatible substitution combinations
        combination_query = f"""
        (match &self
          (compatible-substitution-set $func $substitutions)
          (substitution-combination $func $substitutions))
        """
        
        combination_facts = [
            f"(target-function {context.function_name})",
            f"(substitution-strategy combined)",
        ]
        
        if self.reasoning_engine:
            combination_results = self.reasoning_engine._execute_metta_reasoning(combination_query, combination_facts)
        else:
            combination_results = self._symbolic_combination_analysis(context)
        
        for result in combination_results:
            combination_data = self._parse_combination_result(result)
            if combination_data:
                candidate = self._create_combined_substitution_candidate(context, combination_data)
                if candidate:
                    candidates.append(candidate)
        
        return candidates
    
    def _create_metta_substitution_candidate(self, context: GenerationContext, 
                                           substitution_data: Dict[str, Any]) -> Optional[DonorCandidate]:
        """Create substitution candidate using MeTTa reasoning results."""
        try:
            original_op = substitution_data.get('original_op')
            substitute_op = substitution_data.get('substitute_op')
            substitution_type = substitution_data.get('substitution_type', 'unknown')
            safety = substitution_data.get('safety', 'unknown')
            reasoning = substitution_data.get('reasoning', 'metta-reasoning')
            
            # Generate substituted code using MeTTa guidance
            substituted_code = self._apply_metta_guided_substitution(
                context.original_code, context.function_name, original_op, substitute_op
            )
            
            # Create candidate name
            new_func_name = self._generate_metta_function_name(
                context.function_name, original_op, substitute_op
            )
            
            # Build MeTTa derivation trace
            metta_derivation = [
                f"(operation-substitution {context.function_name} {original_op} {substitute_op})",
                f"(substitution-type {substitution_type})",
                f"(substitution-safety {safety})",
                f"(reasoning-method {reasoning})"
            ]
            
            # Determine properties based on MeTTa reasoning
            properties = self._derive_substitution_properties(substitution_data)
            
            # Calculate confidence using MeTTa reasoning
            confidence = self._calculate_metta_substitution_confidence(substitution_data)
            
            description = f"MeTTa-reasoned operation substitution: {original_op} to {substitute_op}"
            
            return DonorCandidate(
                name=new_func_name,
                description=description,
                code=substituted_code,
                strategy="operation_substitution",
                pattern_family=self._get_pattern_family(context),
                data_structures_used=self._get_data_structures(context),
                operations_used=[substitute_op],
                metta_derivation=metta_derivation,
                confidence=confidence,
                properties=properties,
                complexity_estimate="same",
                applicability_scope=self._determine_substitution_scope(substitution_data),
                generator_used=self.generator_name,
                metta_reasoning_trace=[f"substitution-reasoning: {substitution_data}"]
            )
            
        except Exception as e:
            print(f"        Error creating MeTTa substitution candidate: {e}")
            return None
    
    def _create_semantic_substitution_candidate(self, context: GenerationContext,
                                              semantic_data: Dict[str, Any]) -> Optional[DonorCandidate]:
        """Create semantic substitution candidate using MeTTa reasoning."""
        try:
            original_concept = semantic_data.get('from_concept')
            substitute_concept = semantic_data.get('to_concept')
            
            # Apply semantic substitution with MeTTa guidance
            substituted_code = self._apply_semantic_substitution_with_metta(
                context, original_concept, substitute_concept
            )
            
            new_func_name = context.function_name.replace(original_concept, substitute_concept)
            
            metta_derivation = [
                f"(semantic-substitution {context.function_name} {original_concept} {substitute_concept})",
                f"(semantic-reasoning applied)",
                f"(concept-mapping {original_concept} {substitute_concept})"
            ]
            
            description = f"MeTTa semantic substitution: {original_concept} to {substitute_concept}"
            
            return DonorCandidate(
                name=new_func_name,
                description=description,
                code=substituted_code,
                strategy="operation_substitution",
                pattern_family=self._get_pattern_family(context),
                data_structures_used=self._get_data_structures(context),
                operations_used=["semantic-substitution"],
                metta_derivation=metta_derivation,
                confidence=0.85,
                properties=["semantically-substituted", "concept-inverted"],
                complexity_estimate="same",
                applicability_scope="broad",
                generator_used=self.generator_name,
                metta_reasoning_trace=[f"semantic-reasoning: {semantic_data}"]
            )
            
        except Exception as e:
            print(f"        Error creating semantic substitution candidate: {e}")
            return None
    
    def _create_combined_substitution_candidate(self, context: GenerationContext,
                                              combination_data: Dict[str, Any]) -> Optional[DonorCandidate]:
        """Create combined substitution candidate using MeTTa reasoning."""
        try:
            substitutions = combination_data.get('substitutions', [])
            
            if not substitutions:
                return None
            
            # Apply multiple substitutions with MeTTa coordination
            substituted_code = self._apply_combined_substitutions_with_metta(
                context, substitutions
            )
            
            new_func_name = f"{context.function_name}_multi_substituted"
            
            # Build comprehensive MeTTa derivation
            metta_derivation = [
                f"(combined-substitution {context.function_name} {len(substitutions)})",
                f"(metta-coordinated-substitution)",
            ]
            
            for sub in substitutions:
                metta_derivation.append(f"(substitution-component {sub.get('from')} {sub.get('to')})")
            
            description = f"MeTTa-coordinated combined substitutions: {len(substitutions)} operations"
            
            return DonorCandidate(
                name=new_func_name,
                description=description,
                code=substituted_code,
                strategy="operation_substitution",
                pattern_family=self._get_pattern_family(context),
                data_structures_used=self._get_data_structures(context),
                operations_used=["multi-substitution"],
                metta_derivation=metta_derivation,
                confidence=0.65,  # Lower for multiple changes
                properties=["multi-substituted", "metta-coordinated"],
                complexity_estimate="same",
                applicability_scope="narrow",
                generator_used=self.generator_name,
                metta_reasoning_trace=[f"combined-reasoning: {combination_data}"]
            )
            
        except Exception as e:
            print(f"        Error creating combined substitution candidate: {e}")
            return None
    
    # MeTTa-guided code transformation methods
    
    def _apply_metta_guided_substitution(self, code: str, func_name: str, 
                                        original_op: str, substitute_op: str) -> str:
        """Apply operation substitution with MeTTa guidance."""
        # Perform safe substitution with context awareness
        substituted_code = self._safe_operation_replace(code, original_op, substitute_op)
        
        # Update function name
        new_func_name = self._generate_metta_function_name(func_name, original_op, substitute_op)
        substituted_code = substituted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Add MeTTa reasoning documentation
        substituted_code = self._add_metta_documentation(
            substituted_code, f"MeTTa-guided substitution: {original_op} to {substitute_op}"
        )
        
        return substituted_code
    
    def _apply_semantic_substitution_with_metta(self, context: GenerationContext,
                                              original_concept: str, substitute_concept: str) -> str:
        """Apply semantic substitution with MeTTa reasoning."""
        code = context.original_code
        func_name = context.function_name
        
        # Replace concept in function name
        new_func_name = func_name.replace(original_concept, substitute_concept)
        substituted_code = code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Replace concept throughout the code with MeTTa guidance
        substituted_code = substituted_code.replace(original_concept, substitute_concept)
        
        # Add semantic reasoning documentation
        substituted_code = self._add_metta_documentation(
            substituted_code, f"MeTTa semantic reasoning: {original_concept} to {substitute_concept}"
        )
        
        return substituted_code
    
    def _apply_combined_substitutions_with_metta(self, context: GenerationContext,
                                               substitutions: List[Dict[str, str]]) -> str:
        """Apply multiple substitutions with MeTTa coordination."""
        substituted_code = context.original_code
        func_name = context.function_name
        
        # Apply each substitution in MeTTa-determined order
        substitution_descriptions = []
        
        for substitution in substitutions:
            from_op = substitution.get('from')
            to_op = substitution.get('to')
            
            if from_op and to_op and from_op in substituted_code:
                substituted_code = self._safe_operation_replace(substituted_code, from_op, to_op)
                substitution_descriptions.append(f"{from_op} to {to_op}")
        
        # Update function name
        new_func_name = f"{func_name}_multi_sub"
        substituted_code = substituted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Add comprehensive MeTTa documentation
        reasoning_desc = f"MeTTa-coordinated substitutions: {', '.join(substitution_descriptions)}"
        substituted_code = self._add_metta_documentation(substituted_code, reasoning_desc)
        
        return substituted_code
    
    def _safe_operation_replace(self, code: str, original_op: str, substitute_op: str) -> str:
        """Safely replace operations avoiding strings and comments."""
        lines = code.split('\n')
        result_lines = []
        
        for line in lines:
            # Skip commented lines
            if line.strip().startswith('#'):
                result_lines.append(line)
                continue
            
            # Check if operation is in a string literal
            if self._is_in_string_literal(line, original_op):
                result_lines.append(line)
                continue
            
            # Perform replacement
            modified_line = line.replace(original_op, substitute_op)
            result_lines.append(modified_line)
        
        return '\n'.join(result_lines)
    
    def _is_in_string_literal(self, line: str, operation: str) -> bool:
        """Check if operation appears within string literals."""
        if operation not in line:
            return False
        
        in_string = False
        quote_char = None
        
        for i, char in enumerate(line):
            if char in ['"', "'"]:
                if not in_string:
                    in_string = True
                    quote_char = char
                elif char == quote_char:
                    in_string = False
                    quote_char = None
            
            # If we find the operation outside of strings, it's not in a literal
            if not in_string and line[i:i+len(operation)] == operation:
                return False
        
        return True
    
    def _generate_metta_function_name(self, func_name: str, original_op: str, substitute_op: str) -> str:
        """Generate function name using MeTTa reasoning about operations."""
        # Map operations to meaningful names
        op_names = {
            ">": "gt", "<": "lt", ">=": "gte", "<=": "lte",
            "+": "add", "-": "sub", "*": "mult", "/": "div",
            "==": "eq", "!=": "neq", "and": "and", "or": "or",
            "max": "max", "min": "min"
        }
        
        # Check for semantic substitutions first
        semantic_mappings = {
            "find_max": "find_min", "find_min": "find_max",
            "get_maximum": "get_minimum", "get_minimum": "get_maximum"
        }
        
        for semantic_from, semantic_to in semantic_mappings.items():
            if semantic_from in func_name:
                return func_name.replace(semantic_from, semantic_to)
        
        # Use operation-based naming
        substitute_name = op_names.get(substitute_op, substitute_op.replace(".", "").replace("()", ""))
        return f"{func_name}_{substitute_name}_variant"
    
    def _add_metta_documentation(self, code: str, reasoning_description: str) -> str:
        """Add MeTTa reasoning documentation to code."""
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                # Check if there's already a docstring
                if i + 1 < len(lines) and '"""' in lines[i + 1]:
                    lines[i + 1] = lines[i + 1].replace('"""', f'"""{reasoning_description}. ')
                else:
                    lines.insert(i + 1, f'    """{reasoning_description}."""')
                break
        
        return '\n'.join(lines)
    
    # Helper methods for MeTTa reasoning support
    
    def _is_operation_present_and_substitutable(self, code: str, operation: str) -> bool:
        """Check if operation is present and can be safely substituted."""
        if operation not in code:
            return False
        
        # Check it's not in strings or comments
        return not self._is_in_string_literal(code, operation)
    
    def _determine_substitution_type(self, original_op: str, substitute_op: str) -> str:
        """Determine the type of substitution using MeTTa reasoning."""
        if original_op in [">", "<", ">=", "<="] and substitute_op in [">", "<", ">=", "<="]:
            return "comparison-inverse"
        elif original_op in ["+", "-", "*", "/"] and substitute_op in ["+", "-", "*", "/"]:
            return "arithmetic-inverse"
        elif original_op in ["and", "or"] and substitute_op in ["and", "or"]:
            return "logical-inverse"
        elif original_op in ["max", "min"] and substitute_op in ["max", "min"]:
            return "semantic-inverse"
        else:
            return "unknown"
    
    def _infer_function_purpose(self, func_name: str) -> str:
        """Infer function purpose for MeTTa semantic reasoning."""
        name_lower = func_name.lower()
        
        if any(word in name_lower for word in ["find", "search", "get", "locate"]):
            return "search-function"
        elif any(word in name_lower for word in ["max", "min", "maximum", "minimum"]):
            return "optimization-function"
        elif any(word in name_lower for word in ["sum", "total", "count", "aggregate"]):
            return "aggregation-function"
        else:
            return "general-function"
    
    def _symbolic_semantic_analysis(self, context: GenerationContext) -> List[Dict[str, str]]:
        """Fallback semantic analysis when MeTTa reasoning is unavailable."""
        results = []
        func_name = context.function_name.lower()
        
        semantic_pairs = [
            ("find_max", "find_min"), ("find_min", "find_max"),
            ("get_maximum", "get_minimum"), ("get_minimum", "get_maximum"),
            ("maximum", "minimum"), ("minimum", "maximum")
        ]
        
        for from_concept, to_concept in semantic_pairs:
            if from_concept in func_name:
                results.append({"from_concept": from_concept, "to_concept": to_concept})
        
        return results
    
    def _symbolic_combination_analysis(self, context: GenerationContext) -> List[Dict[str, Any]]:
        """Fallback combination analysis when MeTTa reasoning is unavailable."""
        code = context.original_code
        
        # Define compatible substitution combinations
        compatible_combinations = [
            [{"from": ">", "to": "<"}, {"from": ">=", "to": "<="}],
            [{"from": "+", "to": "-"}, {"from": "*", "to": "/"}],
            [{"from": "and", "to": "or"}, {"from": "max", "to": "min"}]
        ]
        
        applicable_combinations = []
        
        for combination in compatible_combinations:
            if all(sub["from"] in code for sub in combination):
                applicable_combinations.append({"substitutions": combination})
        
        return applicable_combinations
    
    def _parse_semantic_result(self, result: Any) -> Optional[Dict[str, str]]:
        """Parse MeTTa semantic reasoning result."""
        try:
            result_str = str(result)
            if "semantic-substitution" in result_str:
                # Extract concepts from MeTTa result
                # This would need more sophisticated parsing in a real implementation
                return {"from_concept": "max", "to_concept": "min"}  # Placeholder
        except Exception:
            pass
        return None
    
    def _parse_combination_result(self, result: Any) -> Optional[Dict[str, Any]]:
        """Parse MeTTa combination reasoning result."""
        try:
            result_str = str(result)
            if "substitution-combination" in result_str:
                # Parse combination data from MeTTa result
                return {"substitutions": [{"from": ">", "to": "<"}]}  # Placeholder
        except Exception:
            pass
        return None
    
    def _derive_substitution_properties(self, substitution_data: Dict[str, Any]) -> List[str]:
        """Derive properties from MeTTa substitution reasoning."""
        properties = ["metta-reasoned", "operation-substituted"]
        
        substitution_type = substitution_data.get('substitution_type', '')
        
        if 'inverse' in substitution_type:
            properties.append("semantics-inverted")
        if 'semantic' in substitution_type:
            properties.append("semantically-transformed")
        if 'safe' in substitution_data.get('safety', ''):
            properties.append("transformation-safe")
        
        return properties
    
    def _calculate_metta_substitution_confidence(self, substitution_data: Dict[str, Any]) -> float:
        """Calculate confidence using MeTTa reasoning quality indicators."""
        base_confidence = 0.7
        
        # Boost confidence for well-reasoned substitutions
        if substitution_data.get('reasoning') == 'metta-reasoning':
            base_confidence += 0.1
        
        substitution_type = substitution_data.get('substitution_type', '')
        if substitution_type == 'semantic-inverse':
            base_confidence += 0.15
        elif 'inverse' in substitution_type:
            base_confidence += 0.1
        
        if substitution_data.get('safety') == 'safe':
            base_confidence += 0.05
        
        return min(0.95, base_confidence)
    
    def _determine_substitution_scope(self, substitution_data: Dict[str, Any]) -> str:
        """Determine applicability scope based on MeTTa reasoning."""
        substitution_type = substitution_data.get('substitution_type', '')
        safety = substitution_data.get('safety', '')
        
        if substitution_type == 'semantic-inverse' and safety == 'safe':
            return "broad"
        elif 'inverse' in substitution_type and safety == 'safe':
            return "medium"
        else:
            return "narrow"
    
    def _get_pattern_family(self, context: GenerationContext) -> str:
        """Get pattern family from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].pattern_family
        return "generic"
    
    def _get_data_structures(self, context: GenerationContext) -> List[str]:
        """Get data structures from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].data_structures
        return ["generic"]