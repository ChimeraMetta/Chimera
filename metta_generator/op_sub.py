#!/usr/bin/env python3
"""
Operation Substitution Generator Module
Generates donor candidates by substituting operations (e.g., > to <, + to -, etc.)
"""

from typing import List, Dict, Optional
from base import BaseDonorGenerator, GenerationContext, DonorCandidate, GenerationStrategy

class OperationSubstitutionGenerator(BaseDonorGenerator):
    """Generator that creates variants by substituting operations."""
    
    def __init__(self):
        self.operation_mappings = {
            # Comparison operations
            ">": "<",
            "<": ">", 
            ">=": "<=",
            "<=": ">=",
            "==": "!=",
            "!=": "==",
            
            # Arithmetic operations
            "+": "-",
            "-": "+",
            "*": "/",
            "/": "*",
            "//": "%",
            "%": "//",
            "**": "//",
            
            # Logical operations
            "and": "or",
            "or": "and",
            
            # String methods
            ".upper()": ".lower()",
            ".lower()": ".upper()",
            ".strip()": ".rstrip()",
            ".lstrip()": ".strip()",
            ".rstrip()": ".lstrip()",
            
            # List methods
            ".append(": ".insert(0, ",
            ".pop()": ".pop(0)",
            ".sort()": ".reverse()",
            ".reverse()": ".sort()",
            
            # Function names (semantic substitutions)
            "max": "min",
            "min": "max",
            "sum": "len",
            "any": "all",
            "all": "any"
        }
        
        self.semantic_substitutions = {
            # Find operations
            "find_max": "find_min",
            "find_min": "find_max",
            "find_first": "find_last",
            "find_last": "find_first",
            "get_maximum": "get_minimum",
            "get_minimum": "get_maximum",
            
            # Sort operations
            "sort_ascending": "sort_descending",
            "sort_descending": "sort_ascending",
            "ascending": "descending",
            "descending": "ascending",
            
            # Direction operations
            "forward": "backward",
            "backward": "forward",
            "left": "right",
            "right": "left",
            "up": "down",
            "down": "up"
        }
    
    def can_generate(self, context: GenerationContext, strategy: GenerationStrategy) -> bool:
        """Check if this generator can handle the given context and strategy."""
        if strategy != GenerationStrategy.OPERATION_SUBSTITUTION:
            return False
        
        # Check if the function has operations that can be substituted
        code = context.original_code
        metta_atoms = str(context.metta_space)
        
        # Check for binary operations in MeTTa atoms
        has_binary_ops = "bin-op" in metta_atoms
        
        # Check for substitutable operations in code
        has_substitutable_ops = any(op in code for op in self.operation_mappings.keys())
        
        # Check for semantic substitutions in function name
        has_semantic_subs = any(pattern in context.function_name.lower() 
                               for pattern in self.semantic_substitutions.keys())
        
        return has_binary_ops or has_substitutable_ops or has_semantic_subs
    
    def generate_candidates(self, context: GenerationContext, strategy: GenerationStrategy) -> List[DonorCandidate]:
        """Generate operation substitution candidates."""
        candidates = []
        
        # 1. Generate operator substitution variants
        candidates.extend(self._generate_operator_substitutions(context))
        
        # 2. Generate semantic substitution variants
        candidates.extend(self._generate_semantic_substitutions(context))
        
        # 3. Generate method substitution variants
        candidates.extend(self._generate_method_substitutions(context))
        
        # 4. Generate combined substitution variants
        candidates.extend(self._generate_combined_substitutions(context))
        
        return candidates
    
    def get_supported_strategies(self) -> List[GenerationStrategy]:
        """Get list of strategies this generator supports."""
        return [GenerationStrategy.OPERATION_SUBSTITUTION]
    
    def _generate_operator_substitutions(self, context: GenerationContext) -> List[DonorCandidate]:
        """Generate candidates by substituting operators."""
        candidates = []
        
        # Find operators that can be substituted
        substitutable_ops = self._find_substitutable_operators(context.original_code)
        
        for original_op, substitute_op in substitutable_ops:
            candidate = self._create_operator_substitution_candidate(
                context, original_op, substitute_op
            )
            if candidate:
                candidates.append(candidate)
        
        return candidates
    
    def _generate_semantic_substitutions(self, context: GenerationContext) -> List[DonorCandidate]:
        """Generate candidates by substituting semantic elements."""
        candidates = []
        
        # Check function name for semantic substitutions
        func_name = context.function_name.lower()
        
        for original_pattern, substitute_pattern in self.semantic_substitutions.items():
            if original_pattern in func_name:
                candidate = self._create_semantic_substitution_candidate(
                    context, original_pattern, substitute_pattern
                )
                if candidate:
                    candidates.append(candidate)
        
        return candidates
    
    def _generate_method_substitutions(self, context: GenerationContext) -> List[DonorCandidate]:
        """Generate candidates by substituting method calls."""
        candidates = []
        
        # Find method calls that can be substituted
        code = context.original_code
        
        for original_method, substitute_method in self.operation_mappings.items():
            if original_method in code and original_method.startswith('.'):
                candidate = self._create_method_substitution_candidate(
                    context, original_method, substitute_method
                )
                if candidate:
                    candidates.append(candidate)
        
        return candidates
    
    def _generate_combined_substitutions(self, context: GenerationContext) -> List[DonorCandidate]:
        """Generate candidates with multiple substitutions combined."""
        candidates = []
        
        # Apply multiple substitutions at once
        combined_substitutions = self._get_compatible_substitution_combinations()
        
        for substitution_set in combined_substitutions:
            candidate = self._create_combined_substitution_candidate(
                context, substitution_set
            )
            if candidate:
                candidates.append(candidate)
        
        return candidates
    
    def _find_substitutable_operators(self, code: str) -> List[tuple]:
        """Find operators in code that can be substituted."""
        substitutable = []
        
        for original_op, substitute_op in self.operation_mappings.items():
            if not original_op.startswith('.') and original_op in code:
                # Avoid substituting inside strings or comments
                if self._is_operator_substitutable_in_context(code, original_op):
                    substitutable.append((original_op, substitute_op))
        
        return substitutable
    
    def _is_operator_substitutable_in_context(self, code: str, operator: str) -> bool:
        """Check if operator can be safely substituted (not in strings/comments)."""
        # Simple heuristic: check if operator is not inside quotes
        lines = code.split('\n')
        
        for line in lines:
            if operator in line:
                # Skip commented lines
                if line.strip().startswith('#'):
                    continue
                
                # Check if operator is in a string literal
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
                    
                    # If we find the operator outside of strings, it's substitutable
                    if not in_string and line[i:i+len(operator)] == operator:
                        return True
        
        return False
    
    def _create_operator_substitution_candidate(self, context: GenerationContext, 
                                              original_op: str, substitute_op: str) -> Optional[DonorCandidate]:
        """Create a candidate with an operator substitution."""
        try:
            # Create substituted code
            substituted_code = context.original_code.replace(original_op, substitute_op)
            
            # Update function name to reflect substitution
            new_func_name = self._generate_substitution_function_name(
                context.function_name, original_op, substitute_op
            )
            substituted_code = substituted_code.replace(
                f"def {context.function_name}(", 
                f"def {new_func_name}("
            )
            
            # Add documentation
            substituted_code = self._add_substitution_documentation(
                substituted_code, f"Substituted '{original_op}' with '{substitute_op}'"
            )
            
            return DonorCandidate(
                name=new_func_name,
                description=f"Operation substitution: {original_op} → {substitute_op}",
                code=substituted_code,
                strategy="operation_substitution",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=self._get_data_structures_from_context(context),
                operations_used=self._get_operations_from_context(context),
                metta_derivation=[
                    f"(operation-substitution {context.function_name} {original_op} {substitute_op})"
                ],
                confidence=self._calculate_substitution_confidence(original_op, substitute_op),
                properties=["operation-substituted", "semantics-inverted"],
                complexity_estimate="same",
                applicability_scope="medium"
            )
            
        except Exception as e:
            print(f"      Failed to create operator substitution candidate: {e}")
            return None
    
    def _create_semantic_substitution_candidate(self, context: GenerationContext,
                                              original_pattern: str, substitute_pattern: str) -> Optional[DonorCandidate]:
        """Create a candidate with semantic substitution."""
        try:
            # Replace in function name
            new_func_name = context.function_name.replace(original_pattern, substitute_pattern)
            
            # Replace in code
            substituted_code = context.original_code.replace(
                f"def {context.function_name}(",
                f"def {new_func_name}("
            )
            
            # Replace semantic elements throughout the code
            substituted_code = substituted_code.replace(original_pattern, substitute_pattern)
            
            # Add documentation
            substituted_code = self._add_substitution_documentation(
                substituted_code, f"Semantic substitution: {original_pattern} → {substitute_pattern}"
            )
            
            return DonorCandidate(
                name=new_func_name,
                description=f"Semantic substitution: {original_pattern} → {substitute_pattern}",
                code=substituted_code,
                strategy="operation_substitution",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=self._get_data_structures_from_context(context),
                operations_used=self._get_operations_from_context(context),
                metta_derivation=[
                    f"(semantic-substitution {context.function_name} {original_pattern} {substitute_pattern})"
                ],
                confidence=0.85,
                properties=["semantically-substituted", "intent-inverted"],
                complexity_estimate="same",
                applicability_scope="broad"
            )
            
        except Exception as e:
            print(f"      Failed to create semantic substitution candidate: {e}")
            return None
    
    def _create_method_substitution_candidate(self, context: GenerationContext,
                                            original_method: str, substitute_method: str) -> Optional[DonorCandidate]:
        """Create a candidate with method substitution."""
        try:
            # Replace method calls
            substituted_code = context.original_code.replace(original_method, substitute_method)
            
            # Update function name
            new_func_name = f"{context.function_name}_method_sub"
            substituted_code = substituted_code.replace(
                f"def {context.function_name}(",
                f"def {new_func_name}("
            )
            
            # Add documentation
            substituted_code = self._add_substitution_documentation(
                substituted_code, f"Method substitution: {original_method} → {substitute_method}"
            )
            
            return DonorCandidate(
                name=new_func_name,
                description=f"Method substitution: {original_method} → {substitute_method}",
                code=substituted_code,
                strategy="operation_substitution",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=self._get_data_structures_from_context(context),
                operations_used=self._get_operations_from_context(context),
                metta_derivation=[
                    f"(method-substitution {context.function_name} {original_method} {substitute_method})"
                ],
                confidence=0.75,
                properties=["method-substituted", "behavior-modified"],
                complexity_estimate="same",
                applicability_scope="medium"
            )
            
        except Exception as e:
            print(f"      Failed to create method substitution candidate: {e}")
            return None
    
    def _create_combined_substitution_candidate(self, context: GenerationContext,
                                              substitution_set: Dict[str, str]) -> Optional[DonorCandidate]:
        """Create a candidate with multiple combined substitutions."""
        try:
            # Apply all substitutions
            substituted_code = context.original_code
            substitution_descriptions = []
            
            for original, substitute in substitution_set.items():
                if original in substituted_code:
                    substituted_code = substituted_code.replace(original, substitute)
                    substitution_descriptions.append(f"{original}→{substitute}")
            
            if not substitution_descriptions:
                return None
            
            # Update function name
            new_func_name = f"{context.function_name}_combined_sub"
            substituted_code = substituted_code.replace(
                f"def {context.function_name}(",
                f"def {new_func_name}("
            )
            
            # Add documentation
            substituted_code = self._add_substitution_documentation(
                substituted_code, f"Combined substitutions: {', '.join(substitution_descriptions)}"
            )
            
            return DonorCandidate(
                name=new_func_name,
                description=f"Combined substitutions: {', '.join(substitution_descriptions)}",
                code=substituted_code,
                strategy="operation_substitution",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=self._get_data_structures_from_context(context),
                operations_used=self._get_operations_from_context(context),
                metta_derivation=[
                    f"(combined-substitution {context.function_name} {len(substitution_set)})"
                ],
                confidence=0.65,  # Lower confidence for multiple changes
                properties=["multi-substituted", "complex-transformation"],
                complexity_estimate="same",
                applicability_scope="narrow"
            )
            
        except Exception as e:
            print(f"      Failed to create combined substitution candidate: {e}")
            return None
    
    def _get_compatible_substitution_combinations(self) -> List[Dict[str, str]]:
        """Get combinations of substitutions that work well together."""
        return [
            # Comparison inversions
            {">": "<", ">=": "<="},
            {"<": ">", "<=": ">="},
            
            # Arithmetic inversions
            {"+": "-", "*": "/"},
            {"-": "+", "/": "*"},
            
            # Logical inversions
            {"and": "or", "any": "all"},
            {"or": "and", "all": "any"},
            
            # String case inversions
            {".upper()": ".lower()", "upper": "lower"},
            {".lower()": ".upper()", "lower": "upper"}
        ]
    
    def _generate_substitution_function_name(self, original_name: str, 
                                           original_op: str, substitute_op: str) -> str:
        """Generate an appropriate function name for the substitution."""
        # Handle semantic substitutions
        if original_op in self.semantic_substitutions:
            return original_name.replace(original_op, substitute_op)
        
        # Handle operator substitutions
        op_names = {
            ">": "gt", "<": "lt", ">=": "gte", "<=": "lte",
            "+": "add", "-": "sub", "*": "mult", "/": "div",
            "==": "eq", "!=": "neq",
            "and": "and", "or": "or"
        }
        
        original_name_part = op_names.get(original_op, original_op.replace(".", "").replace("()", ""))
        substitute_name_part = op_names.get(substitute_op, substitute_op.replace(".", "").replace("()", ""))
        
        return f"{original_name}_{substitute_name_part}_variant"
    
    def _add_substitution_documentation(self, code: str, substitution_description: str) -> str:
        """Add documentation about the substitution to the code."""
        lines = code.split('\n')
        
        # Find the function definition line
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                # Check if there's already a docstring
                if i + 1 < len(lines) and '"""' in lines[i + 1]:
                    # Modify existing docstring
                    lines[i + 1] = lines[i + 1].replace('"""', f'"""{substitution_description}. ')
                else:
                    # Add new docstring
                    lines.insert(i + 1, f'    """{substitution_description}."""')
                break
        
        return '\n'.join(lines)
    
    def _calculate_substitution_confidence(self, original_op: str, substitute_op: str) -> float:
        """Calculate confidence for a specific substitution."""
        base_confidence = 0.8
        
        # Higher confidence for semantic substitutions
        if original_op in self.semantic_substitutions:
            base_confidence = 0.9
        
        # Lower confidence for complex operators
        if len(original_op) > 3 or len(substitute_op) > 3:
            base_confidence -= 0.1
        
        # Higher confidence for common substitutions
        common_substitutions = [">", "<", "+", "-", "max", "min"]
        if original_op in common_substitutions:
            base_confidence += 0.05
        
        return min(0.95, base_confidence)
    
    def _get_primary_pattern_family(self, context: GenerationContext) -> str:
        """Get the primary pattern family from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].pattern_family
        return "generic"
    
    def _get_data_structures_from_context(self, context: GenerationContext) -> List[str]:
        """Get data structures from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].data_structures
        return ["generic"]
    
    def _get_operations_from_context(self, context: GenerationContext) -> List[str]:
        """Get operations from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].operations
        return ["substitution"]