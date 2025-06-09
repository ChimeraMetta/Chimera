#!/usr/bin/env python3
"""
Fixed Operation Substitution Generator
Ensures generate_candidates method is properly implemented
"""

from typing import List, Dict, Any, Optional
import re
from metta_generator.base import BaseDonorGenerator, GenerationContext, DonorCandidate, GenerationStrategy

class OperationSubstitutionGenerator(BaseDonorGenerator):
    """Fixed generator that creates alternative implementations by substituting operations."""
    
    def __init__(self):
        super().__init__()
        
        # Define operation substitution mappings
        self.operation_mappings = {
            # Comparison operations
            ">": [">=", "!="],
            "<": ["<=", "!="], 
            ">=": [">"],
            "<=": ["<"],
            "==": ["!="],
            "!=": ["=="],
            
            # Arithmetic operations
            "+": ["-", "*"],
            "-": ["+", "/"],
            "*": ["+", "//"],
            "/": ["*", "//"],
            "//": ["/", "%"],
            "%": ["//"],
            
            # Built-in functions
            "max": ["min"],
            "min": ["max"],
            "sum": ["len", "max"],
            "len": ["sum"],
            "abs": ["int"],
            
            # String operations
            ".upper()": [".lower()", ".title()"],
            ".lower()": [".upper()", ".strip()"],
            ".strip()": [".lower()", ".rstrip()"],
            ".split()": [".join()"],
            
            # List operations
            ".append(": [".insert(0, ", ".extend(["],
            ".extend(": [".append("],
            ".remove(": [".pop("],
            ".pop()": [".remove("]
        }
    
    def can_generate(self, context: GenerationContext, strategy) -> bool:
        """Check if this generator can handle the given context and strategy."""
        if hasattr(strategy, 'value'):
            strategy_name = strategy.value
        else:
            strategy_name = str(strategy)
            
        if strategy_name != "operation_substitution":
            return False
        
        # Check if function contains substitutable operations
        return self._has_substitutable_operations(context.original_code)
    
    def generate_candidates(self, context: GenerationContext, strategy) -> List[DonorCandidate]:
        """Generate donor candidates by substituting operations."""
        candidates = self._generate_candidates_impl(context, strategy)
        
        # Ensure all candidates have proper generator attribution
        for candidate in candidates:
            if not hasattr(candidate, 'generator_used') or candidate.generator_used == "UnknownGenerator":
                candidate.generator_used = self.generator_name
            
            # Add MeTTa reasoning trace if not present
            if not getattr(candidate, 'metta_reasoning_trace', None):
                candidate.metta_reasoning_trace = [f"generated-by {self.generator_name}"]
        
        return candidates
    
    def _generate_candidates_impl(self, context: GenerationContext, strategy) -> List[DonorCandidate]:
        """Generate operation substitution candidates."""
        candidates = []
        
        # Find all substitutable operations in the code
        substitutable_ops = self._find_substitutable_operations(context.original_code)
        print(f"      Found substitutable operations: {list(substitutable_ops.keys())}")
        
        # Generate candidates for each operation substitution
        for original_op, substitutions in substitutable_ops.items():
            for substitute_op in substitutions[:2]:  # Limit to 2 substitutions per operation
                try:
                    candidate = self._create_substitution_candidate(
                        context, original_op, substitute_op
                    )
                    if candidate:
                        candidates.append(candidate)
                        print(f"         Created {original_op} -> {substitute_op} candidate")
                except Exception as e:
                    print(f"         Failed {original_op} -> {substitute_op}: {e}")
        
        return candidates
    
    def get_supported_strategies(self) -> List:
        """Get list of strategies this generator supports."""
        return [GenerationStrategy.OPERATION_SUBSTITUTION]
    
    def _has_substitutable_operations(self, code: str) -> bool:
        """Check if code contains operations that can be substituted."""
        for operation in self.operation_mappings.keys():
            if operation in code:
                return True
        return False
    
    def _find_substitutable_operations(self, code: str) -> Dict[str, List[str]]:
        """Find all substitutable operations in the code."""
        found_operations = {}
        
        for operation, substitutions in self.operation_mappings.items():
            if operation in code:
                found_operations[operation] = substitutions
        
        return found_operations
    
    def _create_substitution_candidate(self, context: GenerationContext,
                                     original_op: str, substitute_op: str) -> Optional[DonorCandidate]:
        """Create a candidate with operation substitution."""
        try:
            # Perform the substitution
            substituted_code = self._perform_substitution(
                context.original_code, context.function_name, original_op, substitute_op
            )
            
            if not substituted_code or substituted_code == context.original_code:
                return None
            
            # Create new function name
            safe_original = original_op.replace(".", "_").replace("(", "").replace(")", "").replace("/", "_div_")
            safe_substitute = substitute_op.replace(".", "_").replace("(", "").replace(")", "").replace("/", "_div_")
            new_func_name = f"{context.function_name}_{safe_original}_to_{safe_substitute}"
            
            # Calculate confidence based on operation compatibility
            confidence = self._calculate_substitution_confidence(original_op, substitute_op)
            
            return DonorCandidate(
                name=new_func_name,
                description=f"Operation substitution: {original_op} -> {substitute_op}",
                code=substituted_code,
                strategy="operation_substitution",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=self._get_data_structures_from_context(context),
                operations_used=[substitute_op],
                metta_derivation=[
                    f"(operation-substitution {context.function_name} {original_op} {substitute_op})"
                ],
                confidence=confidence,
                properties=["operation-substituted", f"{safe_original}-to-{safe_substitute}"],
                complexity_estimate="same",
                applicability_scope="medium",
                generator_used=self.generator_name
            )
            
        except Exception as e:
            print(f"        Error creating substitution candidate {original_op}->{substitute_op}: {e}")
            return None
    
    def _perform_substitution(self, code: str, func_name: str, 
                            original_op: str, substitute_op: str) -> str:
        """Perform the actual operation substitution in the code."""
        # Replace function name
        new_func_name = f"{func_name}_{original_op.replace('.', '_').replace('(', '').replace(')', '').replace('/', '_div_')}_to_{substitute_op.replace('.', '_').replace('(', '').replace(')', '').replace('/', '_div_')}"
        substituted_code = code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Perform the operation substitution
        if original_op.endswith("()") and substitute_op.endswith("()"):
            # Method calls
            substituted_code = substituted_code.replace(original_op, substitute_op)
        elif original_op.startswith(".") and substitute_op.startswith("."):
            # Method calls with parameters
            if original_op.endswith("("):
                # Handle method calls with parameters
                pattern = re.escape(original_op)
                replacement = substitute_op
                substituted_code = re.sub(pattern, replacement, substituted_code)
            else:
                # Simple method replacement
                substituted_code = substituted_code.replace(original_op, substitute_op)
        else:
            # Operators and functions
            # Use word boundaries for operator replacement to avoid partial matches
            if original_op in ["max", "min", "sum", "len", "abs"]:
                pattern = r'\b' + re.escape(original_op) + r'\b'
                substituted_code = re.sub(pattern, substitute_op, substituted_code)
            else:
                # For operators, be more careful about context
                substituted_code = substituted_code.replace(f" {original_op} ", f" {substitute_op} ")
                substituted_code = substituted_code.replace(f"({original_op} ", f"({substitute_op} ")
                substituted_code = substituted_code.replace(f" {original_op})", f" {substitute_op})")
        
        # Add documentation about the substitution
        substituted_code = self._add_substitution_docstring(
            substituted_code, original_op, substitute_op
        )
        
        return substituted_code
    
    def _add_substitution_docstring(self, code: str, original_op: str, substitute_op: str) -> str:
        """Add documentation about the operation substitution."""
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                if i + 1 < len(lines) and '"""' in lines[i + 1]:
                    lines[i + 1] = lines[i + 1].replace('"""', f'"""Operation substitution: {original_op} -> {substitute_op}. ')
                else:
                    lines.insert(i + 1, f'    """Operation substitution: {original_op} -> {substitute_op}."""')
                break
        
        return '\n'.join(lines)
    
    def _calculate_substitution_confidence(self, original_op: str, substitute_op: str) -> float:
        """Calculate confidence for the operation substitution."""
        # Base confidence
        base_confidence = 0.6
        
        # Boost confidence for semantically similar operations
        semantic_groups = [
            ["max", "min"],
            [">", ">="],
            ["<", "<="],
            ["==", "!="],
            ["+", "-"],
            ["*", "/"],
            [".upper()", ".lower()"],
            [".append(", ".extend("]
        ]
        
        for group in semantic_groups:
            if original_op in group and substitute_op in group:
                base_confidence += 0.2
                break
        
        # Adjust based on operation type
        if original_op in ["max", "min"] and substitute_op in ["max", "min"]:
            base_confidence += 0.1  # Very similar semantics
        elif original_op in [">", "<"] and substitute_op in [">=", "<="]:
            base_confidence += 0.05  # Similar but boundary conditions differ
        elif original_op in ["+", "-"] and substitute_op in ["*", "/"]:
            base_confidence -= 0.1  # Different operation types
        
        return min(0.9, max(0.3, base_confidence))
    
    def _get_primary_pattern_family(self, context: GenerationContext) -> str:
        """Get the primary pattern family from context."""
        if hasattr(context, 'detected_patterns') and context.detected_patterns:
            return context.detected_patterns[0].pattern_family
        return "generic"
    
    def _get_data_structures_from_context(self, context: GenerationContext) -> List[str]:
        """Get data structures from context."""
        if hasattr(context, 'detected_patterns') and context.detected_patterns:
            return context.detected_patterns[0].data_structures
        return ["generic"]