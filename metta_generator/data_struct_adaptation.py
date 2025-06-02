#!/usr/bin/env python3
"""
Complete Data Structure Adaptation Generator Module
All methods implemented and working - no missing references
"""

from typing import List, Dict, Any, Optional, Set
import re
import ast

from metta_generator.base import BaseDonorGenerator, GenerationContext, DonorCandidate, GenerationStrategy

class DataStructureAdaptationGenerator(BaseDonorGenerator):
    """Complete generator with all methods implemented."""
    
    def __init__(self):
        print("    Initializing Complete Data Structure Adaptation Generator...")
        super().__init__() 
        
        # Only include mappings for methods that are actually implemented
        self.structure_mappings = {
            "list": {
                "set": self._adapt_list_to_set,
                "tuple": self._adapt_list_to_tuple,
                "dict": self._adapt_list_to_dict,
                "generator": self._adapt_list_to_generator
            },
            "dict": {
                "list": self._adapt_dict_to_list,
                "namedtuple": self._adapt_dict_to_namedtuple,
                "set": self._adapt_dict_to_set
            },
            "set": {
                "list": self._adapt_set_to_list,
                "frozenset": self._adapt_set_to_frozenset,
                "dict": self._adapt_set_to_dict
            },
            "string": {
                "list": self._adapt_string_to_list,
                "bytes": self._adapt_string_to_bytes
            },
            "tuple": {
                "list": self._adapt_tuple_to_list,
                "namedtuple": self._adapt_tuple_to_namedtuple
            }
        }
        
        self.compatibility_scores = {
            ("list", "tuple"): 0.9,
            ("list", "set"): 0.8,
            ("list", "dict"): 0.7,
            ("dict", "namedtuple"): 0.9,
            ("set", "frozenset"): 0.95,
            ("string", "list"): 0.8,
            ("tuple", "list"): 0.9,
            ("dict", "list"): 0.8,
            ("set", "list"): 0.85,
            ("string", "bytes"): 0.9
        }
        
        print("     Complete Data Structure Adaptation Generator initialized")
    
    def can_generate(self, context: GenerationContext, strategy) -> bool:
        """Check if this generator can handle the given context and strategy."""
        if hasattr(strategy, 'value'):
            strategy_name = strategy.value
        else:
            strategy_name = str(strategy)
            
        if strategy_name != "data_structure_adaptation":
            return False
        
        # Check if function works with adaptable data structures
        detected_structures = self._detect_data_structures_in_context(context)
        
        # Check if any detected structures have adaptation mappings
        for structure in detected_structures:
            if structure in self.structure_mappings:
                return True
        
        return len(detected_structures) > 0  # Can always try generic adaptations
    
    def _generate_candidates_impl(self, context: GenerationContext, strategy) -> List[DonorCandidate]:
        """Generate data structure adaptation candidates."""
        candidates = []
        
        detected_structures = self._detect_data_structures_in_context(context)
        print(f"      Detected structures: {detected_structures}")
        
        for source_structure in detected_structures:
            if source_structure in self.structure_mappings:
                target_structures = self.structure_mappings[source_structure]
                
                for target_structure, adapter_func in target_structures.items():
                    try:
                        candidate = self._create_adaptation_candidate(
                            context, source_structure, target_structure, adapter_func
                        )
                        if candidate:
                            candidates.append(candidate)
                            print(f"         Created {source_structure}→{target_structure} candidate")
                    except Exception as e:
                        print(f"         Failed {source_structure}→{target_structure}: {e}")
        
        # Always add generic adaptations
        generic_candidates = self._generate_generic_adaptations(context)
        candidates.extend(generic_candidates)
        print(f"         Added {len(generic_candidates)} generic candidates")
        
        return candidates
    
    def get_supported_strategies(self) -> List:
        """Get list of strategies this generator supports."""
        return ["data_structure_adaptation"]
    
    def _detect_data_structures_in_context(self, context: GenerationContext) -> List[str]:
        """Detect data structures used in the function context."""
        structures = set()
        code = context.original_code.lower()
        
        # Simple detection based on code patterns
        if any(pattern in code for pattern in ['[', 'list(', '.append(', '.extend(']):
            structures.add("list")
        
        if any(pattern in code for pattern in ['{', 'dict(', '.keys(', '.values(', '.items(']):
            structures.add("dict")
        
        if any(pattern in code for pattern in ['set(', '.add(', '.union(', '.intersection(']):
            structures.add("set")
        
        if any(pattern in code for pattern in ['"', "'", 'str(', '.split(', '.join(', '.strip(']):
            structures.add("string")
        
        if any(pattern in code for pattern in ['tuple(', '(', ',)']):
            structures.add("tuple")
        
        # If nothing detected, assume list (most common)
        if not structures:
            structures.add("list")
        
        return list(structures)
    
    def _create_adaptation_candidate(self, context: GenerationContext,
                                   source_structure: str, target_structure: str,
                                   adapter_func: callable) -> Optional[DonorCandidate]:
        """Create a data structure adaptation candidate."""
        try:
            # Apply the adaptation
            adapted_code = adapter_func(context.original_code, context.function_name)
            
            if not adapted_code or adapted_code == context.original_code:
                return None
            
            # Calculate compatibility score
            compatibility = self.compatibility_scores.get((source_structure, target_structure), 0.7)
            
            new_func_name = f"{context.function_name}_{target_structure}_adapted"
            
            return DonorCandidate(
                name=new_func_name,
                description=f"Adapted from {source_structure} to {target_structure}",
                code=adapted_code,
                strategy="data_structure_adaptation",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=[target_structure],
                operations_used=self._get_operations_from_context(context),
                metta_derivation=[
                    f"(data-structure-adaptation {context.function_name} {source_structure} {target_structure})"
                ],
                confidence=compatibility,
                properties=["structure-adapted", f"{source_structure}-to-{target_structure}"],
                complexity_estimate="same" if compatibility > 0.8 else "slightly-higher",
                applicability_scope="broad" if compatibility > 0.8 else "medium",
                generator_used=self.generator_name
            )
            
        except Exception as e:
            print(f"        Error creating adaptation candidate {source_structure}→{target_structure}: {e}")
            return None
    
    def _generate_generic_adaptations(self, context: GenerationContext) -> List[DonorCandidate]:
        """Generate generic adaptations that work with multiple data structures."""
        candidates = []
        
        # Generic iterable adaptation
        try:
            generic_candidate = self._create_generic_iterable_candidate(context)
            if generic_candidate:
                candidates.append(generic_candidate)
        except Exception as e:
            print(f"        Error creating generic iterable candidate: {e}")
        
        return candidates
    
    # ============================================================================
    # ADAPTATION METHODS - All implemented
    # ============================================================================
    
    def _adapt_list_to_set(self, code: str, func_name: str) -> str:
        """Adapt list-based code to work with sets."""
        adapted_code = code
        new_func_name = f"{func_name}_set_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Replace list operations with set operations
        adapted_code = adapted_code.replace("[]", "set()")
        adapted_code = adapted_code.replace(".append(", ".add(")
        adapted_code = adapted_code.replace(".extend(", ".update(")
        adapted_code = adapted_code.replace(".remove(", ".discard(")
        
        # Handle indexing (sets don't support indexing)
        adapted_code = re.sub(r'(\w+)\[(\d+)\]', r'list(\1)[\2]', adapted_code)
        adapted_code = re.sub(r'(\w+)\[(\w+)\]', r'list(\1)[\2]', adapted_code)
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "list", "set")
        
        return adapted_code
    
    def _adapt_list_to_tuple(self, code: str, func_name: str) -> str:
        """Adapt list-based code to work with tuples."""
        adapted_code = code
        new_func_name = f"{func_name}_tuple_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Replace mutable operations with immutable alternatives
        adapted_code = adapted_code.replace("[]", "()")
        
        # Handle append operations (tuples are immutable)
        adapted_code = re.sub(
            r'(\w+)\.append\(([^)]+)\)',
            r'\1 = \1 + (\2,)',
            adapted_code
        )
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "list", "tuple")
        
        return adapted_code
    
    def _adapt_list_to_dict(self, code: str, func_name: str) -> str:
        """Adapt list-based code to work with dictionaries."""
        adapted_code = code
        new_func_name = f"{func_name}_dict_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Replace list operations with dict operations
        adapted_code = adapted_code.replace("[]", "{}")
        
        # Handle append operations - convert to dict assignment
        adapted_code = re.sub(
            r'(\w+)\.append\(([^)]+)\)',
            r'\1[len(\1)] = \2',
            adapted_code
        )
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "list", "dict")
        
        return adapted_code
    
    def _adapt_list_to_generator(self, code: str, func_name: str) -> str:
        """Adapt list-based code to work with generators."""
        adapted_code = code
        new_func_name = f"{func_name}_generator_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Add generator logic
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if 'return ' in line and '[' in line:
                # Replace return with yield from
                lines[i] = line.replace('return [', 'yield from [')
        
        adapted_code = '\n'.join(lines)
        adapted_code = self._add_adaptation_docstring(adapted_code, "list", "generator")
        
        return adapted_code
    
    def _adapt_dict_to_list(self, code: str, func_name: str) -> str:
        """Adapt dict-based code to work with lists."""
        adapted_code = code
        new_func_name = f"{func_name}_list_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Replace dict operations with list operations
        adapted_code = adapted_code.replace("{}", "[]")
        adapted_code = adapted_code.replace(".keys()", "range(len(data))")
        adapted_code = adapted_code.replace(".values()", "data")
        adapted_code = adapted_code.replace(".items()", "enumerate(data)")
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "dict", "list")
        
        return adapted_code
    
    def _adapt_dict_to_namedtuple(self, code: str, func_name: str) -> str:
        """Adapt dict-based code to work with namedtuples."""
        adapted_code = "from collections import namedtuple\n\n" + code
        new_func_name = f"{func_name}_namedtuple_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Add namedtuple definition
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 2, "    DataRecord = namedtuple('DataRecord', ['field1', 'field2', 'field3'])")
                break
        
        adapted_code = '\n'.join(lines)
        adapted_code = self._add_adaptation_docstring(adapted_code, "dict", "namedtuple")
        
        return adapted_code
    
    def _adapt_dict_to_set(self, code: str, func_name: str) -> str:
        """Adapt dict-based code to work with sets."""
        adapted_code = code
        new_func_name = f"{func_name}_set_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Replace dict operations with set operations
        adapted_code = adapted_code.replace("{}", "set()")
        adapted_code = adapted_code.replace(".keys()", "")
        adapted_code = adapted_code.replace(".values()", "")
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "dict", "set")
        
        return adapted_code
    
    def _adapt_set_to_list(self, code: str, func_name: str) -> str:
        """Adapt set-based code to work with lists."""
        adapted_code = code
        new_func_name = f"{func_name}_list_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Replace set operations with list operations
        adapted_code = adapted_code.replace("set()", "[]")
        adapted_code = adapted_code.replace(".add(", ".append(")
        adapted_code = adapted_code.replace(".update(", ".extend(")
        adapted_code = adapted_code.replace(".discard(", ".remove(")
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "set", "list")
        
        return adapted_code
    
    def _adapt_set_to_frozenset(self, code: str, func_name: str) -> str:
        """Adapt set-based code to work with frozensets."""
        adapted_code = code
        new_func_name = f"{func_name}_frozenset_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Replace mutable set operations
        adapted_code = adapted_code.replace("set()", "frozenset()")
        adapted_code = adapted_code.replace(".add(", ".union({")
        adapted_code = adapted_code.replace(".update(", ".union(")
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "set", "frozenset")
        
        return adapted_code
    
    def _adapt_set_to_dict(self, code: str, func_name: str) -> str:
        """Adapt set-based code to work with dictionaries."""
        adapted_code = code
        new_func_name = f"{func_name}_dict_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Replace set operations with dict operations
        adapted_code = adapted_code.replace("set()", "{}")
        adapted_code = re.sub(r'\.add\(([^)]+)\)', r'[\1] = True', adapted_code)
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "set", "dict")
        
        return adapted_code
    
    def _adapt_string_to_list(self, code: str, func_name: str) -> str:
        """Adapt string-based code to work with lists."""
        adapted_code = code
        new_func_name = f"{func_name}_list_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Add conversion logic
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 2, "    # Convert string to list of characters")
                lines.insert(i + 3, "    if isinstance(text, str):")
                lines.insert(i + 4, "        text = list(text)")
                break
        
        adapted_code = '\n'.join(lines)
        adapted_code = self._add_adaptation_docstring(adapted_code, "string", "list")
        
        return adapted_code
    
    def _adapt_string_to_bytes(self, code: str, func_name: str) -> str:
        """Adapt string-based code to work with bytes."""
        adapted_code = code
        new_func_name = f"{func_name}_bytes_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Add encoding logic
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 2, "    # Convert string to bytes")
                lines.insert(i + 3, "    if isinstance(text, str):")
                lines.insert(i + 4, "        text = text.encode('utf-8')")
                break
        
        adapted_code = '\n'.join(lines)
        adapted_code = self._add_adaptation_docstring(adapted_code, "string", "bytes")
        
        return adapted_code
    
    def _adapt_tuple_to_list(self, code: str, func_name: str) -> str:
        """Adapt tuple-based code to work with lists."""
        adapted_code = code
        new_func_name = f"{func_name}_list_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Replace tuple operations with list operations
        adapted_code = adapted_code.replace("()", "[]")
        adapted_code = adapted_code.replace("tuple(", "list(")
        
        # Add conversion logic
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 2, "    # Convert tuple to list if needed")
                lines.insert(i + 3, "    if isinstance(data, tuple):")
                lines.insert(i + 4, "        data = list(data)")
                break
        
        adapted_code = '\n'.join(lines)
        adapted_code = self._add_adaptation_docstring(adapted_code, "tuple", "list")
        
        return adapted_code
    
    def _adapt_tuple_to_namedtuple(self, code: str, func_name: str) -> str:
        """Adapt tuple-based code to work with namedtuples."""
        adapted_code = "from collections import namedtuple\n\n" + code
        new_func_name = f"{func_name}_namedtuple_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Add namedtuple definition
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 2, "    DataTuple = namedtuple('DataTuple', ['field1', 'field2', 'field3'])")
                break
        
        adapted_code = '\n'.join(lines)
        adapted_code = self._add_adaptation_docstring(adapted_code, "tuple", "namedtuple")
        
        return adapted_code
    
    # ============================================================================
    # GENERIC ADAPTATION METHODS
    # ============================================================================
    
    def _create_generic_iterable_candidate(self, context: GenerationContext) -> Optional[DonorCandidate]:
        """Create a candidate that works with any iterable."""
        try:
            adapted_code = self._create_generic_iterable_code(context.original_code, context.function_name)
            
            return DonorCandidate(
                name=f"{context.function_name}_iterable_generic",
                description="Generic adaptation to work with any iterable",
                code=adapted_code,
                strategy="data_structure_adaptation",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=["iterable"],
                operations_used=self._get_operations_from_context(context),
                metta_derivation=[f"(generic-iterable-adaptation {context.function_name})"],
                confidence=0.85,
                properties=["generic", "iterable-compatible"],
                complexity_estimate="same",
                applicability_scope="broad",
                generator_used=self.generator_name
            )
        except Exception as e:
            print(f"        Error creating generic iterable candidate: {e}")
            return None
    
    def _create_generic_iterable_code(self, code: str, func_name: str) -> str:
        """Create code that works with any iterable."""
        new_func_name = f"{func_name}_iterable_generic"
        adapted_code = "from typing import Iterable, Any\n\n" + code
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Add conversion logic
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 2, "    \"\"\"Generic adaptation to work with any iterable.\"\"\"")
                lines.insert(i + 3, "    # Convert to list for uniform handling")
                lines.insert(i + 4, "    if not isinstance(data, list):")
                lines.insert(i + 5, "        data = list(data)")
                break
        
        return '\n'.join(lines)
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _add_adaptation_docstring(self, code: str, from_type: str, to_type: str) -> str:
        """Add documentation about the adaptation."""
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                if i + 1 < len(lines) and '"""' in lines[i + 1]:
                    lines[i + 1] = lines[i + 1].replace('"""', f'"""Adapted from {from_type} to {to_type}. ')
                else:
                    lines.insert(i + 1, f'    """Adapted from {from_type} to {to_type}."""')
                break
        
        return '\n'.join(lines)
    
    def _get_primary_pattern_family(self, context: GenerationContext) -> str:
        """Get the primary pattern family from context."""
        if hasattr(context, 'detected_patterns') and context.detected_patterns:
            return context.detected_patterns[0].pattern_family
        return "generic"
    
    def _get_operations_from_context(self, context: GenerationContext) -> List[str]:
        """Get operations from context."""
        if hasattr(context, 'detected_patterns') and context.detected_patterns:
            return context.detected_patterns[0].operations
        return ["adaptation"]


# For standalone testing
def test_complete_generator():
    """Test the complete generator standalone."""
    print("🧪 TESTING COMPLETE DATA STRUCTURE ADAPTATION GENERATOR")
    print("=" * 60)
    
    # Create test context
    class TestContext:
        def __init__(self, code, func_name):
            self.original_code = code
            self.function_name = func_name
            self.detected_patterns = [TestPattern()]
    
    class TestPattern:
        def __init__(self):
            self.pattern_family = "search"
            self.operations = ["comparison"]
    
    # Test function
    test_code = '''def find_max_in_range(numbers, start_idx, end_idx):
    """Find the maximum value in a list within a specific range."""
    if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
        return None
    
    max_val = numbers[start_idx]
    for i in range(start_idx + 1, end_idx):
        if numbers[i] > max_val:
            max_val = numbers[i]
    
    return max_val'''
    
    # Create generator and context
    generator = DataStructureAdaptationGenerator()
    context = TestContext(test_code, "find_max_in_range")
    
    # Test if it can generate
    can_generate = generator.can_generate(context, "data_structure_adaptation")
    print(f"Can generate: {can_generate}")
    
    if can_generate:
        # Generate candidates
        candidates = generator.generate_candidates(context, "data_structure_adaptation")
        print(f"Generated {len(candidates)} candidates:")
        
        for i, candidate in enumerate(candidates, 1):
            print(f"\n{i}. {candidate.name}")
            print(f"   Description: {candidate.description}")
            print(f"   Confidence: {candidate.confidence}")
            
            # Show code preview
            code_lines = candidate.code.split('\n')[:5]
            for line in code_lines:
                if line.strip():
                    print(f"   {line}")
    
    return True


if __name__ == "__main__":
    test_complete_generator()