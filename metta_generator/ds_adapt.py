#!/usr/bin/env python3
"""
Data Structure Adaptation Generator Module
Generates donor candidates by adapting functions to work with different data structures
"""

from typing import List, Optional
import re
from metta_generator.base import BaseDonorGenerator, GenerationContext, DonorCandidate, GenerationStrategy

class DataStructureAdaptationGenerator(BaseDonorGenerator):
    """Generator that creates variants by adapting data structures."""
    
    def __init__(self):
        self.structure_mappings = {
            "list": {
                "set": self._adapt_list_to_set,
                "tuple": self._adapt_list_to_tuple,
                "dict": self._adapt_list_to_dict,
                "generator": self._adapt_list_to_generator,
                "numpy_array": self._adapt_list_to_numpy
            },
            "dict": {
                "list": self._adapt_dict_to_list,
                "namedtuple": self._adapt_dict_to_namedtuple,
                "dataclass": self._adapt_dict_to_dataclass,
                "set": self._adapt_dict_to_set
            },
            "set": {
                "list": self._adapt_set_to_list,
                "frozenset": self._adapt_set_to_frozenset,
                "dict": self._adapt_set_to_dict
            },
            "string": {
                "list": self._adapt_string_to_list,
                "bytes": self._adapt_string_to_bytes,
                "array": self._adapt_string_to_array
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
            ("tuple", "list"): 0.9
        }
    
    def can_generate(self, context: GenerationContext, strategy: GenerationStrategy) -> bool:
        """Check if this generator can handle the given context and strategy."""
        if strategy != GenerationStrategy.DATA_STRUCTURE_ADAPTATION:
            return False
        
        # Check if function works with adaptable data structures
        detected_structures = self._detect_data_structures_in_context(context)
        
        # Check if any detected structures have adaptation mappings
        for structure in detected_structures:
            if structure in self.structure_mappings:
                return True
        
        return False
    
    def generate_candidates(self, context: GenerationContext, strategy: GenerationStrategy) -> List[DonorCandidate]:
        """Generate data structure adaptation candidates."""
        candidates = []
        
        detected_structures = self._detect_data_structures_in_context(context)
        
        for source_structure in detected_structures:
            if source_structure in self.structure_mappings:
                target_structures = self.structure_mappings[source_structure]
                
                for target_structure, adapter_func in target_structures.items():
                    candidate = self._create_adaptation_candidate(
                        context, source_structure, target_structure, adapter_func
                    )
                    if candidate:
                        candidates.append(candidate)
        
        # Generate generic adaptations
        candidates.extend(self._generate_generic_adaptations(context))
        
        # Generate protocol-based adaptations
        candidates.extend(self._generate_protocol_adaptations(context))
        
        return candidates
    
    def get_supported_strategies(self) -> List[GenerationStrategy]:
        """Get list of strategies this generator supports."""
        return [GenerationStrategy.DATA_STRUCTURE_ADAPTATION]
    
    def _detect_data_structures_in_context(self, context: GenerationContext) -> List[str]:
        """Detect data structures used in the function context."""
        structures = set()
        code = context.original_code.lower()
        atoms_str = str(context.metta_space)
        
        # Detection patterns
        structure_patterns = {
            "list": [r'\[.*\]', r'\.append\(', r'\.extend\(', r'list\(', r'List\['],
            "dict": [r'\{.*:.*\}', r'\.keys\(\)', r'\.values\(\)', r'\.items\(\)', r'dict\(', r'Dict\['],
            "set": [r'set\(', r'\.add\(', r'\.union\(', r'\.intersection\(', r'Set\['],
            "tuple": [r'tuple\(', r'\(.*,.*\)', r'Tuple\['],
            "string": [r'str\(', r'\".*\"', r"'.*'", r'\.split\(', r'\.join\(', r'String'],
            "generator": [r'yield\s+', r'for.*in.*:', r'Generator\['],
            "numpy_array": [r'np\.array', r'numpy\.array', r'ndarray']
        }
        
        for structure, patterns in structure_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code) or structure in atoms_str:
                    structures.add(structure)
        
        # Check MeTTa atoms for type information
        if "List" in atoms_str:
            structures.add("list")
        if "Dict" in atoms_str:
            structures.add("dict")
        if "Set" in atoms_str:
            structures.add("set")
        if "String" in atoms_str:
            structures.add("string")
        
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
                applicability_scope="broad" if compatibility > 0.8 else "medium"
            )
            
        except Exception as e:
            print(f"      Failed to create adaptation candidate {source_structure}→{target_structure}: {e}")
            return None
    
    def _generate_generic_adaptations(self, context: GenerationContext) -> List[DonorCandidate]:
        """Generate generic adaptations that work with multiple data structures."""
        candidates = []
        
        # Generic iterable adaptation
        generic_iterable_candidate = self._create_generic_iterable_candidate(context)
        if generic_iterable_candidate:
            candidates.append(generic_iterable_candidate)
        
        # Generic mapping adaptation
        generic_mapping_candidate = self._create_generic_mapping_candidate(context)
        if generic_mapping_candidate:
            candidates.append(generic_mapping_candidate)
        
        # Generic collection adaptation
        generic_collection_candidate = self._create_generic_collection_candidate(context)
        if generic_collection_candidate:
            candidates.append(generic_collection_candidate)
        
        return candidates
    
    def _generate_protocol_adaptations(self, context: GenerationContext) -> List[DonorCandidate]:
        """Generate adaptations based on Python protocols (duck typing)."""
        candidates = []
        
        # Sized protocol adaptation
        if self._uses_len_operation(context):
            sized_candidate = self._create_sized_protocol_candidate(context)
            if sized_candidate:
                candidates.append(sized_candidate)
        
        # Container protocol adaptation
        if self._uses_containment_check(context):
            container_candidate = self._create_container_protocol_candidate(context)
            if container_candidate:
                candidates.append(container_candidate)
        
        return candidates
    
    # Specific adaptation methods
    
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
        
        # Add conversion logic for operations that require ordering
        if "for i in range(" in adapted_code:
            adapted_code = adapted_code.replace(
                "for i in range(len(",
                "for i, item in enumerate(sorted("
            )
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "list", "set")
        
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
        
        # Handle indexing - use .get() for safety
        adapted_code = re.sub(
            r'(\w+)\[(\w+)\](?!\s*=)',  # Not followed by assignment
            r'\1.get(\2)',
            adapted_code
        )
        
        # Handle length operations
        adapted_code = adapted_code.replace("len(", "len(list(")
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "list", "dict")
        
        return adapted_code
    
    def _adapt_dict_to_list(self, code: str, func_name: str) -> str:
        """Adapt dict-based code to work with lists."""
        adapted_code = code
        new_func_name = f"{func_name}_list_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Replace dict operations with list operations
        adapted_code = adapted_code.replace("{}", "[]")
        
        # Handle dict methods
        adapted_code = adapted_code.replace(".keys()", "range(len(data))")
        adapted_code = adapted_code.replace(".values()", "data")
        adapted_code = adapted_code.replace(".items()", "enumerate(data)")
        
        # Handle dict access patterns
        adapted_code = re.sub(
            r'(\w+)\[(["\']?)(\w+)\2\]',
            r'\1[\3] if isinstance(\3, int) else \1[hash(\3) % len(\1)]',
            adapted_code
        )
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "dict", "list")
        
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
        
        # Handle set-specific operations
        adapted_code = adapted_code.replace(".union(", " + ")
        adapted_code = adapted_code.replace(".intersection(", 
                                          "list(set().intersection(")
        
        # Add uniqueness preservation where needed
        if ".add(" in code or "set(" in code:
            adapted_code = self._add_uniqueness_preservation(adapted_code)
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "set", "list")
        
        return adapted_code
    
    def _adapt_string_to_list(self, code: str, func_name: str) -> str:
        """Adapt string-based code to work with lists."""
        adapted_code = code
        new_func_name = f"{func_name}_list_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Handle string operations
        adapted_code = adapted_code.replace(".split()", "")
        adapted_code = adapted_code.replace('""', "[]")
        adapted_code = adapted_code.replace("''", "[]")
        
        # Handle string methods
        adapted_code = adapted_code.replace(".join(", "'.'.join(map(str, ")
        adapted_code = adapted_code.replace(".strip()", ".remove('')")
        adapted_code = adapted_code.replace(".lower()", "")
        adapted_code = adapted_code.replace(".upper()", "")
        
        # Add character-to-element conversion logic
        if "for char in" in adapted_code:
            adapted_code = adapted_code.replace("for char in", "for item in")
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "string", "list")
        
        return adapted_code
    
    def _adapt_list_to_tuple(self, code: str, func_name: str) -> str:
        """Adapt list-based code to work with tuples."""
        adapted_code = code
        new_func_name = f"{func_name}_tuple_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Replace mutable operations with immutable alternatives
        adapted_code = adapted_code.replace("[]", "()")
        adapted_code = adapted_code.replace(".append(", " + (")
        adapted_code = adapted_code.replace(".extend(", " + ")
        
        # Handle operations that modify lists
        if ".append(" in code or ".extend(" in code:
            adapted_code = self._convert_to_immutable_operations(adapted_code)
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "list", "tuple")
        
        return adapted_code
    
    def _adapt_list_to_generator(self, code: str, func_name: str) -> str:
        """Adapt list-based code to work with generators."""
        adapted_code = code
        new_func_name = f"{func_name}_generator_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Convert list operations to generator expressions
        adapted_code = re.sub(
            r'\[([^]]+)\s+for\s+([^]]+)\s+in\s+([^]]+)\]',
            r'(\1 for \2 in \3)',
            adapted_code
        )
        
        # Replace return statements with yield
        if "return [" in adapted_code:
            adapted_code = adapted_code.replace("return [", "yield from [")
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "list", "generator")
        
        return adapted_code
    
    def _adapt_list_to_numpy(self, code: str, func_name: str) -> str:
        """Adapt list-based code to work with NumPy arrays."""
        adapted_code = code
        new_func_name = f"{func_name}_numpy_adapted"
        
        # Add numpy import
        adapted_code = "import numpy as np\n\n" + adapted_code
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Replace list operations with numpy operations
        adapted_code = adapted_code.replace("[]", "np.array([])")
        adapted_code = adapted_code.replace(".append(", "np.append(")
        adapted_code = adapted_code.replace("sum(", "np.sum(")
        adapted_code = adapted_code.replace("max(", "np.max(")
        adapted_code = adapted_code.replace("min(", "np.min(")
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "list", "numpy_array")
        
        return adapted_code
    
    def _adapt_dict_to_namedtuple(self, code: str, func_name: str) -> str:
        """Adapt dict-based code to work with namedtuples."""
        adapted_code = code
        new_func_name = f"{func_name}_namedtuple_adapted"
        
        # Add namedtuple import
        adapted_code = "from collections import namedtuple\n\n" + adapted_code
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Add namedtuple definition (this would need to be context-specific)
        namedtuple_def = "    DataRecord = namedtuple('DataRecord', ['field1', 'field2', 'field3'])\n"
        
        # Insert namedtuple definition after function definition
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 2, namedtuple_def)
                break
        
        adapted_code = '\n'.join(lines)
        
        # Replace dict access with attribute access
        adapted_code = re.sub(r'(\w+)\[(["\'])(\w+)\2\]', r'\1.\3', adapted_code)
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "dict", "namedtuple")
        
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
        adapted_code = adapted_code.replace(".discard(", ".difference({")
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "set", "frozenset")
        
        return adapted_code
    
    # Generic adaptation methods
    
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
                applicability_scope="broad"
            )
        except Exception as e:
            print(f"      Failed to create generic iterable candidate: {e}")
            return None
    
    def _create_generic_mapping_candidate(self, context: GenerationContext) -> Optional[DonorCandidate]:
        """Create a candidate that works with any mapping."""
        if not self._uses_mapping_operations(context):
            return None
        
        try:
            adapted_code = self._create_generic_mapping_code(context.original_code, context.function_name)
            
            return DonorCandidate(
                name=f"{context.function_name}_mapping_generic",
                description="Generic adaptation to work with any mapping",
                code=adapted_code,
                strategy="data_structure_adaptation",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=["mapping"],
                operations_used=self._get_operations_from_context(context),
                metta_derivation=[f"(generic-mapping-adaptation {context.function_name})"],
                confidence=0.8,
                properties=["generic", "mapping-compatible"],
                complexity_estimate="same",
                applicability_scope="broad"
            )
        except Exception as e:
            print(f"      Failed to create generic mapping candidate: {e}")
            return None
    
    def _create_generic_collection_candidate(self, context: GenerationContext) -> Optional[DonorCandidate]:
        """Create a candidate that works with any collection."""
        try:
            adapted_code = self._create_generic_collection_code(context.original_code, context.function_name)
            
            return DonorCandidate(
                name=f"{context.function_name}_collection_generic",
                description="Generic adaptation to work with any collection",
                code=adapted_code,
                strategy="data_structure_adaptation",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=["collection"],
                operations_used=self._get_operations_from_context(context),
                metta_derivation=[f"(generic-collection-adaptation {context.function_name})"],
                confidence=0.75,
                properties=["generic", "collection-compatible"],
                complexity_estimate="slightly-higher",
                applicability_scope="broad"
            )
        except Exception as e:
            print(f"      Failed to create generic collection candidate: {e}")
            return None
    
    # Helper methods
    
    def _create_generic_iterable_code(self, code: str, func_name: str) -> str:
        """Create code that works with any iterable."""
        new_func_name = f"{func_name}_iterable_generic"
        adapted_code = code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Add type hints
        adapted_code = "from typing import Iterable, TypeVar, Any\n\n" + adapted_code
        adapted_code = adapted_code.replace("def ", "def ")
        
        # Replace specific operations with generic ones
        adapted_code = re.sub(r'(\w+)\[(\d+)\]', r'list(\1)[\2]', adapted_code)
        adapted_code = adapted_code.replace("len(", "sum(1 for _ in ")
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "specific", "iterable")
        
        return adapted_code
    
    def _create_generic_mapping_code(self, code: str, func_name: str) -> str:
        """Create code that works with any mapping."""
        new_func_name = f"{func_name}_mapping_generic"
        adapted_code = code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Add type hints
        adapted_code = "from typing import Mapping, Any\n\n" + adapted_code
        
        # Use generic mapping operations
        adapted_code = re.sub(r'(\w+)\[(["\']?)(\w+)\2\]', r'\1.get(\3)', adapted_code)
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "dict", "mapping")
        
        return adapted_code
    
    def _create_generic_collection_code(self, code: str, func_name: str) -> str:
        """Create code that works with any collection."""
        new_func_name = f"{func_name}_collection_generic"
        adapted_code = code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Add type hints
        adapted_code = "from typing import Collection, Any\n\n" + adapted_code
        
        # Convert to list for uniform handling
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 2, "    # Convert to list for uniform handling")
                lines.insert(i + 3, "    if not isinstance(data, list):")
                lines.insert(i + 4, "        data = list(data)")
                break
        
        adapted_code = '\n'.join(lines)
        
        # Add docstring
        adapted_code = self._add_adaptation_docstring(adapted_code, "specific", "collection")
        
        return adapted_code
    
    def _uses_mapping_operations(self, context: GenerationContext) -> bool:
        """Check if the function uses mapping operations."""
        code = context.original_code
        return any(op in code for op in [".keys()", ".values()", ".items()", ".get(", "["])
    
    def _uses_len_operation(self, context: GenerationContext) -> bool:
        """Check if the function uses len() operation."""
        return "len(" in context.original_code
    
    def _uses_containment_check(self, context: GenerationContext) -> bool:
        """Check if the function uses containment checks."""
        return " in " in context.original_code
    
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
    
    def _add_uniqueness_preservation(self, code: str) -> str:
        """Add logic to preserve uniqueness when converting from set to list."""
        lines = code.split('\n')
        
        # Add uniqueness check in append operations
        for i, line in enumerate(lines):
            if ".append(" in line:
                indent = len(line) - len(line.lstrip())
                check_line = " " * indent + "if item not in result:"
                lines[i] = check_line + "\n" + line
        
        return '\n'.join(lines)
    
    def _convert_to_immutable_operations(self, code: str) -> str:
        """Convert mutable operations to immutable ones for tuple adaptation."""
        # This is a simplified conversion - in practice would need more sophisticated handling
        adapted_code = code
        
        # Replace in-place modifications with tuple concatenation
        adapted_code = re.sub(
            r'(\w+)\.append\(([^)]+)\)',
            r'\1 = \1 + (\2,)',
            adapted_code
        )
        
        adapted_code = re.sub(
            r'(\w+)\.extend\(([^)]+)\)',
            r'\1 = \1 + tuple(\2)',
            adapted_code
        )
        
        return adapted_code
    
    def _create_sized_protocol_candidate(self, context: GenerationContext) -> Optional[DonorCandidate]:
        """Create a candidate using the Sized protocol."""
        try:
            adapted_code = "from typing import Sized\n\n" + context.original_code
            new_func_name = f"{context.function_name}_sized_protocol"
            adapted_code = adapted_code.replace(f"def {context.function_name}(", f"def {new_func_name}(")
            
            # Add type annotation for sized objects
            adapted_code = re.sub(
                r'def (\w+)\(([^)]+)\):',
                r'def \1(\2: Sized):',
                adapted_code
            )
            
            return DonorCandidate(
                name=new_func_name,
                description="Protocol-based adaptation using Sized",
                code=adapted_code,
                strategy="data_structure_adaptation",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=["sized"],
                operations_used=self._get_operations_from_context(context),
                metta_derivation=[f"(sized-protocol-adaptation {context.function_name})"],
                confidence=0.8,
                properties=["protocol-based", "sized-compatible"],
                complexity_estimate="same",
                applicability_scope="broad"
            )
        except Exception as e:
            print(f"      Failed to create sized protocol candidate: {e}")
            return None
    
    def _create_container_protocol_candidate(self, context: GenerationContext) -> Optional[DonorCandidate]:
        """Create a candidate using the Container protocol."""
        try:
            adapted_code = "from typing import Container\n\n" + context.original_code
            new_func_name = f"{context.function_name}_container_protocol"
            adapted_code = adapted_code.replace(f"def {context.function_name}(", f"def {new_func_name}(")
            
            return DonorCandidate(
                name=new_func_name,
                description="Protocol-based adaptation using Container",
                code=adapted_code,
                strategy="data_structure_adaptation",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=["container"],
                operations_used=self._get_operations_from_context(context),
                metta_derivation=[f"(container-protocol-adaptation {context.function_name})"],
                confidence=0.8,
                properties=["protocol-based", "container-compatible"],
                complexity_estimate="same",
                applicability_scope="broad"
            )
        except Exception as e:
            print(f"      Failed to create container protocol candidate: {e}")
            return None
    
    def _get_primary_pattern_family(self, context: GenerationContext) -> str:
        """Get the primary pattern family from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].pattern_family
        return "generic"
    
    def _get_operations_from_context(self, context: GenerationContext) -> List[str]:
        """Get operations from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].operations
        return ["adaptation"]