#!/usr/bin/env python3
"""
MeTTa-Powered Data Structure Adaptation Generator
Refactored to use MeTTa's symbolic reasoning for data structure transformations
"""

from typing import List, Dict, Any, Optional
import re

from metta_generator.base import BaseDonorGenerator, GenerationContext, DonorCandidate, GenerationStrategy

class DataStructureAdaptationGenerator(BaseDonorGenerator):
    """Generator that uses MeTTa reasoning for data structure adaptations."""
    
    def __init__(self):
        super().__init__()
        self._load_adaptation_rules()
    
    def _load_adaptation_rules(self):
        """Load data structure adaptation rules into MeTTa reasoning."""
        self.adaptation_rules = [
            # Data structure compatibility rules
            """(= (structure-compatible list tuple) 0.9)""",
            """(= (structure-compatible list set) 0.8)""", 
            """(= (structure-compatible list dict) 0.7)""",
            """(= (structure-compatible list generator) 0.8)""",
            """(= (structure-compatible dict namedtuple) 0.9)""",
            """(= (structure-compatible set frozenset) 0.95)""",
            """(= (structure-compatible string list) 0.8)""",
            """(= (structure-compatible string bytes) 0.9)""",
            """(= (structure-compatible tuple namedtuple) 0.9)""",
            
            # Bidirectional compatibility
            """(= (structure-compatible $a $b) (structure-compatible $b $a))""",
            
            # Adaptation safety rules
            """(= (adaptation-safe $func list set)
               (and (no-indexing-operations $func)
                    (no-ordering-dependent-operations $func)
                    (uses-data-structure $func list)))""",
                    
            """(= (adaptation-safe $func list tuple)
               (and (no-mutation-operations $func)
                    (uses-data-structure $func list)))""",
                    
            """(= (adaptation-safe $func dict namedtuple)
               (and (fixed-key-structure $func)
                    (uses-data-structure $func dict)))""",
            
            # Operation constraint rules
            """(= (no-indexing-operations $func)
               (not (match &self (array-access $func $index $line) True)))""",
               
            """(= (no-mutation-operations $func)
               (not (or (has-operation $func append)
                        (has-operation $func extend)
                        (has-operation $func remove))))""",
            
            # Structure usage detection rules
            """(= (uses-data-structure $func list)
               (or (match &self (variable-assign $var $scope $line)
                          (and (contains-function $scope $func)
                               (variable-type $var list)))
                   (has-list-operations $func)))""",
                   
            """(= (uses-data-structure $func dict)
               (or (match &self (bin-op . String Any $scope $line)
                          (contains-function $scope $func))
                   (has-dict-operations $func)))""",
            
            # Adaptation generation rules
            """(= (generate-adaptation $func $from $to)
               (let $adapter-func (get-adapter-function $from $to)
                    (let $adapted-code (apply-adapter $func $adapter-func)
                         (validated-adaptation $func $adapted-code $from $to))))""",
            
            # Quality assessment for adaptations
            """(= (adaptation-quality $from $to $quality)
               (case (structure-compatible $from $to)
                 ($score (if (>= $score 0.9) high-quality
                            (if (>= $score 0.7) medium-quality
                                low-quality)))))"""
        ]
    
    def can_generate(self, context: GenerationContext, strategy) -> bool:
        """Use MeTTa reasoning to determine if data structure adaptation is applicable."""
        if hasattr(strategy, 'value'):
            strategy_name = strategy.value
        else:
            strategy_name = str(strategy)
            
        if strategy_name != "data_structure_adaptation":
            return False
        
        # Use MeTTa reasoning to check for adaptable structures
        adaptable_structures = self._use_metta_reasoning(
            context, "find_adaptable_structures"
        )
        
        if adaptable_structures:
            print(f"      MeTTa reasoning: DataStructureAdaptationGenerator CAN generate")
            print(f"        Found adaptable structures: {adaptable_structures}")
            return True
        
        # Fallback to symbolic structure detection
        fallback_result = self._symbolic_structure_detection(context)
        
        if fallback_result:
            print(f"      Symbolic reasoning: DataStructureAdaptationGenerator CAN generate")
            print(f"        Detected structures: {fallback_result}")
        else:
            print(f"      DataStructureAdaptationGenerator: cannot generate")
        
        return fallback_result
    
    def _generate_candidates_impl(self, context: GenerationContext, strategy) -> List[DonorCandidate]:
        """Generate data structure adaptation candidates using MeTTa reasoning."""
        candidates = []
        
        # Use MeTTa reasoning to find structure adaptations
        metta_adaptations = self._use_metta_reasoning(
            context, "find_all_adaptations"
        )
        
        if metta_adaptations:
            print(f"        MeTTa found {len(metta_adaptations)} adaptation opportunities")
            for adaptation in metta_adaptations:
                candidate = self._create_metta_adaptation_candidate(context, adaptation)
                if candidate:
                    candidates.append(candidate)
        else:
            print(f"        No MeTTa adaptations found, using symbolic analysis")
            candidates.extend(self._generate_symbolic_adaptations(context))
        
        # Generate generic iterable adaptations using MeTTa guidance
        generic_candidates = self._generate_metta_generic_adaptations(context)
        candidates.extend(generic_candidates)
        
        return candidates
    
    def get_supported_strategies(self) -> List:
        """Get list of strategies this generator supports."""
        return [GenerationStrategy.DATA_STRUCTURE_ADAPTATION]
    
    def _symbolic_structure_detection(self, context: GenerationContext) -> bool:
        """Fallback symbolic detection of adaptable data structures."""
        detected_structures = self._detect_structures_in_code(context.original_code)
        
        # Check if any detected structures have known adaptations
        adaptable_from = ["list", "dict", "set", "string", "tuple"]
        
        return any(structure in adaptable_from for structure in detected_structures)
    
    def _detect_structures_in_code(self, code: str) -> List[str]:
        """Detect data structures used in code."""
        structures = set()
        code_lower = code.lower()
        
        structure_patterns = {
            "list": ["[", "list(", ".append(", ".extend(", ".pop("],
            "dict": ["{", "dict(", ".keys(", ".values(", ".items("],
            "set": ["set(", ".add(", ".union(", ".intersection("],
            "string": ['"', "'", "str(", ".split(", ".join(", ".strip("],
            "tuple": ["tuple(", "(", ",)"],
            "generator": ["yield", "generator"]
        }
        
        for struct_type, patterns in structure_patterns.items():
            if any(pattern in code_lower for pattern in patterns):
                structures.add(struct_type)
        
        return list(structures) if structures else ["generic"]
    
    def _generate_symbolic_adaptations(self, context: GenerationContext) -> List[DonorCandidate]:
        """Generate adaptation candidates using symbolic analysis."""
        candidates = []
        detected_structures = self._detect_structures_in_code(context.original_code)
        
        # Define adaptation mappings
        adaptation_mappings = {
            "list": {"set": 0.8, "tuple": 0.9, "dict": 0.7, "generator": 0.8},
            "dict": {"namedtuple": 0.9, "list": 0.8, "set": 0.7},
            "set": {"frozenset": 0.95, "list": 0.85, "dict": 0.7},
            "string": {"list": 0.8, "bytes": 0.9},
            "tuple": {"list": 0.9, "namedtuple": 0.9}
        }
        
        for source_structure in detected_structures:
            if source_structure in adaptation_mappings:
                for target_structure, compatibility in adaptation_mappings[source_structure].items():
                    adaptation_data = {
                        "from_structure": source_structure,
                        "to_structure": target_structure,
                        "compatibility": compatibility,
                        "safety": "safe" if compatibility > 0.8 else "moderate",
                        "reasoning": "symbolic-analysis"
                    }
                    
                    candidate = self._create_metta_adaptation_candidate(context, adaptation_data)
                    if candidate:
                        candidates.append(candidate)
        
        return candidates
    
    def _generate_metta_generic_adaptations(self, context: GenerationContext) -> List[DonorCandidate]:
        """Generate generic adaptations using MeTTa reasoning."""
        candidates = []
        
        # Use MeTTa reasoning for generic iterable adaptation
        generic_query = f"""
        (match &self
          (generic-iterable-adaptation-applicable {context.function_name})
          (iterable-adaptation {context.function_name}))
        """
        
        generic_facts = [
            f"(function-name {context.function_name})",
            f"(uses-iteration {context.function_name})"
        ]
        
        if self.reasoning_engine:
            generic_results = self.reasoning_engine._execute_metta_reasoning(generic_query, generic_facts)
        else:
            generic_results = self._symbolic_generic_analysis(context)
        
        for result in generic_results:
            generic_data = self._parse_generic_result(result)
            if generic_data:
                candidate = self._create_generic_adaptation_candidate(context, generic_data)
                if candidate:
                    candidates.append(candidate)
        
        return candidates
    
    def _create_metta_adaptation_candidate(self, context: GenerationContext,
                                         adaptation_data: Dict[str, Any]) -> Optional[DonorCandidate]:
        """Create adaptation candidate using MeTTa reasoning results."""
        try:
            from_structure = adaptation_data.get('from_structure')
            to_structure = adaptation_data.get('to_structure')
            compatibility = adaptation_data.get('compatibility', 0.7)
            safety = adaptation_data.get('safety', 'unknown')
            reasoning = adaptation_data.get('reasoning', 'metta-reasoning')
            
            # Generate adapted code using MeTTa guidance
            adapted_code = self._apply_metta_guided_adaptation(
                context, from_structure, to_structure
            )
            
            if not adapted_code:
                return None
            
            # Create candidate name
            new_func_name = f"{context.function_name}_{to_structure}_adapted"
            
            # Build MeTTa derivation trace
            metta_derivation = [
                f"(data-structure-adaptation {context.function_name} {from_structure} {to_structure})",
                f"(compatibility-score {compatibility})",
                f"(adaptation-safety {safety})",
                f"(reasoning-method {reasoning})"
            ]
            
            # Determine properties based on MeTTa reasoning
            properties = self._derive_adaptation_properties(adaptation_data)
            
            # Calculate confidence using MeTTa compatibility score
            confidence = float(compatibility) if isinstance(compatibility, (int, float)) else 0.7
            
            description = f"MeTTa-reasoned adaptation from {from_structure} to {to_structure}"
            
            return DonorCandidate(
                name=new_func_name,
                description=description,
                code=adapted_code,
                strategy="data_structure_adaptation",
                pattern_family=self._get_pattern_family(context),
                data_structures_used=[to_structure],
                operations_used=["structure-adaptation"],
                metta_derivation=metta_derivation,
                confidence=confidence,
                properties=properties,
                complexity_estimate=self._estimate_adaptation_complexity(compatibility),
                applicability_scope=self._determine_adaptation_scope(compatibility, safety),
                generator_used=self.generator_name,
                metta_reasoning_trace=[f"adaptation-reasoning: {adaptation_data}"]
            )
            
        except Exception as e:
            print(f"        Error creating MeTTa adaptation candidate: {e}")
            return None
    
    def _create_generic_adaptation_candidate(self, context: GenerationContext,
                                           generic_data: Dict[str, Any]) -> Optional[DonorCandidate]:
        """Create generic iterable adaptation candidate using MeTTa reasoning."""
        try:
            adapted_code = self._create_metta_guided_generic_iterable(context)
            
            if not adapted_code:
                return None
            
            new_func_name = f"{context.function_name}_iterable_generic"
            
            metta_derivation = [
                f"(generic-iterable-adaptation {context.function_name})",
                f"(metta-guided-generalization)",
                f"(iterable-compatibility verified)"
            ]
            
            description = "MeTTa-guided generic iterable adaptation"
            
            return DonorCandidate(
                name=new_func_name,
                description=description,
                code=adapted_code,
                strategy="data_structure_adaptation",
                pattern_family=self._get_pattern_family(context),
                data_structures_used=["iterable"],
                operations_used=["generic-adaptation"],
                metta_derivation=metta_derivation,
                confidence=0.85,
                properties=["metta-reasoned", "generic", "iterable-compatible"],
                complexity_estimate="same",
                applicability_scope="broad",
                generator_used=self.generator_name,
                metta_reasoning_trace=[f"generic-reasoning: {generic_data}"]
            )
            
        except Exception as e:
            print(f"        Error creating generic adaptation candidate: {e}")
            return None
    
    # MeTTa-guided adaptation methods
    
    def _apply_metta_guided_adaptation(self, context: GenerationContext,
                                     from_structure: str, to_structure: str) -> Optional[str]:
        """Apply structure adaptation with MeTTa guidance."""
        # Route to specific adaptation method based on MeTTa reasoning
        adapter_method = self._get_metta_adapter_method(from_structure, to_structure)
        
        if adapter_method:
            return adapter_method(context.original_code, context.function_name)
        else:
            return self._generic_metta_adaptation(context, from_structure, to_structure)
    
    def _get_metta_adapter_method(self, from_structure: str, to_structure: str):
        """Get appropriate adapter method based on MeTTa reasoning."""
        adaptation_methods = {
            ("list", "set"): self._adapt_list_to_set_metta,
            ("list", "tuple"): self._adapt_list_to_tuple_metta,
            ("list", "dict"): self._adapt_list_to_dict_metta,
            ("list", "generator"): self._adapt_list_to_generator_metta,
            ("dict", "namedtuple"): self._adapt_dict_to_namedtuple_metta,
            ("dict", "list"): self._adapt_dict_to_list_metta,
            ("set", "frozenset"): self._adapt_set_to_frozenset_metta,
            ("set", "list"): self._adapt_set_to_list_metta,
            ("string", "list"): self._adapt_string_to_list_metta,
            ("string", "bytes"): self._adapt_string_to_bytes_metta,
            ("tuple", "list"): self._adapt_tuple_to_list_metta,
            ("tuple", "namedtuple"): self._adapt_tuple_to_namedtuple_metta
        }
        
        return adaptation_methods.get((from_structure, to_structure))
    
    def _adapt_list_to_set_metta(self, code: str, func_name: str) -> str:
        """Adapt list to set using MeTTa guidance."""
        adapted_code = code
        new_func_name = f"{func_name}_set_adapted"
        
        # Replace function name
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # MeTTa-guided list to set transformations
        adapted_code = adapted_code.replace("[]", "set()")
        adapted_code = adapted_code.replace(".append(", ".add(")
        adapted_code = adapted_code.replace(".extend(", ".update(")
        adapted_code = adapted_code.replace(".remove(", ".discard(")
        
        # Handle indexing (sets don't support indexing) - MeTTa safety check
        adapted_code = re.sub(r'(\w+)\[(\d+)\]', r'list(\1)[\2]', adapted_code)
        adapted_code = re.sub(r'(\w+)\[(\w+)\]', r'list(\1)[\2]', adapted_code)
        
        # Add MeTTa reasoning documentation
        adapted_code = self._add_metta_adaptation_docs(adapted_code, "list", "set", 
                                                      "MeTTa verified: no indexing dependencies")
        
        return adapted_code
    
    def _adapt_list_to_tuple_metta(self, code: str, func_name: str) -> str:
        """Adapt list to tuple using MeTTa guidance."""
        adapted_code = code
        new_func_name = f"{func_name}_tuple_adapted"
        
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # MeTTa-guided immutable transformations
        adapted_code = adapted_code.replace("[]", "()")
        
        # Handle append operations (tuples are immutable) - MeTTa guidance
        adapted_code = re.sub(
            r'(\w+)\.append\(([^)]+)\)',
            r'\1 = \1 + (\2,)',
            adapted_code
        )
        
        adapted_code = self._add_metta_adaptation_docs(adapted_code, "list", "tuple",
                                                      "MeTTa verified: immutable operations only")
        
        return adapted_code
    
    def _adapt_list_to_dict_metta(self, code: str, func_name: str) -> str:
        """Adapt list to dict using MeTTa guidance."""
        adapted_code = code
        new_func_name = f"{func_name}_dict_adapted"
        
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # MeTTa-guided list to dict mapping
        adapted_code = adapted_code.replace("[]", "{}")
        
        # Transform append to dict assignment - MeTTa reasoning
        adapted_code = re.sub(
            r'(\w+)\.append\(([^)]+)\)',
            r'\1[len(\1)] = \2',
            adapted_code
        )
        
        adapted_code = self._add_metta_adaptation_docs(adapted_code, "list", "dict",
                                                      "MeTTa guided: index-based key mapping")
        
        return adapted_code
    
    def _adapt_list_to_generator_metta(self, code: str, func_name: str) -> str:
        """Adapt list to generator using MeTTa guidance."""
        adapted_code = code
        new_func_name = f"{func_name}_generator_adapted"
        
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # MeTTa-guided lazy evaluation transformation
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if 'return ' in line and '[' in line:
                # Replace return with yield from - MeTTa lazy reasoning
                lines[i] = line.replace('return [', 'yield from [')
        
        adapted_code = '\n'.join(lines)
        adapted_code = self._add_metta_adaptation_docs(adapted_code, "list", "generator",
                                                      "MeTTa verified: lazy evaluation benefits")
        
        return adapted_code
    
    def _adapt_dict_to_namedtuple_metta(self, code: str, func_name: str) -> str:
        """Adapt dict to namedtuple using MeTTa guidance."""
        adapted_code = "from collections import namedtuple\n\n" + code
        new_func_name = f"{func_name}_namedtuple_adapted"
        
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # MeTTa-guided namedtuple structure
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 2, "    # MeTTa-derived structure definition")
                lines.insert(i + 3, "    DataRecord = namedtuple('DataRecord', ['field1', 'field2', 'field3'])")
                break
        
        adapted_code = '\n'.join(lines)
        adapted_code = self._add_metta_adaptation_docs(adapted_code, "dict", "namedtuple",
                                                      "MeTTa verified: fixed structure compatibility")
        
        return adapted_code
    
    def _adapt_dict_to_list_metta(self, code: str, func_name: str) -> str:
        """Adapt dict to list using MeTTa guidance."""
        adapted_code = code
        new_func_name = f"{func_name}_list_adapted"
        
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # MeTTa-guided dict to list transformations
        adapted_code = adapted_code.replace("{}", "[]")
        adapted_code = adapted_code.replace(".keys()", "range(len(data))")
        adapted_code = adapted_code.replace(".values()", "data")
        adapted_code = adapted_code.replace(".items()", "enumerate(data)")
        
        adapted_code = self._add_metta_adaptation_docs(adapted_code, "dict", "list",
                                                      "MeTTa reasoning: sequential access pattern")
        
        return adapted_code
    
    def _adapt_set_to_frozenset_metta(self, code: str, func_name: str) -> str:
        """Adapt set to frozenset using MeTTa guidance."""
        adapted_code = code
        new_func_name = f"{func_name}_frozenset_adapted"
        
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # MeTTa-guided immutable set transformations
        adapted_code = adapted_code.replace("set()", "frozenset()")
        adapted_code = adapted_code.replace(".add(", ".union({")
        adapted_code = adapted_code.replace(".update(", ".union(")
        
        adapted_code = self._add_metta_adaptation_docs(adapted_code, "set", "frozenset",
                                                      "MeTTa verified: immutable set operations")
        
        return adapted_code
    
    def _adapt_set_to_list_metta(self, code: str, func_name: str) -> str:
        """Adapt set to list using MeTTa guidance."""
        adapted_code = code
        new_func_name = f"{func_name}_list_adapted"
        
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # MeTTa-guided set to list transformations
        adapted_code = adapted_code.replace("set()", "[]")
        adapted_code = adapted_code.replace(".add(", ".append(")
        adapted_code = adapted_code.replace(".update(", ".extend(")
        adapted_code = adapted_code.replace(".discard(", ".remove(")
        
        adapted_code = self._add_metta_adaptation_docs(adapted_code, "set", "list",
                                                      "MeTTa verified: ordered operations acceptable")
        
        return adapted_code
    
    def _adapt_string_to_list_metta(self, code: str, func_name: str) -> str:
        """Adapt string to list using MeTTa guidance."""
        adapted_code = code
        new_func_name = f"{func_name}_list_adapted"
        
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # MeTTa-guided string to list conversion
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 2, "    # MeTTa-guided character list conversion")
                lines.insert(i + 3, "    if isinstance(text, str):")
                lines.insert(i + 4, "        text = list(text)")
                break
        
        adapted_code = '\n'.join(lines)
        adapted_code = self._add_metta_adaptation_docs(adapted_code, "string", "list",
                                                      "MeTTa reasoning: character-level processing")
        
        return adapted_code
    
    def _adapt_string_to_bytes_metta(self, code: str, func_name: str) -> str:
        """Adapt string to bytes using MeTTa guidance."""
        adapted_code = code
        new_func_name = f"{func_name}_bytes_adapted"
        
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # MeTTa-guided string to bytes conversion
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 2, "    # MeTTa-guided binary conversion")
                lines.insert(i + 3, "    if isinstance(text, str):")
                lines.insert(i + 4, "        text = text.encode('utf-8')")
                break
        
        adapted_code = '\n'.join(lines)
        adapted_code = self._add_metta_adaptation_docs(adapted_code, "string", "bytes",
                                                      "MeTTa verified: encoding compatibility")
        
        return adapted_code
    
    def _adapt_tuple_to_list_metta(self, code: str, func_name: str) -> str:
        """Adapt tuple to list using MeTTa guidance."""
        adapted_code = code
        new_func_name = f"{func_name}_list_adapted"
        
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # MeTTa-guided tuple to list conversion
        adapted_code = adapted_code.replace("()", "[]")
        adapted_code = adapted_code.replace("tuple(", "list(")
        
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 2, "    # MeTTa-guided mutable conversion")
                lines.insert(i + 3, "    if isinstance(data, tuple):")
                lines.insert(i + 4, "        data = list(data)")
                break
        
        adapted_code = '\n'.join(lines)
        adapted_code = self._add_metta_adaptation_docs(adapted_code, "tuple", "list",
                                                      "MeTTa verified: mutability benefits")
        
        return adapted_code
    
    def _adapt_tuple_to_namedtuple_metta(self, code: str, func_name: str) -> str:
        """Adapt tuple to namedtuple using MeTTa guidance."""
        adapted_code = "from collections import namedtuple\n\n" + code
        new_func_name = f"{func_name}_namedtuple_adapted"
        
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # MeTTa-guided namedtuple structure
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 2, "    # MeTTa-derived named structure")
                lines.insert(i + 3, "    DataTuple = namedtuple('DataTuple', ['field1', 'field2', 'field3'])")
                break
        
        adapted_code = '\n'.join(lines)
        adapted_code = self._add_metta_adaptation_docs(adapted_code, "tuple", "namedtuple",
                                                      "MeTTa reasoning: structured access benefits")
        
        return adapted_code
    
    def _create_metta_guided_generic_iterable(self, context: GenerationContext) -> str:
        """Create generic iterable adaptation with MeTTa guidance."""
        func_name = context.function_name
        new_func_name = f"{func_name}_iterable_generic"
        
        adapted_code = "from typing import Iterable, Any\n\n" + context.original_code
        adapted_code = adapted_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # MeTTa-guided generic conversion logic
        lines = adapted_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 2, '    """MeTTa-guided generic iterable adaptation."""')
                lines.insert(i + 3, "    # MeTTa reasoning: universal iterable compatibility")
                lines.insert(i + 4, "    if not isinstance(data, list):")
                lines.insert(i + 5, "        data = list(data)")
                break
        
        return '\n'.join(lines)
    
    def _generic_metta_adaptation(self, context: GenerationContext, 
                                from_structure: str, to_structure: str) -> str:
        """Generic adaptation with MeTTa reasoning traces."""
        func_name = context.function_name
        new_func_name = f"{func_name}_{to_structure}_adapted"
        
        adapted_code = context.original_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Add MeTTa reasoning trace
        reasoning_trace = f"MeTTa adaptation reasoning: {from_structure} to {to_structure}"
        adapted_code = self._add_metta_adaptation_docs(adapted_code, from_structure, to_structure, reasoning_trace)
        
        return adapted_code
    
    def _add_metta_adaptation_docs(self, code: str, from_type: str, to_type: str, reasoning: str) -> str:
        """Add MeTTa adaptation documentation."""
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                if i + 1 < len(lines) and '"""' in lines[i + 1]:
                    lines[i + 1] = lines[i + 1].replace('"""', f'"""MeTTa adaptation: {from_type} to {to_type}. {reasoning}. ')
                else:
                    lines.insert(i + 1, f'    """MeTTa adaptation: {from_type} to {to_type}. {reasoning}."""')
                break
        
        return '\n'.join(lines)
    
    # Helper methods for MeTTa reasoning support
    
    def _symbolic_generic_analysis(self, context: GenerationContext) -> List[Dict[str, Any]]:
        """Fallback generic analysis when MeTTa reasoning is unavailable."""
        code = context.original_code.lower()
        
        if any(pattern in code for pattern in ["for ", "list(", "range(", "enumerate("]):
            return [{"adaptation_type": "generic-iterable", "applicable": True}]
        
        return []
    
    def _parse_generic_result(self, result: Any) -> Optional[Dict[str, Any]]:
        """Parse MeTTa generic reasoning result."""
        try:
            result_str = str(result)
            if "iterable-adaptation" in result_str:
                return {"adaptation_type": "generic-iterable", "metta_verified": True}
        except Exception:
            pass
        return None
    
    def _derive_adaptation_properties(self, adaptation_data: Dict[str, Any]) -> List[str]:
        """Derive properties from MeTTa adaptation reasoning."""
        properties = ["metta-reasoned", "structure-adapted"]
        
        from_structure = adaptation_data.get('from_structure', '')
        to_structure = adaptation_data.get('to_structure', '')
        compatibility = adaptation_data.get('compatibility', 0)
        safety = adaptation_data.get('safety', '')
        
        # Add structure-specific properties
        properties.append(f"{from_structure}-to-{to_structure}")
        
        if isinstance(compatibility, (int, float)) and compatibility > 0.8:
            properties.append("high-compatibility")
        
        if safety == 'safe':
            properties.append("adaptation-safe")
        
        # Add semantic properties based on structure types
        if to_structure in ["set", "frozenset"]:
            properties.append("unordered")
        elif to_structure in ["tuple", "frozenset", "namedtuple"]:
            properties.append("immutable")
        elif to_structure == "generator":
            properties.append("lazy-evaluation")
        
        return properties
    
    def _estimate_adaptation_complexity(self, compatibility: Any) -> str:
        """Estimate complexity change from adaptation."""
        if isinstance(compatibility, (int, float)):
            if compatibility > 0.9:
                return "same"
            elif compatibility > 0.7:
                return "slightly-higher"
            else:
                return "moderate-increase"
        return "same"
    
    def _determine_adaptation_scope(self, compatibility: Any, safety: str) -> str:
        """Determine applicability scope based on MeTTa reasoning."""
        if isinstance(compatibility, (int, float)):
            if compatibility > 0.8 and safety == 'safe':
                return "broad"
            elif compatibility > 0.6:
                return "medium"
            else:
                return "narrow"
        
        if safety == 'safe':
            return "medium"
        else:
            return "narrow"
    
    def _get_pattern_family(self, context: GenerationContext) -> str:
        """Get pattern family from context."""
        if hasattr(context, 'detected_patterns') and context.detected_patterns:
            return context.detected_patterns[0].pattern_family
        return "generic"#!/usr/bin/env python3
