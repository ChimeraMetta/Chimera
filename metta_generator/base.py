#!/usr/bin/env python3
"""
Modular MeTTa Donor Generator Architecture
Organized into separate modules for better maintainability and extensibility.
"""

# ==============================================================================
# Core Module
# ==============================================================================

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import ast
import inspect

DONOR_GENERATION_ONTOLOGY = "metta/donor_generation.metta"

@dataclass
class GenerationContext:
    """Context information for donor generation."""
    function_name: str
    original_code: str
    analysis_result: Dict[str, Any]
    metta_atoms: List[str]
    detected_patterns: List['FunctionPattern']
    metta_space: Any

@dataclass
class FunctionPattern:
    """Represents a detected function pattern."""
    pattern_family: str
    pattern_type: str
    data_structures: List[str]
    operations: List[str]
    control_flow: List[str]
    metta_evidence: List[str]
    confidence: float
    properties: List[str]

@dataclass
class DonorCandidate:
    """A generated donor candidate."""
    name: str
    description: str
    code: str
    strategy: str
    pattern_family: str
    data_structures_used: List[str]
    operations_used: List[str]
    metta_derivation: List[str]
    confidence: float
    properties: List[str]
    complexity_estimate: str
    applicability_scope: str

class GenerationStrategy(Enum):
    """Available generation strategies."""
    OPERATION_SUBSTITUTION = "operation_substitution"
    ACCUMULATOR_VARIATION = "accumulator_variation"
    CONDITION_VARIATION = "condition_variation"
    STRUCTURE_PRESERVATION = "structure_preservation"
    PROPERTY_GUIDED = "property_guided"
    PATTERN_EXPANSION = "pattern_expansion"
    DATA_STRUCTURE_ADAPTATION = "data_structure_adaptation"
    ALGORITHM_TRANSFORMATION = "algorithm_transformation"
    CONTROL_FLOW_VARIATION = "control_flow_variation"
    TYPE_GENERALIZATION = "type_generalization"
    FUNCTIONAL_COMPOSITION = "functional_composition"
    ERROR_HANDLING_ENHANCEMENT = "error_handling_enhancement"

class BaseDonorGenerator(ABC):
    """Abstract base class for donor generators."""
    
    @abstractmethod
    def can_generate(self, context: GenerationContext, strategy: GenerationStrategy) -> bool:
        """Check if this generator can handle the given context and strategy."""
        pass
    
    @abstractmethod
    def generate_candidates(self, context: GenerationContext, strategy: GenerationStrategy) -> List[DonorCandidate]:
        """Generate donor candidates for the given context and strategy."""
        pass
    
    @abstractmethod
    def get_supported_strategies(self) -> List[GenerationStrategy]:
        """Get list of strategies this generator supports."""
        pass

# ==============================================================================
# Pattern Detection Module
# ==============================================================================

class PatternDetector:
    """Detects patterns in functions using MeTTa reasoning."""
    
    def __init__(self, metta_space):
        self.metta_space = metta_space
        self.pattern_families = {
            "search": ["linear_search", "binary_search", "find_first", "find_all", "contains"],
            "transform": ["map", "filter", "reduce", "fold", "scan", "zip"],
            "aggregate": ["sum", "count", "average", "min", "max", "group_by"],
            "validate": ["type_check", "bounds_check", "format_check", "constraint_check"],
            "string_ops": ["parse", "format", "split", "join", "replace", "match"],
            "math_ops": ["arithmetic", "statistical", "geometric"],
            "control": ["branch", "loop", "recursion", "exception_handling"],
        }
    
    def detect_patterns(self, context: GenerationContext) -> List[FunctionPattern]:
        """Detect all patterns in the given function context."""
        patterns = []
        
        # Detect data structures, operations, and control flow
        data_structures = self._detect_data_structures(context)
        operations = self._detect_operations(context)
        control_flow = self._detect_control_flow(context)
        
        print(f"     Detected: structures={data_structures}, ops={operations}, flow={control_flow}")
        
        # Check each pattern family
        for family_name, family_patterns in self.pattern_families.items():
            if self._has_pattern_characteristics(context, family_name):
                pattern = FunctionPattern(
                    pattern_family=family_name,
                    pattern_type=self._classify_pattern_type(context, family_name),
                    data_structures=data_structures,
                    operations=operations,
                    control_flow=control_flow,
                    metta_evidence=[f"({family_name.replace('_', '-')}-pattern {context.function_name})"],
                    confidence=self._calculate_pattern_confidence(context, family_name),
                    properties=self._get_pattern_properties(family_name)
                )
                patterns.append(pattern)
        
        # Add generic pattern if no specific patterns found
        if not patterns:
            patterns.append(self._create_generic_pattern(context, data_structures, operations, control_flow))
        
        return patterns
    
    def _detect_data_structures(self, context: GenerationContext) -> List[str]:
        """Detect data structures used in the function."""
        structures = []
        code = context.original_code.lower()
        atoms_str = str(context.metta_space)
        
        structure_indicators = {
            "list": ["list", "append", "extend", "[", "]"],
            "dict": ["dict", ".keys()", ".values()", ".items()", "{"],
            "set": ["set", ".add(", ".union(", ".intersection("],
            "string": ["str", ".split()", ".join()", ".replace()", "String"],
            "tuple": ["tuple", "(", ")"],
            "numeric": ["int", "float", "Number", "sum", "max", "min"]
        }
        
        for struct_type, indicators in structure_indicators.items():
            if any(indicator in code or indicator in atoms_str for indicator in indicators):
                structures.append(struct_type)
        
        return structures if structures else ["generic"]
    
    def _detect_operations(self, context: GenerationContext) -> List[str]:
        """Detect types of operations in the function."""
        operations = []
        atoms_str = str(context.metta_space)
        code = context.original_code
        
        if "bin-op" in atoms_str and any(op in atoms_str for op in ["Lt", "Gt", "Eq"]):
            operations.append("comparison")
        
        if any(op in atoms_str for op in ["Add", "Sub", "Mult", "Div"]):
            operations.append("arithmetic")
        
        if "string-op-pattern" in atoms_str:
            operations.append("string_manipulation")
        
        if any(op in code for op in ["and", "or", "not"]):
            operations.append("logical")
        
        if "variable-assign" in atoms_str:
            operations.append("assignment")
        
        if "function-call" in atoms_str:
            operations.append("function_call")
        
        return operations if operations else ["generic"]
    
    def _detect_control_flow(self, context: GenerationContext) -> List[str]:
        """Detect control flow patterns."""
        control_flow = []
        code = context.original_code
        atoms_str = str(context.metta_space)
        
        if "loop-pattern" in atoms_str or any(kw in code for kw in ["for ", "while "]):
            control_flow.append("loop")
        
        if "if " in code:
            control_flow.append("conditional")
        
        if context.function_name in code.replace(f"def {context.function_name}", ""):
            control_flow.append("recursion")
        
        if any(kw in code for kw in ["try:", "except:", "finally:", "raise"]):
            control_flow.append("exception_handling")
        
        if "return" in code:
            control_flow.append("return")
        
        return control_flow if control_flow else ["sequential"]
    
    def _has_pattern_characteristics(self, context: GenerationContext, family: str) -> bool:
        """Check if function has characteristics of a pattern family."""
        code = context.original_code.lower()
        func_name = context.function_name.lower()
        
        family_keywords = {
            "search": ["find", "search", "locate", "get", "contains", "index"],
            "transform": ["transform", "convert", "map", "filter", "process", "apply"],
            "aggregate": ["sum", "count", "average", "total", "aggregate", "reduce", "accumulate"],
            "validate": ["validate", "check", "verify", "is_", "has_", "test"],
            "string_ops": ["parse", "format", "clean", "normalize", "extract", "split", "join"],
            "math_ops": ["calculate", "compute", "math", "formula", "equation"],
            "control": ["control", "manage", "handle", "process"]
        }
        
        keywords = family_keywords.get(family, [])
        name_match = any(keyword in func_name for keyword in keywords)
        
        # Structural checks
        if family == "search":
            return name_match or (self._has_loop_and_comparison(context) and "return" in code)
        elif family == "transform":
            return name_match or (self._has_loop_and_assignment(context))
        elif family == "aggregate":
            return name_match or (self._has_accumulator_pattern(context))
        elif family == "validate":
            return name_match or (self._has_conditional_and_boolean_return(context))
        elif family == "string_ops":
            return name_match or self._has_string_operations(context)
        elif family == "math_ops":
            return name_match or self._has_math_operations(context)
        
        return name_match
    
    def _has_loop_and_comparison(self, context: GenerationContext) -> bool:
        """Check for loop + comparison pattern."""
        return ("loop-pattern" in str(context.metta_space) and 
                "bin-op" in str(context.metta_space))
    
    def _has_loop_and_assignment(self, context: GenerationContext) -> bool:
        """Check for loop + assignment pattern."""
        return ("loop-pattern" in str(context.metta_space) and 
                "variable-assign" in str(context.metta_space))
    
    def _has_accumulator_pattern(self, context: GenerationContext) -> bool:
        """Check for accumulator pattern."""
        code = context.original_code
        return (self._has_loop_and_assignment(context) and 
                any(op in code for op in ["+=", "*=", "sum", "count", "total"]))
    
    def _has_conditional_and_boolean_return(self, context: GenerationContext) -> bool:
        """Check for conditional + boolean return pattern."""
        code = context.original_code
        return ("if " in code and 
                any(val in code for val in ["True", "False", "return True", "return False"]))
    
    def _has_string_operations(self, context: GenerationContext) -> bool:
        """Check for string operations."""
        code = context.original_code
        return (any(op in code for op in [".split()", ".join()", ".replace()", ".strip()"]) or
                "String" in str(context.metta_space))
    
    def _has_math_operations(self, context: GenerationContext) -> bool:
        """Check for mathematical operations."""
        code = context.original_code
        atoms_str = str(context.metta_space)
        return (any(op in code for op in ["math.", "**", "sqrt", "sin", "cos"]) or
                any(op in atoms_str for op in ["Add", "Sub", "Mult", "Div"]))
    
    def _classify_pattern_type(self, context: GenerationContext, family: str) -> str:
        """Classify the specific pattern type within a family."""
        func_name = context.function_name.lower()
        
        type_mappings = {
            "search": {
                "binary": "binary_search",
                "first": "find_first", 
                "all": "find_all",
                "index": "find_index",
                "default": "linear_search"
            },
            "transform": {
                "map": "map",
                "filter": "filter",
                "convert": "convert",
                "default": "generic_transform"
            },
            "aggregate": {
                "sum": "sum",
                "count": "count",
                "average": "average",
                "mean": "average",
                "max": "max",
                "min": "min",
                "default": "generic_aggregate"
            }
        }
        
        family_types = type_mappings.get(family, {})
        for keyword, pattern_type in family_types.items():
            if keyword != "default" and keyword in func_name:
                return pattern_type
        
        return family_types.get("default", f"generic_{family}")
    
    def _calculate_pattern_confidence(self, context: GenerationContext, family: str) -> float:
        """Calculate confidence for pattern detection."""
        base_confidence = 0.6
        func_name = context.function_name.lower()
        
        # Name-based confidence boost
        family_keywords = self.pattern_families.get(family, [])
        if any(keyword in func_name for keyword in family_keywords):
            base_confidence += 0.2
        
        # Structure-based confidence boost
        if self._has_pattern_characteristics(context, family):
            base_confidence += 0.1
        
        return min(0.95, base_confidence)
    
    def _get_pattern_properties(self, family: str) -> List[str]:
        """Get properties associated with a pattern family."""
        family_properties = {
            "search": ["deterministic", "side-effect-free"],
            "transform": ["functional", "composable"],
            "aggregate": ["associative", "fold-like"],
            "validate": ["predicate", "boolean-return"],
            "string_ops": ["text-processing", "parsing"],
            "math_ops": ["numerical", "computational"],
        }
        
        return family_properties.get(family, ["generic"])
    
    def _create_generic_pattern(self, context: GenerationContext, 
                              data_structures: List[str], 
                              operations: List[str], 
                              control_flow: List[str]) -> FunctionPattern:
        """Create a generic pattern when no specific patterns are detected."""
        return FunctionPattern(
            pattern_family="generic",
            pattern_type="general_function",
            data_structures=data_structures,
            operations=operations,
            control_flow=control_flow,
            metta_evidence=[f"(generic-pattern {context.function_name})"],
            confidence=0.5,
            properties=["generic", "adaptable"]
        )

# ==============================================================================
# Strategy Manager Module
# ==============================================================================

class StrategyManager:
    """Manages generation strategies and determines applicability."""
    
    def __init__(self, generators: List[BaseDonorGenerator]):
        self.generators = generators
        self.strategy_requirements = {
            GenerationStrategy.OPERATION_SUBSTITUTION: ["comparison", "arithmetic"],
            GenerationStrategy.DATA_STRUCTURE_ADAPTATION: ["list", "dict", "set"],
            GenerationStrategy.ALGORITHM_TRANSFORMATION: ["loop", "recursion"],
            GenerationStrategy.TYPE_GENERALIZATION: ["specific_types"],
            GenerationStrategy.FUNCTIONAL_COMPOSITION: ["function_call"],
            GenerationStrategy.ERROR_HANDLING_ENHANCEMENT: ["no_error_handling"],
            GenerationStrategy.CONTROL_FLOW_VARIATION: ["loop", "conditional"],
            GenerationStrategy.ACCUMULATOR_VARIATION: ["loop", "assignment"],
            GenerationStrategy.STRUCTURE_PRESERVATION: [],  # Always applicable
            GenerationStrategy.CONDITION_VARIATION: ["conditional"],
            GenerationStrategy.PROPERTY_GUIDED: ["pattern_detected"],
            GenerationStrategy.PATTERN_EXPANSION: ["loop"]
        }
    
    def get_applicable_strategies(self, context: GenerationContext, 
                                requested_strategies: Optional[List[GenerationStrategy]] = None) -> List[GenerationStrategy]:
        """All strategies are applicable - let generators decide."""
        if requested_strategies is None:
            # Return all strategies
            return list(GenerationStrategy)
        else:
            return requested_strategies
    
    def _is_strategy_applicable(self, context: GenerationContext, strategy: GenerationStrategy) -> bool:
        """Check if a strategy is applicable based on context."""
        requirements = self.strategy_requirements.get(strategy, [])
        
        if not requirements:  # No requirements means always applicable
            return True
        
        # Check each requirement
        for requirement in requirements:
            if not self._check_requirement(context, requirement):
                return False
        
        return True
    
    def _check_requirement(self, context: GenerationContext, requirement: str) -> bool:
        """Check if a specific requirement is met."""
        operations = self._get_operations_from_context(context)
        data_structures = self._get_data_structures_from_context(context)
        control_flow = self._get_control_flow_from_context(context)
        
        if requirement in operations:
            return True
        elif requirement in data_structures:
            return True
        elif requirement in control_flow:
            return True
        elif requirement == "specific_types":
            return any(struct in ["list", "dict", "string"] for struct in data_structures)
        elif requirement == "no_error_handling":
            return "exception_handling" not in control_flow
        elif requirement == "pattern_detected":
            return len(context.detected_patterns) > 0
        
        return False
    
    def _get_operations_from_context(self, context: GenerationContext) -> List[str]:
        """Extract operations from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].operations
        return []
    
    def _get_data_structures_from_context(self, context: GenerationContext) -> List[str]:
        """Extract data structures from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].data_structures
        return []
    
    def _get_control_flow_from_context(self, context: GenerationContext) -> List[str]:
        """Extract control flow from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].control_flow
        return []

# ==============================================================================
# Generator Registry Module
# ==============================================================================

class GeneratorRegistry:
    """Registry for managing different donor generators."""
    
    def __init__(self):
        self.generators = []
        self.strategy_manager = None
    
    def register_generator(self, generator: BaseDonorGenerator):
        """Register a new donor generator."""
        self.generators.append(generator)
        self._update_strategy_manager()
    
    def register_generators(self, generators: List[BaseDonorGenerator]):
        """Register multiple generators at once."""
        self.generators.extend(generators)
        self._update_strategy_manager()
    
    def _update_strategy_manager(self):
        """Update the strategy manager with current generators."""
        self.strategy_manager = StrategyManager(self.generators)
    
    def get_generators_for_strategy(self, strategy: GenerationStrategy) -> List[BaseDonorGenerator]:
        """Get all generators that can handle a specific strategy."""
        return [gen for gen in self.generators if strategy in gen.get_supported_strategies()]
    
    def generate_candidates(self, context: GenerationContext, 
                          strategies: Optional[List[GenerationStrategy]] = None) -> List[DonorCandidate]:
        """Generate candidates using all applicable generators."""
        if not self.strategy_manager:
            self._update_strategy_manager()
        
        applicable_strategies = self.strategy_manager.get_applicable_strategies(context, strategies)
        all_candidates = []
        
        for strategy in applicable_strategies:
            strategy_generators = self.get_generators_for_strategy(strategy)
            
            for generator in strategy_generators:
                if generator.can_generate(context, strategy):
                    candidates = generator.generate_candidates(context, strategy)
                    all_candidates.extend(candidates)
        
        return all_candidates
    
    def get_supported_strategies(self) -> List[GenerationStrategy]:
        """Get all strategies supported by registered generators."""
        all_strategies = set()
        for generator in self.generators:
            all_strategies.update(generator.get_supported_strategies())
        return list(all_strategies)

# ==============================================================================
# Main Coordinator Class
# ==============================================================================

class ModularMettaDonorGenerator:
    """Main coordinator for the modular MeTTa donor generation system."""
    
    def __init__(self, metta_space=None):
        from reflectors.dynamic_monitor import monitor
        
        self.metta_space = metta_space or monitor.metta_space
        self.pattern_detector = PatternDetector(self.metta_space)
        self.registry = GeneratorRegistry()
        self.monitor = monitor
        
        print("  Initializing Modular MeTTa Donor Generator...")
        
    def load_ontology(self, ontology_file: str = DONOR_GENERATION_ONTOLOGY) -> bool:
        """Load MeTTa ontology for modular generation."""
        print(" Loading modular donor generation ontology...")
        return self.monitor.load_metta_rules(ontology_file)
    
    def generate_donors_from_function(self, func: Union[Callable, str], 
                                    strategies: Optional[List[GenerationStrategy]] = None) -> List[Dict[str, Any]]:
        """Generate donor candidates from a function using the modular system."""
        print("  Starting Modular MeTTa Donor Generation...")
        
        # 1. Create generation context
        context = self._create_generation_context(func)
        if not context:
            return []
        
        print(f"  Created context for function: {context.function_name}")
        
        # 2. Detect patterns
        print("  Detecting patterns...")
        context.detected_patterns = self.pattern_detector.detect_patterns(context)
        print(f"  Detected {len(context.detected_patterns)} patterns")
        
        # 3. Generate candidates using registry
        print("  Generating candidates using modular generators...")
        candidates = self.registry.generate_candidates(context, strategies)
        
        if not candidates:
            print("  No candidates generated, using fallback...")
            candidates = self._fallback_generation(context, strategies)
        
        # 4. Rank and return results
        print("  Ranking candidates...")
        ranked_candidates = self._rank_candidates(candidates)
        
        print(f"  Generated {len(ranked_candidates)} candidates using modular system")
        return ranked_candidates
    
    def _create_generation_context(self, func: Union[Callable, str]) -> Optional[GenerationContext]:
        """Create a generation context from a function."""
        try:
            # Extract source code and function name
            if isinstance(func, str):
                original_code = func
                function_name = self._extract_function_name(func)
            else:
                original_code = inspect.getsource(func)
                function_name = func.__name__
            
            # Analyze the function
            from reflectors.static_analyzer import decompose_function, convert_to_metta_atoms, CodeDecomposer
            
            tree = ast.parse(original_code)
            decomposer = CodeDecomposer()
            decomposer.visit(tree)
            metta_atoms = convert_to_metta_atoms(decomposer)
            
            analysis_result = {
                "metta_atoms": metta_atoms,
                "structure": decomposer.atoms,
                "function_calls": decomposer.function_calls,
                "variables": decomposer.variables,
                "module_relationships": decomposer.module_relationships,
                "class_hierarchies": decomposer.class_hierarchies,
                "function_dependencies": decomposer.function_dependencies,
                "line_mapping": decomposer.line_mapping
            }
            
            # Load atoms into MeTTa space
            self._load_atoms_to_metta(metta_atoms, function_name)
            
            return GenerationContext(
                function_name=function_name,
                original_code=original_code,
                analysis_result=analysis_result,
                metta_atoms=metta_atoms,
                detected_patterns=[],  # Will be filled by pattern detector
                metta_space=self.metta_space
            )
            
        except Exception as e:
            print(f"  Failed to create generation context: {e}")
            return None
    
    def _extract_function_name(self, code: str) -> str:
        """Extract function name from code."""
        import re
        match = re.search(r'def\s+(\w+)', code)
        return match.group(1) if match else "unknown_function"
    
    def _load_atoms_to_metta(self, metta_atoms: List[str], function_name: str):
        """Load atoms into MeTTa space."""
        loaded_count = 0
        failed_count = 0
        
        for atom in metta_atoms:
            if self.monitor.add_atom(atom):
                loaded_count += 1
            else:
                failed_count += 1
        
        print(f"   Loaded {loaded_count}/{len(metta_atoms)} atoms ({failed_count} failed)")
        
        # Add function metadata
        self.monitor.add_atom(f"(modular-function {function_name})")
        self.monitor.add_atom(f"(analysis-timestamp {int(__import__('time').time())})")
    
    def _fallback_generation(self, context: GenerationContext, 
                           strategies: Optional[List[GenerationStrategy]]) -> List[DonorCandidate]:
        """Fallback generation when no generators produce candidates."""
        print("    Using modular fallback generation...")
        
        fallback_candidates = []
        
        # Create one basic candidate per requested strategy
        strategies_to_use = strategies or [GenerationStrategy.STRUCTURE_PRESERVATION]
        
        for strategy in strategies_to_use:
            candidate = DonorCandidate(
                name=f"{context.function_name}_{strategy.value}",
                description=f"Fallback {strategy.value.replace('_', ' ')} variant",
                code=self._create_fallback_code(context, strategy),
                strategy=strategy.value,
                pattern_family="generic",
                data_structures_used=["generic"],
                operations_used=["generic"],
                metta_derivation=[f"(fallback-generation {context.function_name} {strategy.value})"],
                confidence=0.5,
                properties=["fallback"],
                complexity_estimate="same",
                applicability_scope="narrow"
            )
            fallback_candidates.append(candidate)
        
        return fallback_candidates
    
    def _create_fallback_code(self, context: GenerationContext, strategy: GenerationStrategy) -> str:
        """Create fallback code for a strategy."""
        lines = context.original_code.split('\n')
        
        # Simple modifications based on strategy
        if strategy == GenerationStrategy.STRUCTURE_PRESERVATION:
            # Just add a comment
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    lines.insert(i+1, '    """Structure-preserving fallback variant."""')
                    break
        elif strategy == GenerationStrategy.ERROR_HANDLING_ENHANCEMENT:
            # Wrap in try-catch
            lines.insert(0, 'def safe_wrapper():')
            lines.insert(1, '    try:')
            lines = ['        ' + line if line.strip() else line for line in lines[2:]]
            lines.append('    except Exception as e:')
            lines.append('        return None')
        
        return '\n'.join(lines)
    
    def _rank_candidates(self, candidates: List[DonorCandidate]) -> List[Dict[str, Any]]:
        """Rank candidates using modular scoring."""
        scored_candidates = []
        
        for candidate in candidates:
            # Base score from confidence
            base_score = candidate.confidence
            
            # Modular scoring bonuses
            pattern_bonus = 0.1 if candidate.pattern_family != "generic" else 0.0
            structure_bonus = 0.05 * len(candidate.data_structures_used)
            operation_bonus = 0.02 * len(candidate.operations_used)
            
            # Strategy innovation bonus
            innovation_strategies = {
                "data_structure_adaptation", "algorithm_transformation", 
                "type_generalization", "functional_composition"
            }
            innovation_bonus = 0.15 if candidate.strategy in innovation_strategies else 0.0
            
            final_score = base_score + pattern_bonus + structure_bonus + operation_bonus + innovation_bonus
            final_score = min(1.0, final_score)
            
            scored_candidates.append({
                "name": candidate.name,
                "description": candidate.description,
                "code": candidate.code,
                "strategy": candidate.strategy,
                "pattern_family": candidate.pattern_family,
                "data_structures_used": candidate.data_structures_used,
                "operations_used": candidate.operations_used,
                "metta_derivation": candidate.metta_derivation,
                "confidence": candidate.confidence,
                "final_score": final_score,
                "properties": candidate.properties,
                "complexity_estimate": candidate.complexity_estimate,
                "applicability_scope": candidate.applicability_scope
            })
        
        return sorted(scored_candidates, key=lambda x: x["final_score"], reverse=True)

# ==============================================================================
# Integration function for compatibility
# ==============================================================================

def integrate_modular_metta_generation(func: Union[Callable, str], 
                                     strategies: Optional[List[GenerationStrategy]] = None) -> List[Dict[str, Any]]:
    """
    Integration function for the modular MeTTa generation system.
    """
    generator = ModularMettaDonorGenerator()
    generator.load_ontology()
    return generator.generate_donors_from_function(func, strategies)

# ==============================================================================
# Demo function
# ==============================================================================

def demonstrate_modular_generation():
    """Demonstrate the modular MeTTa generation system."""
    print("  MODULAR METTA DONOR GENERATION DEMO")
    print("=" * 60)
    
    # Example function 1: Search function
    def find_max_in_range(numbers, start_idx, end_idx):
        """Find the maximum value in a list within a specific range."""
        if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
            return None
        
        max_val = numbers[start_idx]
        for i in range(start_idx + 1, end_idx):
            if numbers[i] > max_val:
                max_val = numbers[i]
        
        return max_val
    
    # Example function 2: String processing function
    def clean_text(text):
        """Clean and normalize text input."""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        cleaned = text.strip()
        
        # Convert to lowercase
        cleaned = cleaned.lower()
        
        # Replace multiple spaces with single space
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned
    
    # Example function 3: Aggregation function
    def calculate_statistics(numbers):
        """Calculate basic statistics for a list of numbers."""
        if not numbers:
            return {}
        
        total = sum(numbers)
        count = len(numbers)
        average = total / count
        
        return {
            'sum': total,
            'count': count,
            'average': average,
            'min': min(numbers),
            'max': max(numbers)
        }
    
    # Test with different functions
    test_functions = [
        ("Search Pattern", find_max_in_range),
        ("String Processing", clean_text),
        ("Aggregation Pattern", calculate_statistics)
    ]
    
    for pattern_name, test_func in test_functions:
        print(f"\n  Testing {pattern_name}: {test_func.__name__}")
        print("-" * 50)
        
        try:
            candidates = integrate_modular_metta_generation(test_func)
            
            print(f"  Generated {len(candidates)} candidates")
            
            # Show top 2 candidates
            for i, candidate in enumerate(candidates[:2], 1):
                print(f"\n  {i}. {candidate['name']}")
                print(f"     Strategy: {candidate['strategy']}")
                print(f"     Pattern Family: {candidate['pattern_family']}")
                print(f"     Score: {candidate['final_score']:.2f}")
                print(f"     Data Structures: {', '.join(candidate['data_structures_used'])}")
                
                # Show code preview
                code_lines = candidate['code'].split('\n')
                print(f"     Code preview:")
                for line in code_lines[:4]:
                    print(f"       {line}")
                if len(code_lines) > 4:
                    print(f"       ... ({len(code_lines)-4} more lines)")
        
        except Exception as e:
            print(f"  Error testing {test_func.__name__}: {e}")
    
    return True

if __name__ == "__main__":
    demonstrate_modular_generation()