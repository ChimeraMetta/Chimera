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
import os
import inspect
import textwrap
from metta_generator.evolution.basic import BasicEvolutionEngine
from metta_generator.genetics.genome import SimpleCodeGenome, MeTTaGene
from metta_generator.evolution_integrator import (
        integrate_semantic_evolution_with_base_generator,
        EnhancedEvolutionIntegrator,
        SEMANTIC_EVOLUTION_AVAILABLE
    )

EVOLUTION_AVAILABLE = True
DONOR_GENERATION_ONTOLOGY = "metta/donor_generation.metta"
SEMANTIC_EVOLUTION_ONTOLOGY = "metta/semantic_evolution.metta"
_WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_WORKSPACE_ROOT, ".."))

@dataclass
class GenerationContext:
    """Context information for donor generation."""
    function_name: str
    original_code: str
    analysis_result: Dict[str, Any]
    metta_atoms: List[str]
    detected_patterns: List['FunctionPattern']
    metta_space: Any
    metta_reasoning_context: str = None  # New: context for MeTTa reasoning

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
    metta_derivation: List[str] = None  # New: MeTTa reasoning trace

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
    generator_used: str = "UnknownGenerator"
    metta_reasoning_trace: List[str] = None  # New: detailed MeTTa reasoning

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

class MeTTaReasoningEngine:
    """Core MeTTa reasoning engine for donor generation."""
    
    def __init__(self, metta_space):
        self.metta_space = metta_space
        
        # Initialize MeTTa parser for atom creation
        from hyperon import MeTTa
        self.metta = MeTTa()
        
        self._load_reasoning_rules()
    
    def _load_reasoning_rules(self):
        """Load core reasoning rules into MeTTa space."""
        core_rules = [
            # Pattern detection rules
            """(= (detect-pattern $func $atoms)
               (match &self 
                 (, (function-structure $func $structure)
                    (pattern-evidence $structure $pattern))
                 (pattern-detected $func $pattern)))""",
            
            # Transformation applicability rules  
            """(= (transformation-applicable $func $from-pattern $to-pattern)
               (and (has-pattern $func $from-pattern)
                    (safe-transformation $from-pattern $to-pattern)
                    (preserves-semantics $func $from-pattern $to-pattern)))""",
            
            # Safety checking rules
            """(= (safe-transformation iterative-pattern recursive-pattern)
               (and (has-clear-termination $func)
                    (no-complex-state $func)
                    (simple-accumulation $func)))""",
            
            # Code generation rules
            """(= (generate-variant $func $pattern $guidance)
               (match $pattern
                 (recursive-pattern (recursive-transform $func $guidance))
                 (functional-pattern (functional-transform $func $guidance))
                 (parallel-pattern (parallel-transform $func $guidance))))""",
            
            # Learning and adaptation rules
            """(= (adapt-rule $rule $feedback)
               (case $feedback
                 (successful (strengthen-rule $rule))
                 (failed (weaken-rule $rule))
                 (partial (refine-rule $rule))))"""
        ]
        
        for rule in core_rules:
            self._add_rule_safely(rule)
    
    def _add_rule_safely(self, rule: str):
        """Safely add rule to MeTTa space."""
        try:
            # Clean up the rule: remove extra whitespace and newlines
            cleaned_rule = ' '.join(rule.split())
            
            # Try different approaches based on what's available
            if hasattr(self.metta_space, 'run'):
                # Use run method if available (preferred)
                self.metta_space.run(f"!({cleaned_rule})")
                return True
            elif hasattr(self.metta_space, 'add_atom'):
                # For dynamic monitor spaces that expect atoms directly
                if hasattr(self, 'metta'):
                    try:
                        parsed_atom = self.metta.parse_single(cleaned_rule)
                        self.metta_space.add_atom(parsed_atom)
                        return True
                    except:
                        # Fallback to string if parsing fails
                        self.metta_space.add_atom(cleaned_rule)
                        return True
                else:
                    # Direct string addition
                    self.metta_space.add_atom(cleaned_rule)
                    return True
        except Exception as e:
            print(f"Failed to add rule: {e}")
            return False
    
    def reason_about_patterns(self, context: GenerationContext) -> List[FunctionPattern]:
        """Use MeTTa reasoning to detect patterns."""
        patterns = []
        
        # Create MeTTa reasoning context
        func_facts = self._create_function_facts(context)
        
        # Query MeTTa for pattern detection
        pattern_query = f"""
        (match &self 
          (detect-pattern {context.function_name} $pattern)
          (pattern-detected {context.function_name} $pattern))
        """
        
        pattern_results = self._execute_metta_reasoning(pattern_query, func_facts)
        
        for result in pattern_results:
            pattern = self._extract_pattern_from_result(result, context)
            if pattern:
                patterns.append(pattern)
        
        return patterns if patterns else [self._create_fallback_pattern(context)]
    
    def reason_about_transformations(self, context: GenerationContext, 
                                   strategy: GenerationStrategy) -> List[Dict[str, Any]]:
        """Use MeTTa reasoning to determine applicable transformations."""
        strategy_name = strategy.value if hasattr(strategy, 'value') else str(strategy)
        
        # Create transformation reasoning query
        transform_query = f"""
        (match &self
          (transformation-applicable {context.function_name} $from $to)
          (applicable-transform {context.function_name} $from $to $guidance))
        """
        
        func_facts = self._create_function_facts(context)
        strategy_facts = self._create_strategy_facts(strategy_name, context)
        
        all_facts = func_facts + strategy_facts
        
        return self._execute_metta_reasoning(transform_query, all_facts)
    
    def generate_code_through_reasoning(self, context: GenerationContext, 
                                      transformation: Dict[str, Any]) -> Optional[str]:
        """Use MeTTa reasoning to generate transformed code."""
        generation_query = f"""
        (match &self
          (generate-variant {context.function_name} {transformation.get('pattern')} $guidance)
          (generated-code {context.function_name} $code))
        """
        
        # Add transformation-specific facts
        transform_facts = self._create_transformation_facts(transformation, context)
        
        results = self._execute_metta_reasoning(generation_query, transform_facts)
        
        if results:
            return self._synthesize_code_from_reasoning(results, context, transformation)
        
        return None
    
    def learn_from_feedback(self, original_func: str, donor: DonorCandidate, 
                          feedback: str) -> None:
        """Learn from donor application feedback using MeTTa reasoning."""
        learning_rule = f"""
        (adapt-rule (transformation-rule {original_func} {donor.strategy}) {feedback})
        """
        
        self._add_rule_safely(learning_rule)
        
        # Update rule weights based on feedback
        if feedback == "successful":
            self._strengthen_related_rules(donor)
        elif feedback == "failed":
            self._weaken_related_rules(donor)
    
    def _create_function_facts(self, context: GenerationContext) -> List[str]:
        """Create MeTTa facts about the function."""
        facts = []
        
        # Basic function structure facts
        facts.append(f"(function-def {context.function_name})")
        
        # Add AST-derived facts
        for atom in context.metta_atoms:
            if context.function_name in atom:
                facts.append(atom)
        
        # Infer structural facts
        code = context.original_code.lower()
        
        if "for " in code or "while " in code:
            facts.append(f"(has-loop-structure {context.function_name})")
        
        if context.function_name in code.replace(f"def {context.function_name}", ""):
            facts.append(f"(calls-self {context.function_name})")
        
        if any(op in code for op in [">", "<", ">=", "<=", "==", "!="]):
            facts.append(f"(has-comparison-ops {context.function_name})")
        
        if "return" in code:
            facts.append(f"(has-return {context.function_name})")
        
        return facts
    
    def _create_strategy_facts(self, strategy: str, context: GenerationContext) -> List[str]:
        """Create MeTTa facts about the generation strategy."""
        facts = []
        
        facts.append(f"(generation-strategy {strategy})")
        facts.append(f"(target-function {context.function_name})")
        
        # Strategy-specific facts
        if strategy == "algorithm_transformation":
            facts.append(f"(requires-algorithmic-analysis {context.function_name})")
        elif strategy == "operation_substitution":
            facts.append(f"(requires-operation-analysis {context.function_name})")
        elif strategy == "data_structure_adaptation":
            facts.append(f"(requires-structure-analysis {context.function_name})")
        
        return facts
    
    def _create_transformation_facts(self, transformation: Dict[str, Any], 
                                   context: GenerationContext) -> List[str]:
        """Create MeTTa facts about a specific transformation."""
        facts = []
        
        pattern = transformation.get('pattern', 'unknown')
        target_pattern = transformation.get('target_pattern', 'unknown')
        
        facts.append(f"(transform-from {pattern})")
        facts.append(f"(transform-to {target_pattern})")
        facts.append(f"(transformation-target {context.function_name})")
        
        # Add guidance facts
        for key, value in transformation.items():
            if key not in ['pattern', 'target_pattern']:
                facts.append(f"(transformation-guidance {key} {value})")
        
        return facts
    
    def _execute_metta_reasoning(self, query: str, facts: List[str]) -> List[Any]:
        """Execute MeTTa reasoning with given query and facts."""
        print(f"        Executing MeTTa reasoning...")
        print(f"          Query: {query.strip()}")
        print(f"          Facts added: {len(facts)}")
        results = []
        
        try:
            # Add facts to reasoning context
            for fact in facts:
                self._add_rule_safely(fact)
            
            # Execute query
            print("          Executing query on MeTTa space...")
            if hasattr(self.metta_space, 'run'):
                try:
                    # Clean the query like we do for rules
                    cleaned_query = ' '.join(query.split())
                    query_result = self.metta_space.run(f"!({cleaned_query})")
                    if query_result:
                        results.extend(query_result)
                except Exception as e:
                    print(f"          MeTTa run failed: {e}")
                    # Try fallback approach without the ! prefix
                    try:
                        cleaned_query = ' '.join(query.split())
                        query_result = self.metta_space.run(f"({cleaned_query})")
                        if query_result:
                            results.extend(query_result)
                    except Exception as e2:
                        print(f"          MeTTa fallback also failed: {e2}")
            elif hasattr(self.metta_space, 'query'):
                try:
                    query_result = self.metta_space.query(query)
                    if query_result:
                        results.extend(query_result)
                except Exception as e:
                    print(f"          MeTTa query failed: {e}")
            
            print(f"          MeTTa execution returned {len(results)} results.")
        
        except Exception as e:
            print(f"        MeTTa reasoning failed: {e}")
            print("        Falling back to symbolic reasoning.")
            # Return symbolic reasoning fallback
            results = self._symbolic_reasoning_fallback(query, facts)
            print(f"          Symbolic fallback returned {len(results)} results.")
        
        return results
    
    def _symbolic_reasoning_fallback(self, query: str, facts: List[str]) -> List[Any]:
        """Fallback symbolic reasoning when MeTTa execution fails."""
        results = []
        
        # Pattern-based symbolic reasoning
        if "detect-pattern" in query:
            results = self._detect_patterns_symbolically(facts)
        elif "transformation-applicable" in query:
            results = self._detect_transformations_symbolically(facts)
        elif "generate-variant" in query:
            results = self._generate_variants_symbolically(facts)
        
        return results
    
    def _detect_patterns_symbolically(self, facts: List[str]) -> List[str]:
        """Symbolic pattern detection fallback."""
        patterns = []
        
        has_loop = any("has-loop-structure" in fact for fact in facts)
        has_comparison = any("has-comparison-ops" in fact for fact in facts)
        has_return = any("has-return" in fact for fact in facts)
        calls_self = any("calls-self" in fact for fact in facts)
        
        if has_loop and has_comparison and has_return:
            patterns.append("search-pattern")
        
        if has_loop and not calls_self:
            patterns.append("iterative-pattern")
        
        if calls_self:
            patterns.append("recursive-pattern")
        
        return patterns
    
    def _detect_transformations_symbolically(self, facts: List[str]) -> List[Dict[str, str]]:
        """Symbolic transformation detection fallback."""
        transformations = []
        
        has_iterative = any("iterative-pattern" in fact for fact in facts)
        has_recursive = any("recursive-pattern" in fact for fact in facts)
        
        if has_iterative:
            transformations.append({
                "from": "iterative-pattern",
                "to": "recursive-pattern",
                "safe": "true"
            })
        
        if has_recursive:
            transformations.append({
                "from": "recursive-pattern", 
                "to": "iterative-pattern",
                "safe": "true"
            })
        
        return transformations
    
    def _generate_variants_symbolically(self, facts: List[str]) -> List[str]:
        """Symbolic variant generation fallback."""
        return ["code-generation-placeholder"]
    
    def _extract_pattern_from_result(self, result: Any, context: GenerationContext) -> Optional[FunctionPattern]:
        """Extract FunctionPattern from MeTTa reasoning result."""
        try:
            # Parse MeTTa result to extract pattern information
            result_str = str(result)
            
            if "search-pattern" in result_str:
                return FunctionPattern(
                    pattern_family="search",
                    pattern_type="linear_search",
                    data_structures=["list"],
                    operations=["comparison"],
                    control_flow=["loop", "conditional"],
                    metta_evidence=[result_str],
                    confidence=0.8,
                    properties=["deterministic"],
                    metta_derivation=[f"metta-reasoning: {result_str}"]
                )
            elif "iterative-pattern" in result_str:
                return FunctionPattern(
                    pattern_family="transform",
                    pattern_type="iterative_processing",
                    data_structures=["iterable"],
                    operations=["traversal"],
                    control_flow=["loop"],
                    metta_evidence=[result_str],
                    confidence=0.85,
                    properties=["iterative"],
                    metta_derivation=[f"metta-reasoning: {result_str}"]
                )
        except Exception as e:
            print(f"Failed to extract pattern: {e}")
        
        return None
    
    def _create_fallback_pattern(self, context: GenerationContext) -> FunctionPattern:
        """Create fallback pattern when MeTTa reasoning fails."""
        return FunctionPattern(
            pattern_family="generic",
            pattern_type="general_function",
            data_structures=["generic"],
            operations=["generic"],
            control_flow=["sequential"],
            metta_evidence=[f"fallback-pattern {context.function_name}"],
            confidence=0.5,
            properties=["generic"],
            metta_derivation=[f"fallback-reasoning {context.function_name}"]
        )
    
    def _synthesize_code_from_reasoning(self, results: List[Any], 
                                      context: GenerationContext,
                                      transformation: Dict[str, Any]) -> str:
        """Synthesize code from MeTTa reasoning results."""
        # For now, return a placeholder that includes reasoning trace
        reasoning_trace = [str(r) for r in results]
        
        base_code = context.original_code
        new_func_name = f"{context.function_name}_{transformation.get('target_pattern', 'variant')}"
        
        # Basic transformation - replace function name and add reasoning comment
        transformed_code = base_code.replace(f"def {context.function_name}(", f"def {new_func_name}(")
        
        # Add MeTTa reasoning trace as comment
        reasoning_comment = f'    ""\"Generated through MeTTa reasoning: {", ".join(reasoning_trace)}"""'
        lines = transformed_code.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 1, reasoning_comment)
                break
        
        return '\n'.join(lines)
    
    def _strengthen_related_rules(self, donor: DonorCandidate):
        """Strengthen rules that led to successful donor."""
        strengthen_rule = f"""
        (= (rule-weight {donor.strategy}) 
           (+ (rule-weight {donor.strategy}) 0.1))
        """
        self._add_rule_safely(strengthen_rule)
    
    def _weaken_related_rules(self, donor: DonorCandidate):
        """Weaken rules that led to failed donor."""
        weaken_rule = f"""
        (= (rule-weight {donor.strategy})
           (- (rule-weight {donor.strategy}) 0.05))
        """
        self._add_rule_safely(weaken_rule)

class BaseDonorGenerator(ABC):
    """Abstract base class for MeTTa-powered donor generators."""
    
    def __init__(self):
        self.generator_name = self.__class__.__name__
        self.reasoning_engine = None  # Set by ModularMettaDonorGenerator
    
    @abstractmethod
    def can_generate(self, context: GenerationContext, strategy: GenerationStrategy) -> bool:
        """Check if this generator can handle the given context and strategy using MeTTa reasoning."""
        pass
    
    @abstractmethod
    def generate_candidates(self, context: GenerationContext, strategy: GenerationStrategy) -> List[DonorCandidate]:
        """Generate donor candidates using MeTTa reasoning."""
        candidates = self._generate_candidates_impl(context, strategy)
        
        # Ensure all candidates have proper generator attribution and MeTTa traces
        for candidate in candidates:
            if not hasattr(candidate, 'generator_used') or candidate.generator_used == "UnknownGenerator":
                candidate.generator_used = self.generator_name
            
            # Add MeTTa reasoning trace if not present
            if not candidate.metta_reasoning_trace:
                candidate.metta_reasoning_trace = [f"generated-by {self.generator_name}"]
        
        return candidates
    
    @abstractmethod
    def _generate_candidates_impl(self, context: GenerationContext, strategy: GenerationStrategy) -> List[DonorCandidate]:
        """Implementation method to be overridden by subclasses."""
        pass
    
    @abstractmethod
    def get_supported_strategies(self) -> List[GenerationStrategy]:
        """Get list of strategies this generator supports."""
        pass
    
    def _use_metta_reasoning(self, context: GenerationContext, query_type: str, **kwargs) -> Any:
        """Use MeTTa reasoning engine for decision making."""
        if not self.reasoning_engine:
            return self._fallback_reasoning(context, query_type, **kwargs)
        
        if query_type == "can_generate":
            return self._metta_can_generate_reasoning(context, **kwargs)
        elif query_type == "find_transformations":
            return self._metta_transformation_reasoning(context, **kwargs)
        elif query_type == "generate_code":
            return self._metta_code_generation_reasoning(context, **kwargs)
        
        return None
    
    def _metta_can_generate_reasoning(self, context: GenerationContext, **kwargs) -> bool:
        """Use MeTTa to determine if generator can handle context."""
        strategy = kwargs.get('strategy')
        
        query = f"""
        (match &self
          (generator-applicable {self.generator_name} {context.function_name} {strategy.value})
          True)
        """
        
        facts = [
            f"(generator-type {self.generator_name})",
            f"(target-function {context.function_name})",
            f"(strategy-requested {strategy.value})"
        ]
        
        results = self.reasoning_engine._execute_metta_reasoning(query, facts)
        return len(results) > 0
    
    def _metta_transformation_reasoning(self, context: GenerationContext, **kwargs) -> List[Dict[str, Any]]:
        """Use MeTTa to find applicable transformations."""
        strategy = kwargs.get('strategy')
        
        return self.reasoning_engine.reason_about_transformations(context, strategy)
    
    def _metta_code_generation_reasoning(self, context: GenerationContext, **kwargs) -> Optional[str]:
        """Use MeTTa to generate code."""
        transformation = kwargs.get('transformation', {})
        
        return self.reasoning_engine.generate_code_through_reasoning(context, transformation)
    
    def _fallback_reasoning(self, context: GenerationContext, query_type: str, **kwargs) -> Any:
        """Fallback reasoning when MeTTa engine is not available."""
        if query_type == "can_generate":
            return True  # Conservative fallback
        elif query_type == "find_transformations":
            return [{"pattern": "generic", "target_pattern": "variant"}]
        elif query_type == "generate_code":
            return self._basic_code_transformation(context, **kwargs)
        
        return None
    
    def _basic_code_transformation(self, context: GenerationContext, **kwargs) -> str:
        """Basic code transformation fallback."""
        transformation = kwargs.get('transformation', {})
        target_pattern = transformation.get('target_pattern', 'variant')
        
        new_func_name = f"{context.function_name}_{target_pattern}"
        return context.original_code.replace(f"def {context.function_name}(", f"def {new_func_name}(")

# ==============================================================================
# Main Coordinator Class
# ==============================================================================

class MeTTaPatternDetector:
    """MeTTa-powered pattern detector using symbolic reasoning."""
    
    def __init__(self, reasoning_engine: MeTTaReasoningEngine):
        self.reasoning_engine = reasoning_engine
        self._load_pattern_detection_rules()
    
    def _load_pattern_detection_rules(self):
        """Load pattern detection rules into MeTTa reasoning engine."""
        pattern_rules = [
            # Core pattern detection rules
            """(= (detect-search-pattern $func)
               (and (has-loop-structure $func)
                    (has-comparison-operations $func)
                    (has-conditional-return $func)))""",
            
            """(= (detect-transform-pattern $func)
               (and (has-loop-structure $func)
                    (has-assignment-operations $func)
                    (processes-collection $func)))""",
            
            """(= (detect-aggregate-pattern $func)
               (and (has-accumulator-variable $func)
                    (has-arithmetic-operations $func)
                    (reduces-collection $func)))""",
            
            """(= (detect-validate-pattern $func)
               (and (has-conditional-logic $func)
                    (returns-boolean $func)
                    (checks-properties $func)))""",
            
            # Pattern classification rules
            """(= (classify-pattern $func search-pattern linear-search)
               (and (detect-search-pattern $func)
                    (sequential-access $func)
                    (not (binary-search-structure $func))))""",
            
            """(= (classify-pattern $func transform-pattern map-like)
               (and (detect-transform-pattern $func)
                    (element-wise-processing $func)
                    (preserves-collection-size $func)))""",
            
            """(= (classify-pattern $func aggregate-pattern reduction)
               (and (detect-aggregate-pattern $func)
                    (combines-elements $func)
                    (produces-single-result $func)))""",
            
            # Pattern confidence rules
            """(= (pattern-confidence $func $pattern $confidence)
               (case $pattern
                 (search-pattern (search-confidence $func))
                 (transform-pattern (transform-confidence $func))
                 (aggregate-pattern (aggregate-confidence $func))
                 (validate-pattern (validate-confidence $func))))""",
            
            # Evidence accumulation rules
            """(= (accumulate-evidence $func $pattern $evidence)
               (collapse (match &self 
                 (pattern-indicator $func $pattern $indicator)
                 $indicator)))"""
        ]
        
        for rule in pattern_rules:
            self.reasoning_engine._add_rule_safely(rule)
    
    def detect_patterns(self, context: GenerationContext) -> List:
        """Use MeTTa reasoning to detect patterns."""
        print(f"     MeTTa pattern detection for: {context.function_name}")
        
        # Use MeTTa reasoning to detect patterns
        metta_patterns = self.reasoning_engine.reason_about_patterns(context)
        
        if metta_patterns:
            print(f"       MeTTa detected {len(metta_patterns)} patterns: {[p.pattern_family for p in metta_patterns]}")
            return metta_patterns
        
        # Enhanced fallback with symbolic reasoning
        print(f"       MeTTa detection failed, using enhanced symbolic fallback")
        return self._enhanced_symbolic_pattern_detection(context)
    
    def _enhanced_symbolic_pattern_detection(self, context: GenerationContext) -> List:
        """Enhanced symbolic pattern detection with MeTTa-like reasoning."""
        from metta_generator.base import FunctionPattern
        
        patterns = []
        code = context.original_code.lower()
        func_name = context.function_name.lower()
        metta_atoms = str(context.metta_space)
        
        # Detect data structures, operations, and control flow
        data_structures = self._detect_structures_symbolically(code, metta_atoms)
        operations = self._detect_operations_symbolically(code, metta_atoms)
        control_flow = self._detect_control_flow_symbolically(code, func_name)
        
        print(f"         Symbolic analysis: structures={data_structures}, ops={operations}, flow={control_flow}")
        
        # Apply MeTTa-like pattern reasoning
        detected_pattern_families = self._apply_symbolic_pattern_reasoning(
            context, data_structures, operations, control_flow
        )
        
        for family in detected_pattern_families:
            pattern_type = self._classify_pattern_type_symbolically(context, family)
            confidence = self._calculate_symbolic_confidence(context, family)
            properties = self._derive_symbolic_properties(family, operations, control_flow)
            
            pattern = FunctionPattern(
                pattern_family=family,
                pattern_type=pattern_type,
                data_structures=data_structures,
                operations=operations,
                control_flow=control_flow,
                metta_evidence=[f"(symbolic-pattern-detection {context.function_name} {family})"],
                confidence=confidence,
                properties=properties,
                metta_derivation=[f"symbolic-reasoning {context.function_name} {family}"]
            )
            patterns.append(pattern)
        
        # Add generic pattern if no specific patterns found
        if not patterns:
            patterns.append(self._create_generic_pattern_enhanced(context, data_structures, operations, control_flow))
        
        return patterns
    
    def _detect_structures_symbolically(self, code: str, metta_atoms: str) -> List[str]:
        """Enhanced symbolic structure detection."""
        structures = set()
        
        # Direct code analysis
        structure_indicators = {
            "list": ["[", "list(", ".append(", ".extend(", ".pop("],
            "dict": ["{", "dict(", ".keys(", ".values(", ".items("],
            "set": ["set(", ".add(", ".union(", ".intersection("],
            "string": ['"', "'", "str(", ".split(", ".join(", ".strip("],
            "tuple": ["tuple(", "(", ",)"],
            "numeric": ["int", "float", "sum", "max", "min"]
        }
        
        for struct_type, indicators in structure_indicators.items():
            if any(indicator in code for indicator in indicators):
                structures.add(struct_type)
        
        # MeTTa atoms analysis
        if "string-op" in metta_atoms:
            structures.add("string")
        if "bin-op" in metta_atoms and any(op in metta_atoms for op in ["Add", "Sub", "Mult"]):
            structures.add("numeric")
        
        return list(structures) if structures else ["generic"]
    
    def _detect_operations_symbolically(self, code: str, metta_atoms: str) -> List[str]:
        """Enhanced symbolic operation detection."""
        operations = set()
        
        # MeTTa atoms analysis (primary)
        if "bin-op" in metta_atoms:
            if any(op in metta_atoms for op in ["Lt", "Gt", "Eq", "NotEq"]):
                operations.add("comparison")
            if any(op in metta_atoms for op in ["Add", "Sub", "Mult", "Div"]):
                operations.add("arithmetic")
        
        # Direct code analysis (secondary)
        if any(op in code for op in ["and", "or", "not"]):
            operations.add("logical")
        if "variable-assign" in metta_atoms:
            operations.add("assignment")
        if "function-call" in metta_atoms:
            operations.add("function_call")
        if any(op in code for op in [".append", ".extend", ".remove", ".pop"]):
            operations.add("mutation")
        if any(op in code for op in [".split", ".join", ".replace", ".strip"]):
            operations.add("string_manipulation")
        
        return list(operations) if operations else ["generic"]
    
    def _detect_control_flow_symbolically(self, code: str, func_name: str) -> List[str]:
        """Enhanced symbolic control flow detection."""
        control_flow = set()
        
        if any(kw in code for kw in ["for ", "while "]):
            control_flow.add("loop")
        if "if " in code:
            control_flow.add("conditional")
        if func_name in code.replace(f"def {func_name}", ""):
            control_flow.add("recursion")
        if any(kw in code for kw in ["try:", "except:", "finally:", "raise"]):
            control_flow.add("exception_handling")
        if "return" in code:
            control_flow.add("return")
        if "yield" in code:
            control_flow.add("generator")
        
        return list(control_flow) if control_flow else ["sequential"]
    
    def _apply_symbolic_pattern_reasoning(self, context: GenerationContext,
                                        data_structures: List[str], 
                                        operations: List[str], 
                                        control_flow: List[str]) -> List[str]:
        """Apply MeTTa-like symbolic reasoning to determine pattern families."""
        patterns = set()
        func_name = context.function_name.lower()
        
        # Search pattern reasoning
        if (("loop" in control_flow and "comparison" in operations and "return" in control_flow) or
            any(keyword in func_name for keyword in ["find", "search", "locate", "get", "contains"])):
            patterns.add("search")
        
        # Transform pattern reasoning
        if (("loop" in control_flow and "assignment" in operations) or
            any(keyword in func_name for keyword in ["transform", "convert", "map", "process", "apply"])):
            patterns.add("transform")
        
        # Aggregate pattern reasoning
        if (("loop" in control_flow and "arithmetic" in operations) or
            any(keyword in func_name for keyword in ["sum", "count", "average", "total", "aggregate", "reduce"])):
            patterns.add("aggregate")
        
        # Validate pattern reasoning
        if (("conditional" in control_flow and "return" in control_flow) or
            any(keyword in func_name for keyword in ["validate", "check", "verify", "is_", "has_", "test"])):
            patterns.add("validate")
        
        # String operations pattern reasoning
        if ("string" in data_structures and "string_manipulation" in operations) or \
           any(keyword in func_name for keyword in ["parse", "format", "clean", "normalize"]):
            patterns.add("string_ops")
        
        # Math operations pattern reasoning
        if ("numeric" in data_structures and "arithmetic" in operations) or \
           any(keyword in func_name for keyword in ["calculate", "compute", "math"]):
            patterns.add("math_ops")
        
        # Control pattern reasoning
        if ("exception_handling" in control_flow or "conditional" in control_flow):
            patterns.add("control")
        
        return list(patterns) if patterns else ["generic"]
    
    def _classify_pattern_type_symbolically(self, context: GenerationContext, family: str) -> str:
        """Classify specific pattern type within family using symbolic reasoning."""
        func_name = context.function_name.lower()
        code = context.original_code.lower()
        
        type_mappings = {
            "search": {
                ("binary", "sorted"): "binary_search",
                ("first", "initial"): "find_first",
                ("all", "every"): "find_all",
                ("index", "position"): "find_index",
                "default": "linear_search"
            },
            "transform": {
                ("map", "apply"): "map_transform",
                ("filter", "select"): "filter_transform",
                ("convert", "change"): "conversion",
                "default": "generic_transform"
            },
            "aggregate": {
                ("sum", "total"): "sum_aggregation",
                ("count", "number"): "count_aggregation",
                ("average", "mean"): "average_aggregation",
                ("max", "maximum"): "max_aggregation",
                ("min", "minimum"): "min_aggregation",
                "default": "generic_aggregation"
            }
        }
        
        family_types = type_mappings.get(family, {})
        
        # Check for specific keywords
        for keywords, pattern_type in family_types.items():
            if keywords != "default" and isinstance(keywords, tuple):
                if any(keyword in func_name for keyword in keywords):
                    return pattern_type
        
        # Check code structure for additional clues
        if family == "search" and "binary" in code and "sorted" in code:
            return "binary_search"
        elif family == "transform" and any(func in code for func in ["map", "filter"]):
            return "higher_order_transform"
        
        return family_types.get("default", f"generic_{family}")
    
    def _calculate_symbolic_confidence(self, context: GenerationContext, family: str) -> float:
        """Calculate confidence using symbolic reasoning."""
        base_confidence = 0.6
        func_name = context.function_name.lower()
        
        # Name-based confidence boost
        family_keywords = {
            "search": ["find", "search", "locate", "get", "contains"],
            "transform": ["transform", "convert", "map", "process", "apply"],
            "aggregate": ["sum", "count", "average", "total", "aggregate"],
            "validate": ["validate", "check", "verify", "is_", "has_"]
        }
        
        keywords = family_keywords.get(family, [])
        if any(keyword in func_name for keyword in keywords):
            base_confidence += 0.2
        
        # Structure-based confidence boost
        code = context.original_code.lower()
        if family == "search" and "return" in code and ("for " in code or "while " in code):
            base_confidence += 0.15
        elif family == "aggregate" and any(op in code for op in ["+=", "sum", "total"]):
            base_confidence += 0.15
        
        return min(0.95, base_confidence)
    
    def _derive_symbolic_properties(self, family: str, operations: List[str], control_flow: List[str]) -> List[str]:
        """Derive properties using symbolic reasoning."""
        properties = ["symbolically-detected"]
        
        # Family-specific properties
        family_properties = {
            "search": ["deterministic", "side-effect-free"],
            "transform": ["functional", "composable"],
            "aggregate": ["associative", "fold-like"],
            "validate": ["predicate", "boolean-return"],
            "string_ops": ["text-processing"],
            "math_ops": ["numerical", "computational"]
        }
        
        properties.extend(family_properties.get(family, ["generic"]))
        
        # Operation-based properties
        if "arithmetic" in operations:
            properties.append("numerical")
        if "comparison" in operations:
            properties.append("comparative")
        if "mutation" not in operations:
            properties.append("side-effect-free")
        
        # Control flow-based properties
        if "recursion" in control_flow:
            properties.append("recursive")
        if "loop" in control_flow:
            properties.append("iterative")
        if "exception_handling" in control_flow:
            properties.append("error-handling")
        
        return properties
    
    def _create_generic_pattern_enhanced(self, context: GenerationContext,
                                       data_structures: List[str],
                                       operations: List[str], 
                                       control_flow: List[str]):
        """Create enhanced generic pattern with symbolic analysis."""
        from metta_generator.base import FunctionPattern
        
        return FunctionPattern(
            pattern_family="generic",
            pattern_type="general_function",
            data_structures=data_structures,
            operations=operations,
            control_flow=control_flow,
            metta_evidence=[f"(enhanced-symbolic-analysis {context.function_name})"],
            confidence=0.5,
            properties=["generic", "symbolically-analyzed", "adaptable"],
            metta_derivation=[f"enhanced-symbolic-reasoning {context.function_name}"]
        )

class MeTTaStrategyManager:
    """MeTTa-powered strategy manager using symbolic reasoning."""
    
    def __init__(self, generators: List, reasoning_engine: MeTTaReasoningEngine):
        self.generators = generators
        self.reasoning_engine = reasoning_engine
        self._load_strategy_rules()
    
    def _load_strategy_rules(self):
        """Load strategy applicability rules into MeTTa reasoning."""
        strategy_rules = [
            # Strategy applicability rules
            """(= (strategy-applicable operation-substitution $func)
               (or (has-substitutable-operations $func)
                   (has-semantic-substitution-opportunities $func)))""",
            
            """(= (strategy-applicable data-structure-adaptation $func)
               (and (uses-adaptable-structures $func)
                    (no-structure-specific-dependencies $func)))""",
            
            """(= (strategy-applicable algorithm-transformation $func)
               (or (has-transformable-algorithm-pattern $func)
                   (has-optimization-opportunities $func)))""",
            
            # Strategy prioritization rules
            """(= (strategy-priority $func operation-substitution high)
               (has-many-substitutable-operations $func))""",
            
            """(= (strategy-priority $func algorithm-transformation high)
               (and (has-clear-algorithm-pattern $func)
                    (transformation-benefits-significant $func)))""",
            
            # Strategy combination rules
            """(= (strategies-compatible operation-substitution structure-preservation) True)""",
            """(= (strategies-compatible data-structure-adaptation property-guided) True)""",
            """(= (strategies-compatible algorithm-transformation structure-preservation) False)"""
        ]
        
        for rule in strategy_rules:
            self.reasoning_engine._add_rule_safely(rule)
    
    def get_applicable_strategies(self, context: GenerationContext,
                                requested_strategies: Optional[List] = None):
        """Get applicable strategies using MeTTa reasoning."""
        if requested_strategies:
            # Convert string strategies to enum values if needed
            converted_strategies = []
            for strategy in requested_strategies:
                if isinstance(strategy, str):
                    # Convert string to enum
                    try:
                        enum_strategy = GenerationStrategy(strategy)
                        converted_strategies.append(enum_strategy)
                    except ValueError:
                        print(f"    Warning: Unknown strategy '{strategy}', skipping")
                        continue
                else:
                    converted_strategies.append(strategy)
            
            print(f"    Using requested strategies: {[s.value if hasattr(s, 'value') else str(s) for s in converted_strategies]}")
            return converted_strategies
        
        # Use MeTTa reasoning to determine applicable strategies
        metta_strategies = self._reason_about_strategy_applicability(context)
        
        if metta_strategies:
            print(f"    MeTTa reasoning found applicable strategies: {[s.value if hasattr(s, 'value') else str(s) for s in metta_strategies]}")
            return metta_strategies
        
        # Enhanced fallback strategy determination
        print(f"    MeTTa strategy reasoning failed, using enhanced symbolic fallback")
        return self._enhanced_symbolic_strategy_determination(context)
    
    def _reason_about_strategy_applicability(self, context: GenerationContext) -> List:
        """Use MeTTa reasoning to determine applicable strategies."""
        applicable_strategies = []
        
        # Query MeTTa for each strategy
        strategies_to_check = [
            GenerationStrategy.OPERATION_SUBSTITUTION,
            GenerationStrategy.DATA_STRUCTURE_ADAPTATION,
            GenerationStrategy.ALGORITHM_TRANSFORMATION,
            GenerationStrategy.STRUCTURE_PRESERVATION,
            GenerationStrategy.PROPERTY_GUIDED
        ]
        
        for strategy in strategies_to_check:
            strategy_name = strategy.value if hasattr(strategy, 'value') else str(strategy)
            
            applicability_query = f"""
            (match &self
              (strategy-applicable {strategy_name} {context.function_name})
              True)
            """
            
            strategy_facts = [
                f"(function-name {context.function_name})",
                f"(strategy-candidate {strategy_name})"
            ]
            
            results = self.reasoning_engine._execute_metta_reasoning(applicability_query, strategy_facts)
            
            if results and len(results) > 0:
                applicable_strategies.append(strategy)
        
        return applicable_strategies
    
    def _enhanced_symbolic_strategy_determination(self, context: GenerationContext) -> List:
        """Enhanced symbolic strategy determination with MeTTa-like reasoning."""
        applicable = []
        code = context.original_code.lower()
        func_name = context.function_name
        metta_atoms = str(context.metta_space)
        
        # Operation substitution applicability
        if (any(op in code for op in [">", "<", ">=", "<=", "+", "-", "*", "/", "==", "!=", "max", "min"]) or
            "bin-op" in metta_atoms):
            applicable.append(GenerationStrategy.OPERATION_SUBSTITUTION)
        
        # Data structure adaptation applicability
        if (any(pattern in code for pattern in ["[", "list(", "dict(", "set(", ".append", ".keys"]) or
            any(pattern in func_name.lower() for pattern in ["list", "dict", "set"])):
            applicable.append(GenerationStrategy.DATA_STRUCTURE_ADAPTATION)
        
        # Algorithm transformation applicability
        if (any(pattern in code for pattern in ["for ", "while ", "range("]) or 
            func_name in code.replace(f"def {func_name}", "") or
            "loop-pattern" in metta_atoms):
            applicable.append(GenerationStrategy.ALGORITHM_TRANSFORMATION)
        
        # Structure preservation (always applicable)
        applicable.append(GenerationStrategy.STRUCTURE_PRESERVATION)
        
        # Property guided (always applicable if patterns detected)
        if hasattr(context, 'detected_patterns') and context.detected_patterns:
            applicable.append(GenerationStrategy.PROPERTY_GUIDED)
        
        # Remove duplicates while preserving order
        seen = set()
        applicable = [x for x in applicable if not (x in seen or seen.add(x))]
        
        return applicable

class MeTTaPoweredModularDonorGenerator:
    """Main coordinator using MeTTa reasoning as the core engine."""
    
    def __init__(self, metta_space=None, enable_evolution=True):
        from reflectors.dynamic_monitor import monitor
        
        self.metta_space = metta_space or monitor
        
        # Initialize MeTTa reasoning engine
        self.reasoning_engine = MeTTaReasoningEngine(self.metta_space)
        
        # Initialize MeTTa-powered components
        self.pattern_detector = MeTTaPatternDetector(self.reasoning_engine)
        self.strategy_manager = MeTTaStrategyManager([], self.reasoning_engine)
        
        # Initialize generators directly
        self.generators = []
        self._setup_generators_with_reasoning()
        
        print("  Initialized MeTTa-Powered Modular Donor Generator...")

       # Add semantic evolution capability
        self.enable_semantic_evolution = enable_evolution and SEMANTIC_EVOLUTION_AVAILABLE
        
        if self.enable_semantic_evolution:
            print("  Enabling semantic evolution integration...")
            integrate_semantic_evolution_with_base_generator(self, enable_semantic=True)
            # Load semantic evolution MeTTa rules
            self._load_semantic_evolution_rules()
        else:
            print("  Semantic evolution not enabled")
    
    def _load_semantic_evolution_rules(self):
        """Load semantic evolution MeTTa rules"""
        semantic_rules_file = os.path.join(_PROJECT_ROOT, SEMANTIC_EVOLUTION_ONTOLOGY)
        if os.path.exists(semantic_rules_file):
            try:
                with open(semantic_rules_file, 'r') as f:
                    rules_content = f.read()
                
                # Parse and load rules
                for line in rules_content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith(';') and line.startswith('(='):
                        self.reasoning_engine._add_rule_safely(line)
                
                print(f"    Loaded semantic evolution rules from {semantic_rules_file}")
            except Exception as e:
                print(f"    Failed to load semantic evolution rules: {e}")
        else:
            print(f"    Semantic evolution rules file not found: {semantic_rules_file}")
    
    def _setup_generators_with_reasoning(self):
        """Setup generators with access to MeTTa reasoning engine."""
        # Import and create MeTTa-powered generators
        from metta_generator.algo_transformation import AlgorithmTransformationGenerator
        from metta_generator.operation_substitution import OperationSubstitutionGenerator
        from metta_generator.data_struct_adaptation import DataStructureAdaptationGenerator
        from metta_generator.structure_preservation import StructurePreservationGenerator
        
        self.generators = [
            AlgorithmTransformationGenerator(),
            OperationSubstitutionGenerator(),
            DataStructureAdaptationGenerator(),
            StructurePreservationGenerator()
        ]
        
        # Set reasoning engine on each generator
        for generator in self.generators:
            generator.reasoning_engine = self.reasoning_engine
        
        # Update strategy manager with generators
        self.strategy_manager.generators = self.generators
    
    def get_generators_for_strategy(self, strategy: GenerationStrategy) -> List:
        """Get all generators that can handle a specific strategy."""
        return [gen for gen in self.generators if strategy in gen.get_supported_strategies()]
    
    def generate_donors_with_evolution(self, func: Union[Callable, str],
                                     strategies: Optional[List] = None) -> List[Dict[str, Any]]:
        """Generate donor candidates using evolutionary algorithm (experimental)"""
        
        if not self.enable_evolution:
            print("  Evolution not enabled, falling back to standard generation")
            return self.generate_donors_from_function(func, strategies)
        
        print("Starting evolutionary donor generation...")
        
        # Create generation context like your existing method
        context = self._create_generation_context(func)
        if not context:
            return []
        
        print(f"  Function to evolve: {context.function_name}")
        
        # Use evolution engine
        try:
            evolved_population = self.evolution_engine.evolve_population(
                context.original_code, 
                context.function_name
            )
            
            # Convert evolved genomes to donor candidates
            candidates = []
            for i, genome in enumerate(evolved_population[:5]):  # Top 5
                evolved_code = self.evolution_engine.genome_to_code(genome, context.function_name)
                
                candidate = {
                    "name": f"{context.function_name}_evolved_gen{genome.generation}_{i+1}",
                    "description": f"Evolved candidate (fitness: {genome.fitness_score:.3f})",
                    "code": evolved_code,
                    "strategy": "evolutionary_algorithm",
                    "pattern_family": "evolved",
                    "data_structures_used": [gene.pattern_type for gene in genome.genes],
                    "operations_used": ["evolution"],
                    "metta_derivation": [gene.metta_atom for gene in genome.genes],
                    "confidence": genome.fitness_score,
                    "final_score": genome.fitness_score,
                    "properties": ["evolved", "metta-guided"],
                    "complexity_estimate": "similar",
                    "applicability_scope": "experimental",
                    "generator_used": "BasicEvolutionEngine",
                    "evolution_metadata": {
                        "genome_id": genome.genome_id,
                        "generation": genome.generation,
                        "parent_ids": genome.parent_ids,
                        "gene_count": len(genome.genes),
                        "gene_types": list(set(gene.pattern_type for gene in genome.genes))
                    }
                }
                candidates.append(candidate)
            
            print(f"Evolution generated {len(candidates)} candidates")
            return candidates
            
        except Exception as e:
            print(f"Evolution failed: {e}")
            print("  Falling back to standard generation")
            return self.generate_donors_from_function(func, strategies)
    
    def generate_donors_with_semantic_evolution(self, func, strategies=None):
        """Generate donors using semantic evolution if available"""
        if not self.enable_semantic_evolution:
            return self.generate_donors_from_function(func, strategies)
        
        # This will use the integrated semantic evolution
        return self.generate_donors_from_function(func, strategies, use_semantic_evolution=True)
    
    def generate_candidates(self, context: GenerationContext, 
                          strategies: Optional[List] = None) -> List:
        """Generate candidates using MeTTa-powered generators."""
        if not strategies:
            strategies = self.strategy_manager.get_applicable_strategies(context)
        
        print(f"    Processing strategies: {[s.value if hasattr(s, 'value') else str(s) for s in strategies]}")  
        
        all_candidates = []
        
        for strategy in strategies:
            strategy_name = strategy.value if hasattr(strategy, 'value') else str(strategy)
            print(f"    Processing strategy: {strategy_name}")  
            
            strategy_generators = self.get_generators_for_strategy(strategy)
            print(f"      Found {len(strategy_generators)} generators for {strategy_name}")  
            
            for generator in strategy_generators:
                print(f"      Checking {generator.generator_name}")  
                
                if generator.can_generate(context, strategy):
                    print(f"        {generator.generator_name} can generate")  
                    try:
                        candidates = generator.generate_candidates(context, strategy)
                        print(f"        Generated {len(candidates)} candidates")  
                        all_candidates.extend(candidates)
                    except Exception as e:
                        print(f"        Error generating candidates: {e}")  
                        import traceback
                        traceback.print_exc()  
                else:
                    print(f"        {generator.generator_name} cannot generate")  
        
        print(f"    Total candidates generated: {len(all_candidates)}")  
        return all_candidates
    
    def load_ontology(self, ontology_file: str = DONOR_GENERATION_ONTOLOGY) -> bool:
        """Load MeTTa ontology for reasoning-powered generation."""
        print(" Loading MeTTa reasoning-powered donor generation ontology...")
        
        # Load enhanced ontology rules
        enhanced_rules = [
            # Core reasoning rules
            """(= (donor-generation-applicable $func $strategy)
               (and (function-analyzable $func)
                    (strategy-applicable $strategy $func)
                    (no-contraindications $func $strategy)))""",
            
            # Quality assessment rules
            """(= (donor-quality $donor $quality)
               (let $correctness (correctness-score $donor)
                    (let $maintainability (maintainability-score $donor)
                         (let $performance (performance-score $donor)
                              (overall-quality $correctness $maintainability $performance)))))""",
            
            # Learning and adaptation rules
            """(= (learn-from-success $original $donor $feedback)
               (strengthen-patterns (patterns-used $original $donor)))""",
            
            """(= (learn-from-failure $original $donor $feedback)
               (weaken-patterns (patterns-used $original $donor)))"""
        ]
        
        for rule in enhanced_rules:
            self.reasoning_engine._add_rule_safely(rule)
        
        return True
    
    def generate_donors_from_function(self, func: Union[Callable, str],
                                    strategies: Optional[List] = None) -> List[Dict[str, Any]]:
        """Generate donor candidates using MeTTa reasoning throughout the process."""
        print("  Starting MeTTa-Powered Donor Generation...")
        
        # 1. Create generation context
        context = self._create_generation_context(func)
        if not context:
            return []
        
        print(f"  Created context for function: {context.function_name}")
        
        # 2. Use MeTTa reasoning for pattern detection
        print("  Using MeTTa reasoning for pattern detection...")
        context.detected_patterns = self.pattern_detector.detect_patterns(context)
        print(f"  MeTTa detected {len(context.detected_patterns)} patterns")
        
        # 3. Use MeTTa reasoning for strategy determination
        print("  Using MeTTa reasoning for strategy determination...")
        applicable_strategies = self.strategy_manager.get_applicable_strategies(context, strategies)
        
        # 4. Generate candidates using MeTTa-powered generators
        print("  Generating candidates using MeTTa-powered generators...")
        candidates = self.generate_candidates(context, applicable_strategies)
        
        if not candidates:
            print("  No MeTTa candidates generated, using enhanced fallback...")
            candidates = self._metta_guided_fallback_generation(context, applicable_strategies)
        
        # 5. Use MeTTa reasoning for candidate ranking
        print("  Using MeTTa reasoning for candidate ranking...")
        ranked_candidates = self._metta_rank_candidates(candidates, context)
        
        print(f"  Generated {len(ranked_candidates)} candidates using MeTTa reasoning")
        return ranked_candidates
    
    def _create_generation_context(self, func: Union[Callable, str]) -> Optional[GenerationContext]:
        """Create generation context with MeTTa reasoning support."""
        print("    Creating generation context...")
        try:
            # Extract source code and function name
            if isinstance(func, str):
                full_code_block = func
                function_name = self._extract_function_name(full_code_block)
                print(f"      Context from string, isolating function '{function_name}'...")

                lines = full_code_block.split('\n')
                start_line_idx = -1
                import re

                func_def_pattern = re.compile(fr'\bdef\s+{function_name}\b')
                for i, line in enumerate(lines):
                    if func_def_pattern.search(line):
                        start_line_idx = i
                        break
                
                if start_line_idx == -1:
                    print(f"      ERROR: Could not find start of function '{function_name}'.")
                    return None

                first_line_idx = start_line_idx
                for i in range(start_line_idx - 1, -1, -1):
                    line = lines[i].strip()
                    if line.startswith('@'):
                        first_line_idx = i
                    elif not line:
                        continue
                    else:
                        break
                
                base_indent_level = len(lines[first_line_idx]) - len(lines[first_line_idx].lstrip())
                
                end_line_idx = len(lines)
                for i in range(start_line_idx + 1, len(lines)):
                    line = lines[i]
                    if not line.strip():
                        continue
                    
                    line_indent_level = len(line) - len(line.lstrip())
                    if line_indent_level <= base_indent_level:
                        end_line_idx = i
                        break
                
                original_code = '\n'.join(lines[first_line_idx:end_line_idx])
                print(f"      Successfully isolated source for '{function_name}'.")

            else:
                original_code = inspect.getsource(func)
                function_name = func.__name__
                print(f"      Context from callable, function name: {function_name}")
            
            original_code = textwrap.dedent(original_code)
            
            # Analyze the function using existing infrastructure
            print("      Analyzing function with static analyzer...")
            from reflectors.static_analyzer import decompose_function, convert_to_metta_atoms, CodeDecomposer
            
            parsable_code = original_code
            # If the code is for a method (e.g., starts with @), ast.parse will fail.
            # We can wrap it in a dummy class to make it syntactically valid.
            if parsable_code.strip().startswith('@'):
                print("      Detected decorated function, wrapping in a dummy class for parsing.")
                parsable_code = f"class DummyWrapper:\n{textwrap.indent(parsable_code, '    ')}"

            tree = ast.parse(parsable_code)
            decomposer = CodeDecomposer()
            decomposer.visit(tree)
            metta_atoms = convert_to_metta_atoms(decomposer)
            print(f"      Static analysis complete. Found {len(metta_atoms)} MeTTa atoms.")
            
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
            
            # Load atoms into MeTTa reasoning space
            self._load_atoms_for_reasoning(metta_atoms, function_name)
            
            # Create reasoning context identifier
            reasoning_context = f"reasoning-context-{function_name}-{int(__import__('time').time())}"
            print(f"      Created MeTTa reasoning context: {reasoning_context}")
            
            context = GenerationContext(
                function_name=function_name,
                original_code=original_code,
                analysis_result=analysis_result,
                metta_atoms=metta_atoms,
                detected_patterns=[],  # Will be filled by MeTTa pattern detector
                metta_space=self.metta_space,
                metta_reasoning_context=reasoning_context
            )
            print("    Generation context created successfully.")
            return context
            
        except Exception as e:
            print(f"  Failed to create generation context: {e}")
            print("  --- Problematic Code ---")
            print(original_code)
            print("  ------------------------")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_function_name(self, code: str) -> str:
        """Extract function name from code."""
        import re
        match = re.search(r'def\s+(\w+)', code)
        return match.group(1) if match else "unknown_function"
    
    def _load_atoms_for_reasoning(self, metta_atoms: List[str], function_name: str):
        """Load atoms into MeTTa reasoning space."""
        loaded_count = 0
        failed_count = 0
        
        # Load function-specific atoms
        for atom in metta_atoms:
            if self.reasoning_engine._add_rule_safely(atom):
                loaded_count += 1
            else:
                failed_count += 1
        
        print(f"   Loaded {loaded_count}/{len(metta_atoms)} atoms for reasoning ({failed_count} failed)")
        
        # Add function metadata for reasoning
        function_facts = [
            f"(analyzed-function {function_name})",
            f"(reasoning-timestamp {int(__import__('time').time())})",
            f"(atoms-loaded {loaded_count})"
        ]
        
        for fact in function_facts:
            self.reasoning_engine._add_rule_safely(fact)
    
    def _metta_guided_fallback_generation(self, context: GenerationContext,
                                        strategies: List) -> List[DonorCandidate]:
        """MeTTa-guided fallback generation when main generators fail."""
        print("    Using MeTTa-guided fallback generation...")
        
        fallback_candidates = []
        
        # Use MeTTa reasoning to determine fallback approaches
        fallback_query = f"""
        (match &self
          (fallback-generation-approach {context.function_name} $approach)
          (fallback-strategy {context.function_name} $approach))
        """
        
        fallback_facts = [
            f"(function-name {context.function_name})",
            f"(main-generation-failed True)",
            f"(detected-patterns {len(context.detected_patterns)})"
        ]
        
        fallback_approaches = self.reasoning_engine._execute_metta_reasoning(fallback_query, fallback_facts)
        
        if not fallback_approaches:
            # Symbolic fallback determination
            fallback_approaches = self._determine_symbolic_fallback_approaches(context, strategies)
        
        # Generate fallback candidates based on MeTTa reasoning
        for approach in fallback_approaches:
            candidate = self._create_metta_fallback_candidate(context, approach, strategies)
            if candidate:
                fallback_candidates.append(candidate)
        
        return fallback_candidates
    
    def _determine_symbolic_fallback_approaches(self, context: GenerationContext, 
                                              strategies: List) -> List[str]:
        """Determine fallback approaches using symbolic reasoning."""
        approaches = []
        
        # Basic structure preservation approach
        approaches.append("structure-preservation")
        
        # Pattern-based approaches
        if context.detected_patterns:
            for pattern in context.detected_patterns:
                approaches.append(f"pattern-based-{pattern.pattern_family}")
        
        # Strategy-based approaches
        for strategy in strategies:
            strategy_name = strategy.value if hasattr(strategy, 'value') else str(strategy)
            approaches.append(f"minimal-{strategy_name}")
        
        return approaches[:3]  # Limit to top 3 approaches
    
    def _create_metta_fallback_candidate(self, context: GenerationContext, approach: str, strategies: List) -> Optional[DonorCandidate]:
        """Create fallback candidate using MeTTa reasoning guidance."""
        try:
            from metta_generator.base import DonorCandidate
            
            approach_str = str(approach)
            
            # Generate fallback code based on MeTTa-guided approach
            if "structure-preservation" in approach_str:
                fallback_code = self._create_structure_preserving_fallback(context)
                strategy_name = "structure_preservation"
            elif "pattern-based" in approach_str:
                fallback_code = self._create_pattern_based_fallback(context, approach_str)
                strategy_name = "pattern_guided"
            elif "minimal" in approach_str:
                fallback_code = self._create_minimal_transformation_fallback(context, approach_str)
                strategy_name = approach_str.replace("minimal-", "")
            else:
                fallback_code = self._create_generic_fallback(context)
                strategy_name = "generic"
            
            new_func_name = f"{context.function_name}_metta_fallback_{strategy_name}"
            
            metta_derivation = [
                f"(metta-fallback-generation {context.function_name})",
                f"(fallback-approach {approach_str})",
                f"(reasoning-guided fallback)"
            ]
            
            return DonorCandidate(
                name=new_func_name,
                description=f"MeTTa-guided fallback: {approach_str}",
                code=fallback_code,
                strategy=strategy_name,
                pattern_family="fallback",
                data_structures_used=["generic"],
                operations_used=["fallback"],
                metta_derivation=metta_derivation,
                confidence=0.5,
                properties=["metta-fallback", "reasoning-guided"],
                complexity_estimate="same",
                applicability_scope="narrow",
                generator_used="MeTTaFallback",
                metta_reasoning_trace=[f"fallback-reasoning: {approach}"]
            )
            
        except Exception as e:
            print(f"      Failed to create MeTTa fallback candidate: {e}")
            return None
    
    def _create_structure_preserving_fallback(self, context: GenerationContext) -> str:
        """Create structure-preserving fallback with MeTTa guidance."""
        lines = context.original_code.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i+1, '    """MeTTa-guided structure-preserving fallback."""')
                break
        
        return '\n'.join(lines)
    
    def _create_pattern_based_fallback(self, context: GenerationContext, approach: str) -> str:
        """Create pattern-based fallback using MeTTa reasoning."""
        pattern_family = approach.split("pattern-based-")[-1]
        
        lines = context.original_code.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i+1, f'    """MeTTa pattern-guided fallback: {pattern_family}."""')
                break
        
        # Add pattern-specific comment
        for i, line in enumerate(lines):
            if 'return' in line:
                lines.insert(i, f'        # MeTTa reasoning: {pattern_family} pattern detected')
                break
        
        return '\n'.join(lines)
    
    def _create_minimal_transformation_fallback(self, context: GenerationContext, approach: str) -> str:
        """Create minimal transformation fallback with MeTTa guidance."""
        strategy_name = approach.replace("minimal-", "")
        
        transformed_code = context.original_code
        
        # Apply minimal transformation based on strategy
        if "operation" in strategy_name:
            # Minimal operation change
            transformed_code = transformed_code.replace("def ", "def minimally_")
        elif "structure" in strategy_name:
            # Minimal structure change
            transformed_code = transformed_code.replace("def ", "def adapted_")
        elif "algorithm" in strategy_name:
            # Minimal algorithm change
            transformed_code = transformed_code.replace("def ", "def transformed_")
        
        # Add MeTTa reasoning documentation
        lines = transformed_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i+1, f'    """MeTTa minimal transformation: {strategy_name}."""')
                break
        
        return '\n'.join(lines)
    
    def _create_generic_fallback(self, context: GenerationContext) -> str:
        """Create generic fallback with MeTTa reasoning traces."""
        new_func_name = f"{context.function_name}_metta_generic"
        fallback_code = context.original_code.replace(f"def {context.function_name}(", f"def {new_func_name}(")
        
        lines = fallback_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i+1, '    """MeTTa generic fallback with reasoning traces."""')
                break
        
        return '\n'.join(lines)
    
    def _metta_rank_candidates(self, candidates: List, context: GenerationContext) -> List[Dict[str, Any]]:
        """Rank candidates using MeTTa reasoning."""
        scored_candidates = []
        
        for candidate in candidates:
            # Use MeTTa reasoning to score candidate
            metta_score = self._calculate_metta_candidate_score(candidate, context)
            
            # Combine with traditional scoring
            traditional_score = self._calculate_traditional_score(candidate)
            
            # Weight MeTTa reasoning more heavily
            final_score = (0.7 * metta_score) + (0.3 * traditional_score)
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
                "metta_score": metta_score,
                "properties": candidate.properties,
                "complexity_estimate": candidate.complexity_estimate,
                "applicability_scope": candidate.applicability_scope,
                "generator_used": candidate.generator_used,
                "metta_reasoning_trace": getattr(candidate, 'metta_reasoning_trace', [])
            })
        
        # Sort by final score (MeTTa-weighted)
        return sorted(scored_candidates, key=lambda x: x["final_score"], reverse=True)
    
    def _calculate_metta_candidate_score(self, candidate, context: GenerationContext) -> float:
        """Calculate candidate score using MeTTa reasoning."""
        print(f"      Calculating MeTTa score for candidate: {candidate.name}")
        # Query MeTTa for candidate quality assessment
        quality_query = f"""
        (match &self
          (donor-quality {candidate.name} $quality)
          (quality-score {candidate.name} $quality))
        """
        
        quality_facts = [
            f"(candidate-name {candidate.name})",
            f"(candidate-strategy {candidate.strategy})",
            f"(source-function {context.function_name})",
            f"(metta-derived {len(candidate.metta_derivation)})"
        ]
        
        # Add reasoning trace facts
        if hasattr(candidate, 'metta_reasoning_trace') and candidate.metta_reasoning_trace is not None:
            for trace in candidate.metta_reasoning_trace:
                quality_facts.append(f"(reasoning-trace {trace})")
        
        print(f"        MeTTa quality query: {quality_query.strip()}")
        print(f"        MeTTa quality facts: {quality_facts}")

        quality_results = self.reasoning_engine._execute_metta_reasoning(quality_query, quality_facts)
        print(f"        MeTTa quality results: {quality_results}")
        
        if quality_results:
            print("        Using MeTTa reasoning results for scoring.")
            try:
                # Extract numeric score from MeTTa result
                result_str = str(quality_results[0])
                print(f"          MeTTa result string: {result_str}")
                if "high" in result_str:
                    metta_score = 0.9
                elif "medium" in result_str:
                    metta_score = 0.7
                elif "low" in result_str:
                    metta_score = 0.5
                else:
                    metta_score = 0.6
                print(f"          Extracted score: {metta_score}")
            except Exception as e:
                print(f"          Error extracting score from MeTTa result: {e}. Defaulting to 0.6.")
                metta_score = 0.6
        else:
            print("        MeTTa reasoning yielded no results, using fallback symbolic scoring.")
            # Fallback to symbolic scoring
            metta_score = self._symbolic_metta_scoring(candidate, context)
            print(f"          Fallback symbolic score: {metta_score}")
        
        print(f"      Final MeTTa score for {candidate.name}: {metta_score}")
        return metta_score
    
    def _symbolic_metta_scoring(self, candidate, context: GenerationContext) -> float:
        """Fallback symbolic scoring that mimics MeTTa reasoning."""
        base_score = candidate.confidence
        
        # MeTTa reasoning quality bonuses
        if hasattr(candidate, 'metta_derivation') and candidate.metta_derivation:
            base_score += 0.1 * len(candidate.metta_derivation) / 5  # Max 0.1 bonus
        
        if hasattr(candidate, 'metta_reasoning_trace') and candidate.metta_reasoning_trace:
            base_score += 0.05
        
        # Property-based bonuses (MeTTa-style reasoning)
        if "metta-reasoned" in candidate.properties:
            base_score += 0.1
        if "reasoning-guided" in candidate.properties:
            base_score += 0.05
        if "safe" in candidate.properties or "transformation-safe" in candidate.properties:
            base_score += 0.05
        
        # Strategy-specific bonuses
        if candidate.strategy in ["algorithm_transformation", "operation_substitution"]:
            base_score += 0.05
        
        return min(1.0, base_score)
    
    def _calculate_traditional_score(self, candidate) -> float:
        """Calculate traditional scoring for combination with MeTTa score."""
        base_score = candidate.confidence
        
        # Pattern bonus
        pattern_bonus = 0.1 if candidate.pattern_family != "generic" else 0.0
        
        # Structure and operation bonuses
        structure_bonus = 0.02 * len(candidate.data_structures_used)
        operation_bonus = 0.02 * len(candidate.operations_used)
        
        # Applicability bonus
        applicability_bonus = {"broad": 0.1, "medium": 0.05, "narrow": 0.0}.get(
            candidate.applicability_scope, 0.0)
        
        final_score = base_score + pattern_bonus + structure_bonus + operation_bonus + applicability_bonus
        return min(1.0, final_score)
    
    def learn_from_feedback(self, original_func: str, donor_results: List[Dict[str, Any]]) -> None:
        """Learn from donor application results using MeTTa reasoning."""
        print(f"  Learning from {len(donor_results)} donor applications using MeTTa reasoning...")
        
        for result in donor_results:
            donor_name = result.get('donor_name')
            feedback = result.get('feedback', 'unknown')  # 'successful', 'failed', 'partial'
            
            # Use MeTTa reasoning for learning
            learning_rule = f"""
            (learn-from-feedback {original_func} {donor_name} {feedback})
            """
            
            self.reasoning_engine._add_rule_safely(learning_rule)
            
            # Specific learning based on feedback
            if feedback == 'successful':
                self._strengthen_successful_patterns(result)
            elif feedback == 'failed':
                self._weaken_failed_patterns(result)
            
        print(f"    MeTTa learning completed, reasoning base updated")
    
    def _strengthen_successful_patterns(self, result: Dict[str, Any]):
        """Strengthen patterns that led to successful donors."""
        strategy = result.get('strategy')
        pattern_family = result.get('pattern_family')
        
        if strategy and pattern_family:
            strengthening_rule = f"""
            (= (pattern-strategy-success {pattern_family} {strategy}) 
               (+ (pattern-strategy-success {pattern_family} {strategy}) 0.1))
            """
            self.reasoning_engine._add_rule_safely(strengthening_rule)
    
    def _weaken_failed_patterns(self, result: Dict[str, Any]):
        """Weaken patterns that led to failed donors."""
        strategy = result.get('strategy')
        pattern_family = result.get('pattern_family')
        
        if strategy and pattern_family:
            weakening_rule = f"""
            (= (pattern-strategy-failure {pattern_family} {strategy})
               (+ (pattern-strategy-failure {pattern_family} {strategy}) 0.1))
            """
            self.reasoning_engine._add_rule_safely(weakening_rule)

# Integration function for compatibility
def integrate_metta_powered_generation(func: Union[Callable, str],
                                     strategies: Optional[List] = None) -> List[Dict[str, Any]]:
    """
    Integration function for the MeTTa-powered generation system.
    """
    generator = MeTTaPoweredModularDonorGenerator()
    generator.load_ontology()
    return generator.generate_donors_from_function(func, strategies)

# Demo function
def demonstrate_metta_powered_generation():
    """Demonstrate the MeTTa-powered generation system."""
    print("  METTA-POWERED DONOR GENERATION DEMO")
    print("=" * 60)
    
    # Example function
    def find_max_in_range(numbers, start_idx, end_idx):
        """Find the maximum value in a list within a specific range."""
        if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
            return None
        
        max_val = numbers[start_idx]
        for i in range(start_idx + 1, end_idx):
            if numbers[i] > max_val:
                max_val = numbers[i]
        
        return max_val
    
    print(f"\n  Testing MeTTa-Powered Generation: {find_max_in_range.__name__}")
    print("-" * 50)
    
    try:
        candidates = integrate_metta_powered_generation(find_max_in_range)
        
        print(f"  Generated {len(candidates)} candidates using MeTTa reasoning")
        
        # Show top candidates
        for i, candidate in enumerate(candidates[:3], 1):
            print(f"\n  {i}. {candidate['name']}")
            print(f"     Strategy: {candidate['strategy']}")
            print(f"     MeTTa Score: {candidate.get('metta_score', 'N/A'):.3f}")
            print(f"     Final Score: {candidate['final_score']:.3f}")
            print(f"     MeTTa Derivation: {', '.join(candidate.get('metta_derivation', []))}")
            
            # Show MeTTa reasoning trace
            if candidate.get('metta_reasoning_trace'):
                print(f"     MeTTa Reasoning: {', '.join(candidate['metta_reasoning_trace'])}")
            
            # Show code preview
            code_lines = candidate['code'].split('\n')[:6]
            print(f"     Code preview:")
            for line in code_lines:
                if line.strip():
                    print(f"       {line}")
    
    except Exception as e:
        print(f"  Error testing MeTTa-powered generation: {e}")
        import traceback
        traceback.print_exc()
    
    return True

def test_basic_evolution():
    """Quick test of the basic evolution system"""
    print("Testing Basic Evolution System")
    print("=" * 50)
    
    # Test function
    def find_max_in_range(numbers, start_idx, end_idx):
        """Find the maximum value in a list within a specific range."""
        if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
            return None
        
        max_val = numbers[start_idx]
        for i in range(start_idx + 1, end_idx):
            if numbers[i] > max_val:
                max_val = numbers[i]
        
        return max_val
    
    try:
        # Initialize generator with evolution
        generator = MeTTaPoweredModularDonorGenerator(enable_evolution=True)
        
        # Generate evolutionary donors
        print(f"\nTesting evolution on: {find_max_in_range.__name__}")
        candidates = generator.generate_donors_with_evolution(find_max_in_range)
        
        print(f"\nResults:")
        for i, candidate in enumerate(candidates, 1):
            print(f"\n{i}. {candidate['name']}")
            print(f"   Fitness: {candidate['confidence']:.3f}")
            print(f"   Gene types: {candidate['evolution_metadata']['gene_types']}")
            print(f"   Generation: {candidate['evolution_metadata']['generation']}")
        
        if candidates:
            print(f"\nBest candidate:")
            best = candidates[0]
            print(f"   Name: {best['name']}")
            print(f"   Fitness: {best['confidence']:.3f}")
            print(f"   Code preview:")
            for line_num, line in enumerate(best['code'].split('\n')[:10], 1):
                print(f"     {line_num:2d}: {line}")
        
        return True
        
    except Exception as e:
        print(f"Evolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Add this to enable command line testing
if __name__ == "__main__":
    test_basic_evolution()