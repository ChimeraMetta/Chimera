#!/usr/bin/env python3
"""
MeTTa-Guided Algorithm Transformation Generator
Uses MeTTa reasoning to determine and generate algorithm transformations
"""

from typing import List, Dict, Optional
from metta_generator.base import BaseDonorGenerator, GenerationContext, DonorCandidate, GenerationStrategy

class AlgorithmTransformationGenerator(BaseDonorGenerator):
    """Generator that uses MeTTa ontology to guide algorithm transformations."""
    
    def __init__(self):
        self.metta_space = None  # Will be set from context
        self._current_context = None  # For fallback methods
        
        # Basic template mappings for code generation
        self.transformation_templates = {
            "iterative_to_recursive": self._create_recursive_template,
            "recursive_to_iterative": self._create_iterative_template,
            "imperative_to_functional": self._create_functional_template,
            "sequential_to_parallel": self._create_parallel_template,
            "eager_to_lazy": self._create_lazy_template
        }
    
    def can_generate(self, context: GenerationContext, strategy) -> bool:
        """Check if this generator can handle the given context and strategy."""
        if hasattr(strategy, 'value'):
            strategy_name = strategy.value
        else:
            strategy_name = str(strategy)
            
        if strategy_name != "algorithm_transformation":
            return False
        
        self.metta_space = context.metta_space
        
        # Use MeTTa reasoning to determine if transformations are applicable
        applicable_transformations = self._query_metta_for_applicable_transformations(context)
        
        can_generate = len(applicable_transformations) > 0
        
        if can_generate:
            print(f"      MeTTa-guided AlgorithmTransformationGenerator: CAN generate")
            print(f"        MeTTa suggested transformations: {applicable_transformations}")
        else:
            print(f"      MeTTa-guided AlgorithmTransformationGenerator: cannot generate")
            
        return can_generate
    
    def generate_candidates(self, context: GenerationContext, strategy) -> List[DonorCandidate]:
        """Generate algorithm transformation candidates using MeTTa reasoning."""
        candidates = []
        
        # Store context for fallback methods
        self._current_context = context
        
        # Query MeTTa for applicable transformations
        transformations = self._query_metta_for_applicable_transformations(context)
        
        print(f"        MeTTa identified {len(transformations)} applicable transformations: {transformations}")
        
        for transformation_name in transformations:
            # Query MeTTa for transformation details
            transformation_guidance = self._query_metta_for_transformation_guidance(
                context, transformation_name
            )
            
            print(f"        Processing {transformation_name} with guidance: {transformation_guidance}")
            
            candidate = self._create_metta_guided_candidate(
                context, transformation_name, transformation_guidance
            )
            
            if candidate:
                candidates.append(candidate)
                print(f"         ✓ Created MeTTa-guided {transformation_name} candidate")
            else:
                print(f"         ✗ Failed to create {transformation_name} candidate")
        
        # Clean up context reference
        self._current_context = None
        
        return candidates
    
    def get_supported_strategies(self) -> List:
        """Get list of strategies this generator supports."""
        return ["algorithm_transformation"]
    
    def _query_metta_for_applicable_transformations(self, context: GenerationContext) -> List[str]:
        """Use MeTTa reasoning to determine applicable transformations."""
        applicable = []
        
        # Store context for use in fallback methods
        self._current_context = context
        
        # Query the MeTTa ontology for algorithm transformation rules
        func_name = context.function_name
        
        print(f"        Querying MeTTa ontology for applicable transformations for {func_name}")
        
        # Check for iterative to recursive transformation
        safety_result = self._query_metta_rule(f"(algorithm-transformation-safe {func_name} iterative-to-recursive)")
        if safety_result and safety_result != "unsafe":
            applicable.append("iterative_to_recursive")
            print(f"          ✓ iterative_to_recursive: {safety_result}")
        else:
            print(f"          ✗ iterative_to_recursive: {safety_result or 'not applicable'}")
        
        # Check for recursive to iterative transformation  
        safety_result = self._query_metta_rule(f"(algorithm-transformation-safe {func_name} recursive-to-iterative)")
        if safety_result and safety_result != "unsafe":
            applicable.append("recursive_to_iterative")
            print(f"          ✓ recursive_to_iterative: {safety_result}")
        else:
            print(f"          ✗ recursive_to_iterative: {safety_result or 'not applicable'}")
        
        # Check for imperative to functional transformation
        safety_result = self._query_metta_rule(f"(algorithm-transformation-safe {func_name} imperative-to-functional)")
        if safety_result and safety_result != "unsafe":
            applicable.append("imperative_to_functional")
            print(f"          ✓ imperative_to_functional: {safety_result}")
        else:
            print(f"          ✗ imperative_to_functional: {safety_result or 'not applicable'}")
        
        # Check for sequential to parallel transformation
        safety_result = self._query_metta_rule(f"(algorithm-transformation-safe {func_name} sequential-to-parallel)")
        if safety_result and safety_result != "unsafe":
            applicable.append("sequential_to_parallel")
            print(f"          ✓ sequential_to_parallel: {safety_result}")
        else:
            print(f"          ✗ sequential_to_parallel: {safety_result or 'not applicable'}")
        
        # Check for eager to lazy transformation
        applicability_result = self._query_metta_rule(f"(algorithm-transformation-applicable {func_name} eager-to-lazy)")
        if applicability_result and applicability_result != "not-applicable":
            applicable.append("eager_to_lazy")
            print(f"          ✓ eager_to_lazy: {applicability_result}")
        else:
            print(f"          ✗ eager_to_lazy: {applicability_result or 'not applicable'}")
        
        # If MeTTa doesn't find anything, use fallback pattern detection
        if not applicable:
            print(f"        No MeTTa transformations found, using fallback pattern detection")
            applicable = self._fallback_pattern_detection_full(context)
            print(f"        Fallback found: {applicable}")
        
        return applicable
    
    def _query_metta_rule(self, query: str) -> Optional[str]:
        """Query the MeTTa space for a specific rule."""
        try:
            if not self.metta_space:
                return None
            
            # Use the actual MeTTa space to query rules
            # First, add the query as an atom to check
            from hyperon import MeTTa, E, S
            
            # Parse the query and execute it
            if "algorithm-transformation-safe" in query:
                return self._check_transformation_safety(query)
            elif "algorithm-transformation-applicable" in query:
                return self._check_transformation_applicability(query)
            elif "transformation-complexity-impact" in query:
                return self._get_complexity_impact_from_metta(query)
            elif "has-iterative-pattern" in query:
                return self._check_iterative_pattern_in_metta(query)
            elif "has-recursive-pattern" in query:
                return self._check_recursive_pattern_in_metta(query)
            elif "no-side-effects" in query:
                return self._check_side_effects_in_metta(query)
            
            # Try direct query execution
            try:
                result = self.metta_space.run(f"!({query})")
                if result and len(result) > 0:
                    return str(result[0])
            except Exception as e:
                print(f"        Direct MeTTa query failed: {e}")
            
            return None
        except Exception as e:
            print(f"        MeTTa query failed: {e}")
            return None
    
    def _check_transformation_safety(self, query: str) -> Optional[str]:
        """Check if a transformation is safe using MeTTa rules."""
        try:
            # Extract function name and transformation from query
            import re
            match = re.search(r'\(algorithm-transformation-safe\s+(\w+)\s+([^)]+)\)', query)
            if not match:
                return None
            
            func_name = match.group(1)
            transformation = match.group(2)
            
            # Query MeTTa ontology for safety rules
            safety_conditions = {
                "iterative-to-recursive": [
                    f"(has-iterative-pattern {func_name})",
                    f"(has-clear-termination {func_name})",
                    f"(no-complex-state {func_name})"
                ],
                "recursive-to-iterative": [
                    f"(has-recursive-pattern {func_name})",
                    f"(tail-recursive {func_name})",
                    f"(simple-recursive-structure {func_name})"
                ],
                "imperative-to-functional": [
                    f"(no-side-effects {func_name})",
                    f"(deterministic-behavior {func_name})",
                    f"(composable-operations {func_name})"
                ],
                "sequential-to-parallel": [
                    f"(independent-iterations {func_name})",
                    f"(no-shared-state {func_name})",
                    f"(commutative-operations {func_name})"
                ]
            }
            
            conditions = safety_conditions.get(transformation, [])
            if not conditions:
                return "unknown"
            
            # Check each condition using MeTTa
            all_satisfied = True
            for condition in conditions:
                try:
                    result = self.metta_space.run(f"!({condition})")
                    if not result or len(result) == 0:
                        # If direct query fails, use pattern-based fallback
                        if not self._fallback_condition_check(condition, func_name):
                            all_satisfied = False
                            break
                except:
                    # Fallback to pattern-based checking
                    if not self._fallback_condition_check(condition, func_name):
                        all_satisfied = False
                        break
            
            return "safe" if all_satisfied else "unsafe"
            
        except Exception as e:
            print(f"        Safety check failed: {e}")
            return "unknown"
    
    def _check_transformation_applicability(self, query: str) -> Optional[str]:
        """Check if a transformation is applicable using MeTTa rules."""
        try:
            import re
            match = re.search(r'\(algorithm-transformation-applicable\s+(\w+)\s+([^)]+)\)', query)
            if not match:
                return None
            
            func_name = match.group(1)
            transformation = match.group(2)
            
            # Query MeTTa for applicability rules
            applicability_rules = {
                "eager-to-lazy": f"(supports-lazy-evaluation {func_name})",
                "divide-and-conquer": f"(divisible-problem {func_name})",
                "dynamic-programming": f"(has-overlapping-subproblems {func_name})",
                "greedy-approach": f"(has-greedy-choice-property {func_name})",
                "backtracking": f"(has-search-space {func_name})"
            }
            
            rule = applicability_rules.get(transformation)
            if not rule:
                return "unknown"
            
            try:
                result = self.metta_space.run(f"!({rule})")
                if result and len(result) > 0:
                    return "applicable"
            except:
                pass
            
            # Fallback to pattern-based checking
            return "applicable" if self._fallback_applicability_check(transformation, func_name) else "not-applicable"
            
        except Exception as e:
            print(f"        Applicability check failed: {e}")
            return "unknown"
    
    def _get_complexity_impact_from_metta(self, query: str) -> Optional[str]:
        """Get complexity impact from MeTTa ontology."""
        try:
            import re
            match = re.search(r'\(transformation-complexity-impact\s+([^)]+)\s+\$impact\)', query)
            if not match:
                return None
            
            transformation = match.group(1)
            
            # Query MeTTa for complexity rules
            complexity_query = f"(transformation-complexity-impact {transformation} $impact)"
            try:
                result = self.metta_space.run(f"!({complexity_query})")
                if result and len(result) > 0:
                    return str(result[0])
            except:
                pass
            
            # Fallback to predefined mappings from ontology
            complexity_mappings = {
                "iterative-to-recursive": "same",
                "recursive-to-iterative": "improved",
                "imperative-to-functional": "same", 
                "functional-to-imperative": "same",
                "sequential-to-parallel": "speedup",
                "eager-to-lazy": "space-optimized",
                "divide-and-conquer": "logarithmic-improvement",
                "dynamic-programming": "time-optimized",
                "greedy-approach": "linear-time",
                "backtracking": "exponential-worst-case"
            }
            
            return complexity_mappings.get(transformation, "same")
            
        except Exception as e:
            print(f"        Complexity impact query failed: {e}")
            return "same"
    
    def _check_iterative_pattern_in_metta(self, query: str) -> Optional[str]:
        """Check for iterative pattern using MeTTa rules."""
        try:
            import re
            match = re.search(r'\(has-iterative-pattern\s+(\w+)\)', query)
            if not match:
                return None
            
            func_name = match.group(1)
            
            # Query MeTTa for iterative pattern detection
            conditions = [
                f"(has-loop-structure {func_name})",
                f"(not (calls-self {func_name}))",
                f"(sequential-processing {func_name})"
            ]
            
            for condition in conditions:
                try:
                    result = self.metta_space.run(f"!({condition})")
                    if not result or len(result) == 0:
                        # Use fallback pattern detection
                        if not self._fallback_pattern_detection_check(condition, func_name):
                            return None
                except:
                    if not self._fallback_pattern_detection_check(condition, func_name):
                        return None
            
            return "true"
            
        except Exception as e:
            print(f"        Iterative pattern check failed: {e}")
            return None
    
    def _check_recursive_pattern_in_metta(self, query: str) -> Optional[str]:
        """Check for recursive pattern using MeTTa rules."""
        try:
            import re
            match = re.search(r'\(has-recursive-pattern\s+(\w+)\)', query)
            if not match:
                return None
            
            func_name = match.group(1)
            
            # Query MeTTa for recursive pattern detection
            conditions = [
                f"(calls-self {func_name})",
                f"(has-base-case {func_name})", 
                f"(reduces-problem-size {func_name})"
            ]
            
            for condition in conditions:
                try:
                    result = self.metta_space.run(f"!({condition})")
                    if not result or len(result) == 0:
                        if not self._fallback_pattern_detection_check(condition, func_name):
                            return None
                except:
                    if not self._fallback_pattern_detection_check(condition, func_name):
                        return None
            
            return "true"
            
        except Exception as e:
            print(f"        Recursive pattern check failed: {e}")
            return None
    
    def _check_side_effects_in_metta(self, query: str) -> Optional[str]:
        """Check for side effects using MeTTa rules."""
        try:
            import re
            match = re.search(r'\(no-side-effects\s+(\w+)\)', query)
            if not match:
                return None
            
            func_name = match.group(1)
            
            # Query MeTTa for side effect analysis
            side_effect_checks = [
                f"(not (modifies-global-state {func_name}))",
                f"(not (modifies-input-parameters {func_name}))",
                f"(not (performs-io-operations {func_name}))"
            ]
            
            for check in side_effect_checks:
                try:
                    result = self.metta_space.run(f"!({check})")
                    if not result or len(result) == 0:
                        if not self._fallback_side_effect_check(check, func_name):
                            return None
                except:
                    if not self._fallback_side_effect_check(check, func_name):
                        return None
            
            return "true"
            
        except Exception as e:
            print(f"        Side effect check failed: {e}")
            return None
    
    def _fallback_condition_check(self, condition: str, func_name: str) -> bool:
        """Fallback condition checking when MeTTa queries fail."""
        try:
            # Get the original code from context if available
            if hasattr(self, '_current_context') and self._current_context:
                code = self._current_context.original_code
            else:
                # Try to infer from function name patterns
                code = ""
            
            if "has-iterative-pattern" in condition:
                return any(pattern in code for pattern in ["for ", "while "])
            elif "has-recursive-pattern" in condition:
                return func_name in code.replace(f"def {func_name}", "")
            elif "has-clear-termination" in condition:
                return any(pattern in code for pattern in ["break", "return", ">=", "<=", ">", "<"])
            elif "no-complex-state" in condition:
                # Simple heuristic: no complex data structures being modified
                return not any(pattern in code for pattern in ["dict[", "list[", "set[", ".append(", ".update("])
            elif "tail-recursive" in condition:
                # Check if recursive calls are at the end of functions
                return "return " + func_name in code
            elif "simple-recursive-structure" in condition:
                # Check for straightforward recursive structure
                return code.count(func_name) <= 3  # Definition + 1-2 recursive calls
            elif "no-side-effects" in condition:
                return not any(pattern in code for pattern in ["global ", "print(", "open(", "write(", "input("])
            elif "deterministic-behavior" in condition:
                return not any(pattern in code for pattern in ["random", "time", "datetime", "uuid"])
            elif "composable-operations" in condition:
                return not any(pattern in code for pattern in ["global ", "nonlocal "])
            elif "independent-iterations" in condition:
                # Check if loop iterations don't depend on each other
                return "for " in code and not any(pattern in code for pattern in ["previous", "last", "accumulate"])
            elif "no-shared-state" in condition:
                return not any(pattern in code for pattern in ["global ", "class ", "self."])
            elif "commutative-operations" in condition:
                # Check for operations that can be reordered
                return any(pattern in code for pattern in ["max", "min", "sum", "+", "*", "and", "or"])
            
            return True  # Default to allowing transformation
            
        except Exception as e:
            print(f"        Fallback condition check failed: {e}")
            return False
    
    def _fallback_applicability_check(self, transformation: str, func_name: str) -> bool:
        """Fallback applicability checking when MeTTa queries fail."""
        try:
            # Basic pattern-based applicability checks
            if hasattr(self, '_current_context') and self._current_context:
                code = self._current_context.original_code
            else:
                code = ""
            
            if transformation == "eager-to-lazy":
                # Check if function processes collections that could be streamed
                return any(pattern in code for pattern in ["for ", "list(", "map(", "filter("])
            elif transformation == "divide-and-conquer":
                # Check if problem can be divided
                return any(pattern in code for pattern in ["len(", "range(", "sort", "search"])
            elif transformation == "dynamic-programming":
                # Check for recursive structure with potential overlapping subproblems
                return func_name in code and "if " in code
            elif transformation == "greedy-approach":
                # Check for optimization problems
                return any(pattern in func_name.lower() for pattern in ["max", "min", "best", "optimal"])
            elif transformation == "backtracking":
                # Check for search or constraint satisfaction problems
                return any(pattern in func_name.lower() for pattern in ["find", "search", "solve", "valid"])
            
            return True  # Default to allowing transformation
            
        except Exception as e:
            print(f"        Fallback applicability check failed: {e}")
            return False
    
    def _fallback_pattern_detection_check(self, condition: str, func_name: str) -> bool:
        """Fallback pattern detection when MeTTa queries fail."""
        try:
            if hasattr(self, '_current_context') and self._current_context:
                code = self._current_context.original_code
            else:
                code = ""
            
            if "has-loop-structure" in condition:
                return any(pattern in code for pattern in ["for ", "while "])
            elif "calls-self" in condition:
                return func_name in code.replace(f"def {func_name}", "")
            elif "sequential-processing" in condition:
                return "for " in code and any(pattern in code for pattern in ["next", "append", "process"])
            elif "has-base-case" in condition:
                return any(pattern in code for pattern in ["if ", "return", "break"])
            elif "reduces-problem-size" in condition:
                # Check for patterns that suggest problem size reduction
                return any(pattern in code for pattern in ["len(", "//", "slice", "[:", "range("])
            
            return True
            
        except Exception as e:
            print(f"        Fallback pattern detection failed: {e}")
            return False
    
    def _fallback_side_effect_check(self, check: str, func_name: str) -> bool:
        """Fallback side effect checking when MeTTa queries fail."""
        try:
            if hasattr(self, '_current_context') and self._current_context:
                code = self._current_context.original_code
            else:
                code = ""
            
            if "modifies-global-state" in check:
                return not any(pattern in code for pattern in ["global ", "nonlocal "])
            elif "modifies-input-parameters" in check:
                # Check for in-place modifications
                return not any(pattern in code for pattern in [".append(", ".extend(", ".remove(", ".clear(", ".sort("])
            elif "performs-io-operations" in check:
                return not any(pattern in code for pattern in ["print(", "input(", "open(", "read(", "write("])
            
            return True
            
        except Exception as e:
            print(f"        Fallback side effect check failed: {e}")
            return False
    
    def _fallback_pattern_detection_full(self, context: GenerationContext) -> List[str]:
        """Comprehensive fallback pattern detection when MeTTa rules are not available."""
        applicable = []
        code = context.original_code
        func_name = context.function_name
        
        print(f"        Running comprehensive fallback pattern detection")
        
        # Check for iterative patterns
        if any(pattern in code for pattern in ["for ", "while "]):
            applicable.append("iterative_to_recursive")
            applicable.append("imperative_to_functional")
            print(f"          Found iterative patterns")
        
        # Check for recursive patterns
        if func_name in code.replace(f"def {func_name}", ""):
            applicable.append("recursive_to_iterative")
            print(f"          Found recursive patterns")
        
        # Check for functional transformation opportunities
        if not any(pattern in code for pattern in ["global ", "print(", "input(", "open("]):
            if "imperative_to_functional" not in applicable:
                applicable.append("imperative_to_functional")
            print(f"          Found functional transformation opportunity")
        
        # Check for parallelization opportunities
        if ("for " in code and 
            not any(pattern in code for pattern in ["global ", "nonlocal ", "self."]) and
            any(pattern in code for pattern in ["max", "min", "sum", "+", "*"])):
            applicable.append("sequential_to_parallel")
            print(f"          Found parallelization opportunity")
        
        # Check for lazy evaluation opportunities
        if any(pattern in code for pattern in ["list(", "for ", "range("]):
            applicable.append("eager_to_lazy")
            print(f"          Found lazy evaluation opportunity")
        
        return applicable
    
    def _query_metta_for_transformation_guidance(self, context: GenerationContext, 
                                               transformation_name: str) -> Dict[str, str]:
        """Query MeTTa for specific guidance on how to perform the transformation."""
        guidance = {}
        func_name = context.function_name
        
        print(f"          Querying MeTTa for {transformation_name} guidance")
        
        # Query for complexity impact
        complexity_result = self._query_metta_rule(
            f"(transformation-complexity-impact {transformation_name} $impact)"
        )
        if complexity_result:
            guidance["complexity_impact"] = complexity_result
            print(f"            Complexity impact: {complexity_result}")
        else:
            guidance["complexity_impact"] = "same"
            print(f"            Complexity impact: same (default)")
        
        # Query for safety conditions
        safety_result = self._query_metta_rule(
            f"(algorithm-transformation-safe {func_name} {transformation_name})"
        )
        guidance["is_safe"] = safety_result not in [None, "unsafe", "unknown"]
        print(f"            Safety: {guidance['is_safe']} (result: {safety_result})")
        
        # Query for specific transformation requirements based on type
        if transformation_name == "iterative_to_recursive":
            guidance.update(self._get_recursive_transformation_guidance(context))
        elif transformation_name == "recursive_to_iterative":
            guidance.update(self._get_iterative_transformation_guidance(context))
        elif transformation_name == "imperative_to_functional":
            guidance.update(self._get_functional_transformation_guidance(context))
        elif transformation_name == "sequential_to_parallel":
            guidance.update(self._get_parallel_transformation_guidance(context))
        elif transformation_name == "eager_to_lazy":
            guidance.update(self._get_lazy_transformation_guidance(context))
        
        # Query for additional properties
        try:
            # Check for deterministic behavior
            deterministic_result = self._query_metta_rule(f"(deterministic-behavior {func_name})")
            guidance["deterministic"] = bool(deterministic_result)
            
            # Check for side effects
            side_effects_result = self._query_metta_rule(f"(no-side-effects {func_name})")
            guidance["no_side_effects"] = bool(side_effects_result)
            
            # Check for thread safety (for parallel transformations)
            if transformation_name == "sequential_to_parallel":
                thread_safe_result = self._query_metta_rule(f"(thread-safe {func_name})")
                guidance["thread_safe"] = bool(thread_safe_result)
        
        except Exception as e:
            print(f"            Additional property queries failed: {e}")
        
        print(f"            Final guidance: {guidance}")
        return guidance
    
    def _get_recursive_transformation_guidance(self, context: GenerationContext) -> Dict[str, str]:
        """Get MeTTa guidance for recursive transformations."""
        func_name = context.function_name
        code = context.original_code
        
        guidance = {}
        
        # Analyze termination condition
        if "range(" in code:
            guidance["termination_condition"] = "index >= end"
            guidance["base_case"] = "return None"
            guidance["recursive_step"] = "process_and_recurse"
        elif "len(" in code:
            guidance["termination_condition"] = "len(data) == 0"
            guidance["base_case"] = "return base_value"
            guidance["recursive_step"] = "process_first_and_recurse_rest"
        else:
            guidance["termination_condition"] = "base_case_reached"
            guidance["base_case"] = "return result"
            guidance["recursive_step"] = "divide_and_conquer"
        
        # Analyze accumulation pattern
        if any(pattern in code for pattern in ["max(", "maximum"]):
            guidance["accumulation_type"] = "maximum"
            guidance["initial_value"] = "float('-inf')"
            guidance["combine_operation"] = "max"
        elif any(pattern in code for pattern in ["min(", "minimum"]):
            guidance["accumulation_type"] = "minimum"
            guidance["initial_value"] = "float('inf')"
            guidance["combine_operation"] = "min"
        elif any(pattern in code for pattern in ["sum", "+="]):
            guidance["accumulation_type"] = "sum"
            guidance["initial_value"] = "0"
            guidance["combine_operation"] = "add"
        elif any(pattern in code for pattern in ["count", "len"]):
            guidance["accumulation_type"] = "count"
            guidance["initial_value"] = "0"
            guidance["combine_operation"] = "increment"
        else:
            guidance["accumulation_type"] = "generic"
            guidance["initial_value"] = "None"
            guidance["combine_operation"] = "combine"
        
        # Check for tail recursion potential
        if "return " in code and func_name not in code.split("return ")[-1].split("\n")[0]:
            guidance["tail_recursive_potential"] = "high"
        else:
            guidance["tail_recursive_potential"] = "low"
        
        return guidance
    
    def _get_iterative_transformation_guidance(self, context: GenerationContext) -> Dict[str, str]:
        """Get MeTTa guidance for iterative transformations."""
        func_name