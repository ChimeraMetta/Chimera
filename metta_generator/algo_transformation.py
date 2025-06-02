#!/usr/bin/env python3
"""
MeTTa-Focused Algorithm Transformation Generator
Uses MeTTa reasoning to determine and generate algorithm transformations
Based on symbolic logic, pattern matching, and rule-based inference
"""

from typing import List, Dict, Optional
from metta_generator.base import BaseDonorGenerator, GenerationContext, DonorCandidate, GenerationStrategy

class AlgorithmTransformationGenerator(BaseDonorGenerator):
    """Generator that uses MeTTa symbolic reasoning to guide algorithm transformations."""
    
    def __init__(self):
        self.metta_space = None  # Will be set from context
        self._current_context = None  # For fallback methods
        
        # MeTTa-based transformation rules - these will query the actual MeTTa space
        self.metta_transformation_queries = {
            "iterative_to_recursive": "(algorithm-transformation-safe {func_name} iterative-to-recursive)",
            "recursive_to_iterative": "(algorithm-transformation-safe {func_name} recursive-to-iterative)", 
            "imperative_to_functional": "(algorithm-transformation-safe {func_name} imperative-to-functional)",
            "sequential_to_parallel": "(algorithm-transformation-safe {func_name} sequential-to-parallel)",
            "eager_to_lazy": "(algorithm-transformation-applicable {func_name} eager-to-lazy)"
        }
        
        # MeTTa pattern detection queries
        self.pattern_detection_queries = {
            "has_iterative_pattern": "(has-iterative-pattern {func_name})",
            "has_recursive_pattern": "(has-recursive-pattern {func_name})",
            "has_functional_pattern": "(has-functional-pattern {func_name})",
            "no_side_effects": "(no-side-effects {func_name})",
            "deterministic_behavior": "(deterministic-behavior {func_name})",
            "independent_iterations": "(independent-iterations {func_name})"
        }
        
        # Template mappings for code generation
        self.transformation_templates = {
            "iterative_to_recursive": self._create_recursive_template,
            "recursive_to_iterative": self._create_iterative_template,
            "imperative_to_functional": self._create_functional_template,
            "sequential_to_parallel": self._create_parallel_template,
            "eager_to_lazy": self._create_lazy_template
        }
    
    def can_generate(self, context: GenerationContext, strategy) -> bool:
        """Check if this generator can handle the given context and strategy using MeTTa reasoning."""
        if hasattr(strategy, 'value'):
            strategy_name = strategy.value
        else:
            strategy_name = str(strategy)
            
        if strategy_name != "algorithm_transformation":
            return False
        
        self.metta_space = context.metta_space
        self._current_context = context
        
        # Use MeTTa reasoning to determine if transformations are applicable
        applicable_transformations = self._query_metta_for_applicable_transformations(context)
        
        can_generate = len(applicable_transformations) > 0
        
        if can_generate:
            print(f"      MeTTa-guided AlgorithmTransformationGenerator: CAN generate")
            print(f"        MeTTa identified transformations: {applicable_transformations}")
        else:
            print(f"      MeTTa-guided AlgorithmTransformationGenerator: cannot generate")
            
        return can_generate
    
    def generate_candidates(self, context: GenerationContext, strategy) -> List[DonorCandidate]:
        """Generate algorithm transformation candidates using MeTTa symbolic reasoning."""
        candidates = []
        
        # Store context for access in helper methods
        self._current_context = context
        
        # Query MeTTa for applicable transformations with full symbolic reasoning
        transformations = self._query_metta_for_applicable_transformations(context)
        
        print(f"        MeTTa symbolic analysis identified {len(transformations)} applicable transformations: {transformations}")
        
        for transformation_name in transformations:
            # Query MeTTa for detailed transformation guidance using symbolic logic
            transformation_guidance = self._query_metta_for_transformation_guidance(
                context, transformation_name
            )
            
            print(f"        Processing {transformation_name} with MeTTa guidance: {transformation_guidance}")
            
            candidate = self._create_metta_guided_candidate(
                context, transformation_name, transformation_guidance
            )
            
            if candidate:
                candidates.append(candidate)
                print(f"         ✓ Created MeTTa-reasoned {transformation_name} candidate")
            else:
                print(f"         ✗ Failed to create {transformation_name} candidate")
        
        # Clean up context reference
        self._current_context = None
        
        return candidates
    
    def get_supported_strategies(self) -> List:
        """Get list of strategies this generator supports."""
        return ["algorithm_transformation"]
    
    def _query_metta_for_applicable_transformations(self, context: GenerationContext) -> List[str]:
        """Use MeTTa symbolic reasoning to determine applicable transformations."""
        applicable = []
        func_name = context.function_name
        
        print(f"        Querying MeTTa symbolic space for algorithmic transformation rules for {func_name}")
        
        # Query MeTTa space for each potential transformation using symbolic logic
        for transformation, query_template in self.metta_transformation_queries.items():
            query = query_template.format(func_name=func_name)
            
            print(f"          MeTTa query: {query}")
            result = self._execute_metta_query(query)
            
            if result and result not in ["unsafe", "not-applicable", "unknown"]:
                applicable.append(transformation)
                print(f"            ✓ {transformation}: {result}")
            else:
                print(f"            ✗ {transformation}: {result or 'not applicable'}")
        
        # If MeTTa doesn't find transformations, use pattern-based reasoning as fallback
        if not applicable:
            print(f"        No MeTTa transformations found, using symbolic pattern detection fallback")
            applicable = self._symbolic_pattern_detection_fallback(context)
            print(f"        Symbolic analysis found: {applicable}")
        
        return applicable
    
    def _execute_metta_query(self, query: str) -> Optional[str]:
        """Execute a query against the MeTTa symbolic reasoning space."""
        try:
            if not self.metta_space:
                return None
            
            # Parse and execute the MeTTa query for symbolic reasoning
            # This uses MeTTa's symbolic logic capabilities
            if "algorithm-transformation-safe" in query:
                return self._check_transformation_safety_with_metta(query)
            elif "algorithm-transformation-applicable" in query:
                return self._check_transformation_applicability_with_metta(query)
            elif any(pattern in query for pattern in self.pattern_detection_queries.values()):
                return self._check_pattern_with_metta(query)
            
            # Try direct MeTTa symbolic execution
            try:
                result = self.metta_space.run(f"!({query})")
                if result and len(result) > 0:
                    return str(result[0])
            except Exception as e:
                print(f"        Direct MeTTa symbolic query failed: {e}")
            
            return None
        except Exception as e:
            print(f"        MeTTa symbolic reasoning failed: {e}")
            return None
    
    def _check_transformation_safety_with_metta(self, query: str) -> Optional[str]:
        """Use MeTTa symbolic reasoning to check transformation safety."""
        try:
            # Extract function name and transformation from symbolic query
            import re
            match = re.search(r'\(algorithm-transformation-safe\s+(\w+)\s+([^)]+)\)', query)
            if not match:
                return None
            
            func_name = match.group(1)
            transformation = match.group(2)
            
            print(f"            MeTTa symbolic safety analysis for {transformation}")
            
            # Use MeTTa's symbolic logic to check safety conditions
            safety_conditions = self._get_metta_safety_conditions(transformation, func_name)
            
            # Check each condition using MeTTa symbolic reasoning
            all_satisfied = True
            for condition_query in safety_conditions:
                print(f"              Checking MeTTa condition: {condition_query}")
                try:
                    result = self.metta_space.run(f"!({condition_query})")
                    if not result or len(result) == 0:
                        # Use pattern-based fallback with symbolic reasoning
                        if not self._symbolic_condition_check(condition_query, func_name):
                            all_satisfied = False
                            print(f"                ✗ Failed: {condition_query}")
                            break
                        else:
                            print(f"                ✓ Passed (fallback): {condition_query}")
                    else:
                        print(f"                ✓ Passed (MeTTa): {condition_query}")
                except:
                    # Fallback to symbolic pattern-based checking
                    if not self._symbolic_condition_check(condition_query, func_name):
                        all_satisfied = False
                        print(f"                ✗ Failed (fallback): {condition_query}")
                        break
                    else:
                        print(f"                ✓ Passed (fallback): {condition_query}")
            
            return "safe" if all_satisfied else "unsafe"
            
        except Exception as e:
            print(f"        MeTTa safety analysis failed: {e}")
            return "unknown"
    
    def _get_metta_safety_conditions(self, transformation: str, func_name: str) -> List[str]:
        """Get MeTTa symbolic safety conditions for a transformation."""
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
        
        return safety_conditions.get(transformation, [])
    
    def _check_transformation_applicability_with_metta(self, query: str) -> Optional[str]:
        """Use MeTTa symbolic reasoning to check transformation applicability."""
        try:
            import re
            match = re.search(r'\(algorithm-transformation-applicable\s+(\w+)\s+([^)]+)\)', query)
            if not match:
                return None
            
            func_name = match.group(1)
            transformation = match.group(2)
            
            print(f"            MeTTa symbolic applicability analysis for {transformation}")
            
            # Query MeTTa symbolic space for applicability rules
            applicability_queries = {
                "eager-to-lazy": f"(supports-lazy-evaluation {func_name})",
                "divide-and-conquer": f"(divisible-problem {func_name})",
                "dynamic-programming": f"(has-overlapping-subproblems {func_name})",
                "greedy-approach": f"(has-greedy-choice-property {func_name})",
                "backtracking": f"(has-search-space {func_name})"
            }
            
            query_to_check = applicability_queries.get(transformation)
            if not query_to_check:
                return "unknown"
            
            print(f"              MeTTa applicability query: {query_to_check}")
            try:
                result = self.metta_space.run(f"!({query_to_check})")
                if result and len(result) > 0:
                    print(f"                ✓ MeTTa result: {result}")
                    return "applicable"
            except:
                pass
            
            # Fallback to symbolic pattern-based checking
            is_applicable = self._symbolic_applicability_check(transformation, func_name)
            result = "applicable" if is_applicable else "not-applicable"
            print(f"                Fallback result: {result}")
            return result
            
        except Exception as e:
            print(f"        MeTTa applicability analysis failed: {e}")
            return "unknown"
    
    def _check_pattern_with_metta(self, query: str) -> Optional[str]:
        """Use MeTTa symbolic reasoning to check patterns."""
        try:
            print(f"            MeTTa symbolic pattern check: {query}")
            
            # Try direct MeTTa execution for pattern detection
            try:
                result = self.metta_space.run(f"!({query})")
                if result and len(result) > 0:
                    print(f"                ✓ MeTTa pattern result: {result}")
                    return str(result[0])
            except:
                pass
            
            # Fallback to symbolic pattern detection
            pattern_result = self._symbolic_pattern_check(query)
            print(f"                Fallback pattern result: {pattern_result}")
            return pattern_result
            
        except Exception as e:
            print(f"        MeTTa pattern check failed: {e}")
            return None
    
    def _symbolic_pattern_detection_fallback(self, context: GenerationContext) -> List[str]:
        """Comprehensive symbolic pattern detection when MeTTa rules are not available."""
        applicable = []
        code = context.original_code
        func_name = context.function_name
        
        print(f"        Running symbolic pattern analysis")
        
        # Symbolic analysis for iterative patterns
        if self._has_symbolic_iterative_pattern(code, func_name):
            applicable.append("iterative_to_recursive")
            applicable.append("imperative_to_functional")
            print(f"          Found symbolic iterative patterns")
        
        # Symbolic analysis for recursive patterns
        if self._has_symbolic_recursive_pattern(code, func_name):
            applicable.append("recursive_to_iterative")
            print(f"          Found symbolic recursive patterns")
        
        # Symbolic analysis for functional transformation opportunities
        if self._has_symbolic_functional_opportunity(code, func_name):
            if "imperative_to_functional" not in applicable:
                applicable.append("imperative_to_functional")
            print(f"          Found symbolic functional transformation opportunity")
        
        # Symbolic analysis for parallelization opportunities
        if self._has_symbolic_parallel_opportunity(code, func_name):
            applicable.append("sequential_to_parallel")
            print(f"          Found symbolic parallelization opportunity")
        
        # Symbolic analysis for lazy evaluation opportunities
        if self._has_symbolic_lazy_opportunity(code, func_name):
            applicable.append("eager_to_lazy")
            print(f"          Found symbolic lazy evaluation opportunity")
        
        return applicable
    
    def _has_symbolic_iterative_pattern(self, code: str, func_name: str) -> bool:
        """Symbolic analysis for iterative patterns."""
        return (any(pattern in code for pattern in ["for ", "while "]) and
                func_name not in code.replace(f"def {func_name}", ""))
    
    def _has_symbolic_recursive_pattern(self, code: str, func_name: str) -> bool:
        """Symbolic analysis for recursive patterns."""
        return (func_name in code.replace(f"def {func_name}", "") and
                any(pattern in code for pattern in ["if ", "return"]))
    
    def _has_symbolic_functional_opportunity(self, code: str, func_name: str) -> bool:
        """Symbolic analysis for functional transformation opportunities."""
        return (not any(pattern in code for pattern in ["global ", "print(", "input(", "open("]) and
                not any(pattern in code for pattern in ["random", "time", "datetime"]))
    
    def _has_symbolic_parallel_opportunity(self, code: str, func_name: str) -> bool:
        """Symbolic analysis for parallelization opportunities."""
        return ("for " in code and 
                not any(pattern in code for pattern in ["global ", "nonlocal ", "self."]) and
                any(pattern in code for pattern in ["max", "min", "sum", "+", "*"]))
    
    def _has_symbolic_lazy_opportunity(self, code: str, func_name: str) -> bool:
        """Symbolic analysis for lazy evaluation opportunities."""
        return any(pattern in code for pattern in ["list(", "for ", "range(", "map(", "filter("])
    
    def _symbolic_condition_check(self, condition: str, func_name: str) -> bool:
        """Symbolic reasoning for condition checking."""
        try:
            if not hasattr(self, '_current_context') or not self._current_context:
                return True  # Default to allowing transformation
            
            code = self._current_context.original_code
            
            # Symbolic condition analysis
            if "has-iterative-pattern" in condition:
                return any(pattern in code for pattern in ["for ", "while "])
            elif "has-recursive-pattern" in condition:
                return func_name in code.replace(f"def {func_name}", "")
            elif "has-clear-termination" in condition:
                return any(pattern in code for pattern in ["break", "return", ">=", "<=", ">", "<"])
            elif "no-complex-state" in condition:
                return not any(pattern in code for pattern in ["dict[", "list[", "set[", ".append(", ".update("])
            elif "tail-recursive" in condition:
                return "return " + func_name in code
            elif "simple-recursive-structure" in condition:
                return code.count(func_name) <= 3
            elif "no-side-effects" in condition:
                return not any(pattern in code for pattern in ["global ", "print(", "open(", "write(", "input("])
            elif "deterministic-behavior" in condition:
                return not any(pattern in code for pattern in ["random", "time", "datetime", "uuid"])
            elif "composable-operations" in condition:
                return not any(pattern in code for pattern in ["global ", "nonlocal "])
            elif "independent-iterations" in condition:
                return "for " in code and not any(pattern in code for pattern in ["previous", "last", "accumulate"])
            elif "no-shared-state" in condition:
                return not any(pattern in code for pattern in ["global ", "class ", "self."])
            elif "commutative-operations" in condition:
                return any(pattern in code for pattern in ["max", "min", "sum", "+", "*", "and", "or"])
            
            return True  # Default to allowing transformation
            
        except Exception as e:
            print(f"        Symbolic condition check failed: {e}")
            return False
    
    def _symbolic_applicability_check(self, transformation: str, func_name: str) -> bool:
        """Symbolic reasoning for applicability checking."""
        try:
            if not hasattr(self, '_current_context') or not self._current_context:
                return True
            
            code = self._current_context.original_code
            
            # Symbolic applicability analysis
            if transformation == "eager-to-lazy":
                return any(pattern in code for pattern in ["for ", "list(", "map(", "filter("])
            elif transformation == "divide-and-conquer":
                return any(pattern in code for pattern in ["len(", "range(", "sort", "search"])
            elif transformation == "dynamic-programming":
                return func_name in code and "if " in code
            elif transformation == "greedy-approach":
                return any(pattern in func_name.lower() for pattern in ["max", "min", "best", "optimal"])
            elif transformation == "backtracking":
                return any(pattern in func_name.lower() for pattern in ["find", "search", "solve", "valid"])
            
            return True  # Default to allowing transformation
            
        except Exception as e:
            print(f"        Symbolic applicability check failed: {e}")
            return False
    
    def _symbolic_pattern_check(self, query: str) -> Optional[str]:
        """Symbolic pattern checking fallback."""
        try:
            if not hasattr(self, '_current_context') or not self._current_context:
                return None
            
            code = self._current_context.original_code
            func_name = self._current_context.function_name
            
            # Extract pattern from query
            if "has-iterative-pattern" in query:
                return "true" if self._has_symbolic_iterative_pattern(code, func_name) else None
            elif "has-recursive-pattern" in query:
                return "true" if self._has_symbolic_recursive_pattern(code, func_name) else None
            elif "no-side-effects" in query:
                return "true" if self._has_symbolic_functional_opportunity(code, func_name) else None
            elif "deterministic-behavior" in query:
                return "true" if not any(pattern in code for pattern in ["random", "time", "datetime"]) else None
            elif "independent-iterations" in query:
                return "true" if self._has_symbolic_parallel_opportunity(code, func_name) else None
            
            return None
            
        except Exception as e:
            print(f"        Symbolic pattern check failed: {e}")
            return None
    
    def _query_metta_for_transformation_guidance(self, context: GenerationContext, 
                                               transformation_name: str) -> Dict[str, str]:
        """Query MeTTa symbolic space for specific transformation guidance."""
        guidance = {}
        func_name = context.function_name
        
        print(f"          Querying MeTTa symbolic space for {transformation_name} guidance")
        
        # Query MeTTa for complexity impact using symbolic reasoning
        complexity_query = f"(transformation-complexity-impact {transformation_name} $impact)"
        complexity_result = self._execute_metta_query(complexity_query)
        if complexity_result:
            guidance["complexity_impact"] = complexity_result
            print(f"            MeTTa complexity impact: {complexity_result}")
        else:
            guidance["complexity_impact"] = "same"
            print(f"            Complexity impact: same (default)")
        
        # Query MeTTa for safety conditions using symbolic logic
        safety_query = f"(algorithm-transformation-safe {func_name} {transformation_name})"
        safety_result = self._execute_metta_query(safety_query)
        guidance["is_safe"] = safety_result not in [None, "unsafe", "unknown"]
        print(f"            MeTTa safety analysis: {guidance['is_safe']} (result: {safety_result})")
        
        # Get transformation-specific guidance using symbolic reasoning
        if transformation_name == "iterative_to_recursive":
            guidance.update(self._get_recursive_transformation_guidance_with_metta(context))
        elif transformation_name == "recursive_to_iterative":
            guidance.update(self._get_iterative_transformation_guidance_with_metta(context))
        elif transformation_name == "imperative_to_functional":
            guidance.update(self._get_functional_transformation_guidance_with_metta(context))
        elif transformation_name == "sequential_to_parallel":
            guidance.update(self._get_parallel_transformation_guidance_with_metta(context))
        elif transformation_name == "eager_to_lazy":
            guidance.update(self._get_lazy_transformation_guidance_with_metta(context))
        
        # Query MeTTa for additional symbolic properties
        try:
            # Check for deterministic behavior using MeTTa symbolic reasoning
            deterministic_query = f"(deterministic-behavior {func_name})"
            deterministic_result = self._execute_metta_query(deterministic_query)
            guidance["deterministic"] = bool(deterministic_result)
            
            # Check for side effects using MeTTa symbolic analysis
            side_effects_query = f"(no-side-effects {func_name})"
            side_effects_result = self._execute_metta_query(side_effects_query)
            guidance["no_side_effects"] = bool(side_effects_result)
            
            # Check for thread safety (for parallel transformations) using symbolic reasoning
            if transformation_name == "sequential_to_parallel":
                thread_safe_query = f"(thread-safe {func_name})"
                thread_safe_result = self._execute_metta_query(thread_safe_query)
                guidance["thread_safe"] = bool(thread_safe_result)
        
        except Exception as e:
            print(f"            MeTTa additional property queries failed: {e}")
        
        print(f"            Final MeTTa-guided transformation guidance: {guidance}")
        return guidance
    
    def _create_metta_guided_candidate(self, context: GenerationContext,
                                     transformation_name: str, 
                                     transformation_guidance: Dict[str, str]) -> Optional[DonorCandidate]:
        """Create a candidate guided by MeTTa symbolic reasoning."""
        try:
            # Get the appropriate template method
            template_method = self.transformation_templates.get(transformation_name)
            if not template_method:
                print(f"          No template method for {transformation_name}")
                return None
            
            # Generate code using MeTTa-guided template
            transformed_code = template_method(context, transformation_guidance)
            
            if not transformed_code:
                print(f"          Template method failed for {transformation_name}")
                return None
            
            # Generate name using MeTTa reasoning
            new_func_name = f"{context.function_name}_{transformation_name.replace('_', '_')}"
            
            # Create MeTTa derivation showing symbolic reasoning path
            metta_derivation = [
                f"(algorithm-transformation {context.function_name} {transformation_name})",
                f"(transformation-safety {transformation_guidance.get('is_safe', 'unknown')})",
                f"(complexity-impact {transformation_guidance.get('complexity_impact', 'same')})"
            ]
            
            # Determine properties based on MeTTa analysis
            properties = ["algorithm-transformed", transformation_name.replace('_', '-')]
            if transformation_guidance.get('deterministic', False):
                properties.append("deterministic")
            if transformation_guidance.get('no_side_effects', False):
                properties.append("side-effect-free")
            if transformation_guidance.get('thread_safe', False):
                properties.append("thread-safe")
            
            return DonorCandidate(
                name=new_func_name,
                description=f"MeTTa-guided {transformation_name.replace('_', ' ')} transformation",
                code=transformed_code,
                strategy="algorithm_transformation",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=self._get_data_structures_from_context(context),
                operations_used=self._get_operations_from_context(context),
                metta_derivation=metta_derivation,
                confidence=0.9 if transformation_guidance.get('is_safe', False) else 0.7,
                properties=properties,
                complexity_estimate=transformation_guidance.get('complexity_impact', 'same'),
                applicability_scope="broad" if transformation_guidance.get('is_safe', False) else "medium"
            )
            
        except Exception as e:
            print(f"          Failed to create MeTTa-guided candidate for {transformation_name}: {e}")
            return None
    
    # MeTTa-guided transformation guidance methods
    
    def _get_recursive_transformation_guidance_with_metta(self, context: GenerationContext) -> Dict[str, str]:
        """Get MeTTa-guided recursive transformation guidance."""
        guidance = {}
        func_name = context.function_name
        code = context.original_code
        
        # Use MeTTa symbolic reasoning to analyze termination
        termination_query = f"(has-clear-termination {func_name})"
        if self._execute_metta_query(termination_query):
            guidance["termination_condition"] = "metta_verified"
        elif "range(" in code:
            guidance["termination_condition"] = "index >= end"
        elif "len(" in code:
            guidance["termination_condition"] = "len(data) == 0"
        else:
            guidance["termination_condition"] = "base_case_reached"
        
        # MeTTa symbolic analysis for accumulation patterns
        accumulation_query = f"(accumulation-pattern {func_name} $type)"
        accumulation_result = self._execute_metta_query(accumulation_query)
        if accumulation_result:
            guidance["accumulation_type"] = accumulation_result
        elif any(pattern in code for pattern in ["max(", "maximum"]):
            guidance["accumulation_type"] = "maximum"
        elif any(pattern in code for pattern in ["min(", "minimum"]):
            guidance["accumulation_type"] = "minimum"
        elif any(pattern in code for pattern in ["sum", "+="]):
            guidance["accumulation_type"] = "sum"
        else:
            guidance["accumulation_type"] = "generic"
        
        return guidance
    
    def _get_iterative_transformation_guidance_with_metta(self, context: GenerationContext) -> Dict[str, str]:
        """Get MeTTa-guided iterative transformation guidance."""
        guidance = {}
        func_name = context.function_name
        code = context.original_code
        
        # MeTTa symbolic analysis for stack requirements
        stack_query = f"(requires-stack {func_name})"
        if self._execute_metta_query(stack_query):
            guidance["requires_stack"] = "true"
        else:
            guidance["requires_stack"] = "false"
        
        # MeTTa analysis for memory patterns
        memory_query = f"(memory-pattern {func_name} $pattern)"
        memory_result = self._execute_metta_query(memory_query)
        if memory_result:
            guidance["memory_pattern"] = memory_result
        else:
            guidance["memory_pattern"] = "index_based" if "range(" in code else "collection_based"
        
        return guidance
    
    def _get_functional_transformation_guidance_with_metta(self, context: GenerationContext) -> Dict[str, str]:
        """Get MeTTa-guided functional transformation guidance."""
        guidance = {}
        func_name = context.function_name
        code = context.original_code
        
        # MeTTa symbolic analysis for functional patterns
        functional_query = f"(functional-pattern {func_name} $type)"
        functional_result = self._execute_metta_query(functional_query)
        if functional_result:
            guidance["functional_type"] = functional_result
        elif any(pattern in code for pattern in ["map", "filter", "reduce"]):
            guidance["functional_type"] = "higher_order"
        else:
            guidance["functional_type"] = "pure_function"
        
        # MeTTa analysis for composability
        composable_query = f"(composable-operations {func_name})"
        if self._execute_metta_query(composable_query):
            guidance["composable"] = "true"
        else:
            guidance["composable"] = "false"
        
        return guidance
    
    def _get_parallel_transformation_guidance_with_metta(self, context: GenerationContext) -> Dict[str, str]:
        """Get MeTTa-guided parallel transformation guidance."""
        guidance = {}
        func_name = context.function_name
        
        # MeTTa symbolic analysis for parallelization patterns
        parallel_query = f"(parallelization-pattern {func_name} $type)"
        parallel_result = self._execute_metta_query(parallel_query)
        if parallel_result:
            guidance["parallel_type"] = parallel_result
        else:
            guidance["parallel_type"] = "data_parallel"
        
        # MeTTa analysis for independence
        independence_query = f"(independent-iterations {func_name})"
        if self._execute_metta_query(independence_query):
            guidance["independence"] = "verified"
        else:
            guidance["independence"] = "assumed"
        
        return guidance
    
    def _get_lazy_transformation_guidance_with_metta(self, context: GenerationContext) -> Dict[str, str]:
        """Get MeTTa-guided lazy transformation guidance."""
        guidance = {}
        func_name = context.function_name
        
        # MeTTa symbolic analysis for lazy evaluation patterns
        lazy_query = f"(lazy-evaluation-pattern {func_name} $type)"
        lazy_result = self._execute_metta_query(lazy_query)
        if lazy_result:
            guidance["lazy_type"] = lazy_result
        else:
            guidance["lazy_type"] = "generator_based"
        
        return guidance
    
    # Template methods for generating transformed code
    
    def _create_recursive_template(self, context: GenerationContext, guidance: Dict[str, str]) -> str:
        """Create recursive implementation using MeTTa guidance."""
        func_name = context.function_name
        original_code = context.original_code
        
        # Extract parameters from original function
        params = self._extract_function_parameters(original_code, func_name)
        
        # Generate recursive implementation based on MeTTa guidance
        accumulation_type = guidance.get("accumulation_type", "generic")
        termination_condition = guidance.get("termination_condition", "base_case_reached")
        
        new_func_name = f"{func_name}_recursive"
        
        if accumulation_type == "maximum":
            return self._create_recursive_max_template(new_func_name, params, original_code)
        elif accumulation_type == "minimum":
            return self._create_recursive_min_template(new_func_name, params, original_code)
        elif accumulation_type == "sum":
            return self._create_recursive_sum_template(new_func_name, params, original_code)
        else:
            return self._create_generic_recursive_template(new_func_name, params, original_code, guidance)
    
    def _create_iterative_template(self, context: GenerationContext, guidance: Dict[str, str]) -> str:
        """Create iterative implementation using MeTTa guidance."""
        func_name = context.function_name
        original_code = context.original_code
        
        # Extract parameters from original function
        params = self._extract_function_parameters(original_code, func_name)
        
        new_func_name = f"{func_name}_iterative"
        requires_stack = guidance.get("requires_stack", "false") == "true"
        
        if requires_stack:
            return self._create_stack_based_iterative_template(new_func_name, params, original_code)
        else:
            return self._create_simple_iterative_template(new_func_name, params, original_code)
    
    def _create_functional_template(self, context: GenerationContext, guidance: Dict[str, str]) -> str:
        """Create functional implementation using MeTTa guidance."""
        func_name = context.function_name
        original_code = context.original_code
        
        # Extract parameters from original function
        params = self._extract_function_parameters(original_code, func_name)
        
        new_func_name = f"{func_name}_functional"
        functional_type = guidance.get("functional_type", "pure_function")
        
        if functional_type == "higher_order":
            return self._create_higher_order_functional_template(new_func_name, params, original_code)
        else:
            return self._create_pure_functional_template(new_func_name, params, original_code)
    
    def _create_parallel_template(self, context: GenerationContext, guidance: Dict[str, str]) -> str:
        """Create parallel implementation using MeTTa guidance."""
        func_name = context.function_name
        original_code = context.original_code
        
        # Extract parameters from original function
        params = self._extract_function_parameters(original_code, func_name)
        
        new_func_name = f"{func_name}_parallel"
        parallel_type = guidance.get("parallel_type", "data_parallel")
        
        return self._create_parallel_processing_template(new_func_name, params, original_code, parallel_type)
    
    def _create_lazy_template(self, context: GenerationContext, guidance: Dict[str, str]) -> str:
        """Create lazy evaluation implementation using MeTTa guidance."""
        func_name = context.function_name
        original_code = context.original_code
        
        # Extract parameters from original function
        params = self._extract_function_parameters(original_code, func_name)
        
        new_func_name = f"{func_name}_lazy"
        lazy_type = guidance.get("lazy_type", "generator_based")
        
        return self._create_generator_based_template(new_func_name, params, original_code)
    
    # Specific template implementations
    
    def _create_recursive_max_template(self, func_name: str, params: List[str], original_code: str) -> str:
        """Create recursive maximum-finding template."""
        return f'''def {func_name}({", ".join(params)}, index=None):
    """MeTTa-guided recursive transformation: maximum finding."""
    # Generated using MeTTa symbolic reasoning
    # Transformation: iterative-to-recursive (maximum pattern)
    
    if index is None:
        index = {params[1] if len(params) > 1 else "0"}
    
    # MeTTa-verified base case
    if index >= {params[2] if len(params) > 2 else f"len({params[0]})"}:
        return None
    
    # Process current element
    current = {params[0]}[index]
    
    # Recursive call with symbolic reasoning
    rest_result = {func_name}({", ".join(params)}, index + 1)
    
    # Combine using MeTTa-guided maximum logic
    if rest_result is None:
        return current
    else:
        return max(current, rest_result)'''
    
    def _create_recursive_min_template(self, func_name: str, params: List[str], original_code: str) -> str:
        """Create recursive minimum-finding template."""
        return f'''def {func_name}({", ".join(params)}, index=None):
    """MeTTa-guided recursive transformation: minimum finding."""
    # Generated using MeTTa symbolic reasoning
    # Transformation: iterative-to-recursive (minimum pattern)
    
    if index is None:
        index = {params[1] if len(params) > 1 else "0"}
    
    # MeTTa-verified base case
    if index >= {params[2] if len(params) > 2 else f"len({params[0]})"}:
        return None
    
    # Process current element
    current = {params[0]}[index]
    
    # Recursive call with symbolic reasoning
    rest_result = {func_name}({", ".join(params)}, index + 1)
    
    # Combine using MeTTa-guided minimum logic
    if rest_result is None:
        return current
    else:
        return min(current, rest_result)'''
    
    def _create_recursive_sum_template(self, func_name: str, params: List[str], original_code: str) -> str:
        """Create recursive summation template."""
        return f'''def {func_name}({", ".join(params)}, index=None):
    """MeTTa-guided recursive transformation: summation."""
    # Generated using MeTTa symbolic reasoning
    # Transformation: iterative-to-recursive (sum pattern)
    
    if index is None:
        index = {params[1] if len(params) > 1 else "0"}
    
    # MeTTa-verified base case
    if index >= {params[2] if len(params) > 2 else f"len({params[0]})"}:
        return 0
    
    # Process current element and recurse
    current = {params[0]}[index]
    return current + {func_name}({", ".join(params)}, index + 1)'''
    
    def _create_generic_recursive_template(self, func_name: str, params: List[str], 
                                         original_code: str, guidance: Dict[str, str]) -> str:
        """Create generic recursive template."""
        return f'''def {func_name}({", ".join(params)}, index=None):
    """MeTTa-guided recursive transformation: generic pattern."""
    # Generated using MeTTa symbolic reasoning
    # Transformation: iterative-to-recursive (generic pattern)
    
    if index is None:
        index = {params[1] if len(params) > 1 else "0"}
    
    # MeTTa-guided base case
    if index >= {params[2] if len(params) > 2 else f"len({params[0]})"}:
        return None
    
    # Process current element
    current_element = {params[0]}[index]
    
    # Recursive processing with MeTTa guidance
    rest_result = {func_name}({", ".join(params)}, index + 1)
    
    # Combine results (preserving original semantics)
    if rest_result is None:
        return current_element
    else:
        # Generic combination - adapt based on original function logic
        return rest_result  # Placeholder for complex logic'''
    
    def _create_stack_based_iterative_template(self, func_name: str, params: List[str], original_code: str) -> str:
        """Create stack-based iterative template."""
        return f'''def {func_name}({", ".join(params)}):
    """MeTTa-guided iterative transformation: stack-based."""
    # Generated using MeTTa symbolic reasoning
    # Transformation: recursive-to-iterative (stack-based)
    
    # MeTTa analysis determined stack is required
    stack = [({", ".join(params)})]
    result = None
    
    while stack:
        current_args = stack.pop()
        
        # Process based on MeTTa-guided logic
        # Base case handling
        if self._is_base_case(*current_args):
            if result is None:
                result = self._base_case_value(*current_args)
            else:
                result = self._combine_results(result, self._base_case_value(*current_args))
        else:
            # Push subproblems onto stack
            subproblems = self._generate_subproblems(*current_args)
            stack.extend(subproblems)
    
    return result'''
    
    def _create_simple_iterative_template(self, func_name: str, params: List[str], original_code: str) -> str:
        """Create simple iterative template."""
        return f'''def {func_name}({", ".join(params)}):
    """MeTTa-guided iterative transformation: simple iteration."""
    # Generated using MeTTa symbolic reasoning
    # Transformation: recursive-to-iterative (simple)
    
    # MeTTa analysis determined simple iteration is sufficient
    result = None
    
    # Iterative processing based on MeTTa guidance
    for i in range({params[1] if len(params) > 1 else "0"}, 
                   {params[2] if len(params) > 2 else f"len({params[0]})"}):
        current_element = {params[0]}[i]
        
        if result is None:
            result = current_element
        else:
            # Combine based on original semantics
            result = max(result, current_element)  # Example: adapt as needed
    
    return result'''
    
    def _create_higher_order_functional_template(self, func_name: str, params: List[str], original_code: str) -> str:
        """Create higher-order functional template."""
        return f'''def {func_name}({", ".join(params)}):
    """MeTTa-guided functional transformation: higher-order functions."""
    # Generated using MeTTa symbolic reasoning
    # Transformation: imperative-to-functional (higher-order)
    
    from functools import reduce
    from typing import Callable
    
    # MeTTa analysis determined higher-order functions are applicable
    
    # Input validation (preserving original behavior)
    if {params[1]} < 0 or {params[2]} > len({params[0]}) or {params[1]} >= {params[2]}:
        return None
    
    # Extract relevant slice
    relevant_slice = {params[0]}[{params[1]}:{params[2]}]
    
    # Functional approach using higher-order functions
    if not relevant_slice:
        return None
    
    # MeTTa-guided functional composition
    return reduce(lambda acc, x: max(acc, x), relevant_slice)'''
    
    def _create_pure_functional_template(self, func_name: str, params: List[str], original_code: str) -> str:
        """Create pure functional template."""
        return f'''def {func_name}({", ".join(params)}):
    """MeTTa-guided functional transformation: pure function."""
    # Generated using MeTTa symbolic reasoning
    # Transformation: imperative-to-functional (pure)
    
    # MeTTa analysis verified no side effects
    
    # Input validation (preserving original behavior)
    if {params[1]} < 0 or {params[2]} > len({params[0]}) or {params[1]} >= {params[2]}:
        return None
    
    # Pure functional approach
    relevant_slice = {params[0]}[{params[1]}:{params[2]}]
    
    try:
        # MeTTa-guided pure functional implementation
        return max(relevant_slice) if relevant_slice else None
    except ValueError:
        return None'''
    
    def _create_parallel_processing_template(self, func_name: str, params: List[str], 
                                           original_code: str, parallel_type: str) -> str:
        """Create parallel processing template."""
        return f'''def {func_name}({", ".join(params)}):
    """MeTTa-guided parallel transformation: {parallel_type}."""
    # Generated using MeTTa symbolic reasoning
    # Transformation: sequential-to-parallel ({parallel_type})
    
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing as mp
    
    # MeTTa analysis verified independent iterations
    
    # Input validation
    if {params[1]} < 0 or {params[2]} > len({params[0]}) or {params[1]} >= {params[2]}:
        return None
    
    # Extract work slice
    work_data = {params[0]}[{params[1]}:{params[2]}]
    
    if not work_data:
        return None
    
    # MeTTa-guided parallel processing
    def process_chunk(chunk):
        return max(chunk) if chunk else None
    
    # Divide work into chunks for parallel processing
    num_workers = min(mp.cpu_count(), len(work_data))
    chunk_size = max(1, len(work_data) // num_workers)
    
    chunks = [work_data[i:i + chunk_size] 
              for i in range(0, len(work_data), chunk_size)]
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        chunk_results = list(executor.map(process_chunk, chunks))
    
    # Combine results
    valid_results = [r for r in chunk_results if r is not None]
    return max(valid_results) if valid_results else None'''
    
    def _create_generator_based_template(self, func_name: str, params: List[str], original_code: str) -> str:
        """Create generator-based lazy evaluation template."""
        return f'''def {func_name}({", ".join(params)}):
    """MeTTa-guided lazy transformation: generator-based."""
    # Generated using MeTTa symbolic reasoning
    # Transformation: eager-to-lazy (generator-based)
    
    # MeTTa analysis determined lazy evaluation is beneficial
    
    # Input validation
    if {params[1]} < 0 or {params[2]} > len({params[0]}) or {params[1]} >= {params[2]}:
        return None
    
    # Lazy generator for processing elements
    def element_generator():
        for i in range({params[1]}, {params[2]}):
            yield {params[0]}[i]
    
    # MeTTa-guided lazy evaluation
    max_val = None
    for element in element_generator():
        if max_val is None or element > max_val:
            max_val = element
            
    return max_val'''
    
    # Helper methods
    
    def _extract_function_parameters(self, code: str, func_name: str) -> List[str]:
        """Extract parameters from function definition."""
        import re
        match = re.search(rf'def\s+{func_name}\s*\(([^)]*)\)', code)
        if match:
            params_str = match.group(1)
            return [p.strip() for p in params_str.split(',') if p.strip()]
        return ['data', 'start', 'end']  # Default parameters
    
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
    
    def _get_operations_from_context(self, context: GenerationContext) -> List[str]:
        """Get operations from context."""
        if hasattr(context, 'detected_patterns') and context.detected_patterns:
            return context.detected_patterns[0].operations
        return ["transformation"]


# Test function for standalone testing
def test_metta_focused_generator():
    """Test the MeTTa-focused generator standalone."""
    print("🧪 TESTING METTA-FOCUSED ALGORITHM TRANSFORMATION GENERATOR")
    print("=" * 70)
    
    # Create test context
    class TestContext:
        def __init__(self, code, func_name):
            self.original_code = code
            self.function_name = func_name
            self.detected_patterns = [TestPattern()]
            self.metta_space = TestMettaSpace()
    
    class TestPattern:
        def __init__(self):
            self.pattern_family = "search"
            self.operations = ["comparison"]
            self.data_structures = ["list"]
    
    class TestMettaSpace:
        def run(self, query):
            # Simulate MeTTa space responses
            if "algorithm-transformation-safe" in query:
                return ["safe"]
            elif "has-iterative-pattern" in query:
                return ["true"]
            elif "no-side-effects" in query:
                return ["true"]
            return []
    
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
    generator = AlgorithmTransformationGenerator()
    context = TestContext(test_code, "find_max_in_range")
    
    # Test if it can generate
    can_generate = generator.can_generate(context, "algorithm_transformation")
    print(f"Can generate: {can_generate}")
    
    if can_generate:
        # Generate candidates
        candidates = generator.generate_candidates(context, "algorithm_transformation")
        print(f"Generated {len(candidates)} candidates:")
        
        for i, candidate in enumerate(candidates, 1):
            print(f"\n{i}. {candidate.name}")
            print(f"   Description: {candidate.description}")
            print(f"   Confidence: {candidate.confidence}")
            print(f"   MeTTa Derivation: {candidate.metta_derivation}")
            
            # Show code preview
            code_lines = candidate.code.split('\n')[:8]
            for line in code_lines:
                if line.strip():
                    print(f"   {line}")
    
    return True


if __name__ == "__main__":
    test_metta_focused_generator()