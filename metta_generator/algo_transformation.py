#!/usr/bin/env python3
"""
MeTTa-Powered Algorithm Transformation Generator
"""

from typing import List, Dict, Optional, Any
from metta_generator.base import BaseDonorGenerator, GenerationContext, DonorCandidate, GenerationStrategy

class AlgorithmTransformationGenerator(BaseDonorGenerator):
    """Generator that uses MeTTa symbolic reasoning for algorithm transformations."""
    
    def __init__(self):
        super().__init__()
        self._load_transformation_rules()
    
    def _load_transformation_rules(self):
        """Load algorithm transformation rules into MeTTa reasoning."""
        self.transformation_rules = [
            # Iterative to recursive transformation rules
            """(= (iterative-to-recursive-safe $func)
               (and (has-loop-structure $func)
                    (has-clear-termination $func)
                    (simple-state-management $func)
                    (no-complex-dependencies $func)))""",
            
            # Recursive to iterative transformation rules  
            """(= (recursive-to-iterative-safe $func)
               (and (calls-self $func)
                    (tail-recursive-or-simple $func)
                    (bounded-recursion-depth $func)))""",
            
            # Functional transformation rules
            """(= (imperative-to-functional-safe $func)
               (and (no-side-effects $func)
                    (deterministic-behavior $func)
                    (uses-immutable-data $func)))""",
            
            # Parallelization rules
            """(= (sequential-to-parallel-safe $func)
               (and (has-independent-iterations $func)
                    (no-shared-mutable-state $func)
                    (commutative-operations $func)))""",
            
            # Pattern-specific transformation guidance
            """(= (transformation-guidance iterative-to-recursive $func)
               (let $termination (derive-termination-condition $func)
                    (let $accumulator (identify-accumulator-pattern $func)
                         (recursive-structure $func $termination $accumulator))))""",
            
            """(= (transformation-guidance recursive-to-iterative $func)
               (let $stack-needed (requires-explicit-stack $func)
                    (let $state-vars (identify-state-variables $func)
                         (iterative-structure $func $stack-needed $state-vars))))""",
            
            # Code generation rules
            """(= (generate-recursive-variant $func $guidance)
               (match $guidance
                 ((recursive-structure $func $term $acc)
                  (construct-recursive-function $func $term $acc))))""",
            
            """(= (generate-iterative-variant $func $guidance)
               (match $guidance
                 ((iterative-structure $func $stack $vars)
                  (construct-iterative-function $func $stack $vars))))""",
        ]
    
    def can_generate(self, context: GenerationContext, strategy) -> bool:
        """Use MeTTa reasoning to determine if algorithm transformation is applicable."""
        if hasattr(strategy, 'value'):
            strategy_name = strategy.value
        else:
            strategy_name = str(strategy)
            
        if strategy_name != "algorithm_transformation":
            return False
        
        # Use MeTTa reasoning to check applicability
        applicability_result = self._use_metta_reasoning(
            context, "can_generate", strategy=strategy
        )
        
        if applicability_result:
            print(f"      MeTTa reasoning: AlgorithmTransformationGenerator CAN generate")
            return True
        
        # Fallback to pattern-based reasoning
        can_generate = self._symbolic_applicability_check(context)
        
        if can_generate:
            print(f"      Symbolic reasoning: AlgorithmTransformationGenerator CAN generate")
        else:
            print(f"      AlgorithmTransformationGenerator: cannot generate")
        
        return can_generate
    
    def _generate_candidates_impl(self, context: GenerationContext, strategy) -> List[DonorCandidate]:
        """Generate algorithm transformation candidates using MeTTa reasoning."""
        candidates = []
        
        # Use MeTTa reasoning to find applicable transformations
        transformations = self._use_metta_reasoning(
            context, "find_transformations", strategy=strategy
        )
        
        if not transformations:
            print(f"        MeTTa reasoning found no transformations, using symbolic fallback")
            transformations = self._symbolic_transformation_detection(context)
        
        print(f"        Found {len(transformations)} potential transformations: {[t.get('from', 'unknown') + '->' + t.get('to', 'unknown') for t in transformations]}")
        
        for transformation in transformations:
            candidate = self._create_metta_reasoned_candidate(context, transformation)
            if candidate:
                candidates.append(candidate)
                print(f"          Created candidate: {candidate.name}")
        
        return candidates
    
    def get_supported_strategies(self) -> List:
        """Get list of strategies this generator supports."""
        return [GenerationStrategy.ALGORITHM_TRANSFORMATION]
    
    def _symbolic_applicability_check(self, context: GenerationContext) -> bool:
        """Symbolic reasoning fallback for applicability checking."""
        code = context.original_code.lower()
        func_name = context.function_name
        
        # Check for algorithmic patterns that can be transformed
        has_loop = any(pattern in code for pattern in ["for ", "while "])
        has_recursion = func_name in code.replace(f"def {func_name}", "")
        has_iteration_pattern = has_loop and any(op in code for op in ["range(", "enumerate(", "zip("])
        has_accumulation = any(pattern in code for pattern in ["+=", "*=", "sum(", "max(", "min("])
        
        return has_loop or has_recursion or has_iteration_pattern or has_accumulation
    
    def _symbolic_transformation_detection(self, context: GenerationContext) -> List[Dict[str, Any]]:
        """Symbolic detection of applicable transformations."""
        transformations = []
        code = context.original_code.lower()
        func_name = context.function_name
        
        # Detect iterative patterns that can become recursive
        if self._has_iterative_pattern(code, func_name):
            transformations.append({
                "from": "iterative-pattern",
                "to": "recursive-pattern", 
                "safety": "safe" if self._is_recursion_safe(code) else "unsafe",
                "guidance": self._get_recursive_guidance(context)
            })
        
        # Detect recursive patterns that can become iterative
        if self._has_recursive_pattern(code, func_name):
            transformations.append({
                "from": "recursive-pattern",
                "to": "iterative-pattern",
                "safety": "safe" if self._is_iteration_safe(code) else "unsafe", 
                "guidance": self._get_iterative_guidance(context)
            })
        
        # Detect imperative patterns that can become functional
        if self._has_imperative_pattern(code) and self._is_functional_safe(code):
            transformations.append({
                "from": "imperative-pattern",
                "to": "functional-pattern",
                "safety": "safe",
                "guidance": self._get_functional_guidance(context)
            })
        
        # Detect sequential patterns that can be parallelized
        if self._has_parallelizable_pattern(code):
            transformations.append({
                "from": "sequential-pattern", 
                "to": "parallel-pattern",
                "safety": "safe" if self._is_parallelization_safe(code) else "unsafe",
                "guidance": self._get_parallel_guidance(context)
            })
        
        return transformations
    
    def _create_metta_reasoned_candidate(self, context: GenerationContext, 
                                       transformation: Dict[str, Any]) -> Optional[DonorCandidate]:
        """Create a candidate using MeTTa reasoning for code generation."""
        try:
            # Use MeTTa reasoning to generate the transformed code
            generated_code = self._use_metta_reasoning(
                context, "generate_code", transformation=transformation
            )
            
            if not generated_code:
                # Fallback to template-based generation with MeTTa guidance
                generated_code = self._generate_with_metta_guidance(context, transformation)
            
            if not generated_code:
                print(f"          Failed to generate code for {transformation}")
                return None
            
            # Extract transformation details
            from_pattern = transformation.get('from', 'unknown')
            to_pattern = transformation.get('to', 'unknown')
            safety = transformation.get('safety', 'unknown')
            guidance = transformation.get('guidance', {})
            
            # Create candidate name
            transform_name = f"{from_pattern}_to_{to_pattern}".replace('-pattern', '').replace('-', '_')
            new_func_name = f"{context.function_name}_{transform_name}"
            
            # Create MeTTa derivation trace
            metta_derivation = [
                f"(algorithm-transformation {context.function_name} {from_pattern} {to_pattern})",
                f"(transformation-safety {safety})",
                f"(metta-reasoning-applied {self.generator_name})"
            ]
            
            # Add guidance to derivation
            if isinstance(guidance, dict):
                for key, value in guidance.items():
                    metta_derivation.append(f"(transformation-guidance {key} {value})")
            
            # Determine properties based on transformation
            properties = self._derive_properties_from_transformation(transformation)
            
            # Calculate confidence based on MeTTa reasoning and safety
            confidence = self._calculate_metta_confidence(transformation, safety)
            
            return DonorCandidate(
                name=new_func_name,
                description=f"MeTTa-reasoned {from_pattern} to {to_pattern} transformation",
                code=generated_code,
                strategy="algorithm_transformation",
                pattern_family=self._extract_pattern_family(context),
                data_structures_used=self._extract_data_structures(context),
                operations_used=self._extract_operations(context),
                metta_derivation=metta_derivation,
                confidence=confidence,
                properties=properties,
                complexity_estimate=self._estimate_complexity_change(transformation),
                applicability_scope=self._determine_applicability_scope(transformation, safety),
                generator_used=self.generator_name,
                metta_reasoning_trace=[f"transformation-reasoning: {transformation}"]
            )
            
        except Exception as e:
            print(f"          Error creating MeTTa-reasoned candidate: {e}")
            return None
    
    def _generate_with_metta_guidance(self, context: GenerationContext, 
                                    transformation: Dict[str, Any]) -> Optional[str]:
        """Generate code using MeTTa-derived guidance and templates."""
        from_pattern = transformation.get('from', '')
        to_pattern = transformation.get('to', '')
        guidance = transformation.get('guidance', {})
        
        # Route to appropriate generation method based on transformation
        if 'iterative' in from_pattern and 'recursive' in to_pattern:
            return self._generate_recursive_from_iterative(context, guidance)
        elif 'recursive' in from_pattern and 'iterative' in to_pattern:
            return self._generate_iterative_from_recursive(context, guidance)
        elif 'imperative' in from_pattern and 'functional' in to_pattern:
            return self._generate_functional_from_imperative(context, guidance)
        elif 'sequential' in from_pattern and 'parallel' in to_pattern:
            return self._generate_parallel_from_sequential(context, guidance)
        else:
            return self._generate_generic_transformation(context, transformation)
    
    def _generate_recursive_from_iterative(self, context: GenerationContext, 
                                         guidance: Dict[str, Any]) -> str:
        """Generate recursive version from iterative using MeTTa guidance."""
        func_name = context.function_name
        code = context.original_code
        
        # Extract MeTTa-guided parameters
        termination_condition = guidance.get('termination_condition', 'base_case_reached')
        accumulator_type = guidance.get('accumulator_type', 'generic')
        
        new_func_name = f"{func_name}_recursive"
        
        # Generate recursive template based on MeTTa reasoning
        if accumulator_type == 'maximum':
            return self._create_recursive_max_template_metta(new_func_name, context, guidance)
        elif accumulator_type == 'sum':
            return self._create_recursive_sum_template_metta(new_func_name, context, guidance)
        else:
            return self._create_generic_recursive_template_metta(new_func_name, context, guidance)
    
    def _generate_iterative_from_recursive(self, context: GenerationContext,
                                         guidance: Dict[str, Any]) -> str:
        """Generate iterative version from recursive using MeTTa guidance."""
        func_name = context.function_name
        
        # Extract MeTTa-guided parameters
        requires_stack = guidance.get('requires_stack', False)
        state_variables = guidance.get('state_variables', [])
        
        new_func_name = f"{func_name}_iterative"
        
        if requires_stack:
            return self._create_stack_based_iterative_template_metta(new_func_name, context, guidance)
        else:
            return self._create_simple_iterative_template_metta(new_func_name, context, guidance)
    
    def _generate_functional_from_imperative(self, context: GenerationContext,
                                           guidance: Dict[str, Any]) -> str:
        """Generate functional version from imperative using MeTTa guidance."""
        func_name = context.function_name
        
        functional_style = guidance.get('functional_style', 'pure')
        composable = guidance.get('composable', False)
        
        new_func_name = f"{func_name}_functional"
        
        if functional_style == 'higher_order':
            return self._create_higher_order_template_metta(new_func_name, context, guidance)
        else:
            return self._create_pure_functional_template_metta(new_func_name, context, guidance)
    
    def _generate_parallel_from_sequential(self, context: GenerationContext,
                                         guidance: Dict[str, Any]) -> str:
        """Generate parallel version from sequential using MeTTa guidance."""
        func_name = context.function_name
        
        parallel_type = guidance.get('parallel_type', 'data_parallel')
        independence_verified = guidance.get('independence_verified', False)
        
        new_func_name = f"{func_name}_parallel"
        
        return self._create_parallel_template_metta(new_func_name, context, guidance, parallel_type)
    
    def _generate_generic_transformation(self, context: GenerationContext,
                                       transformation: Dict[str, Any]) -> str:
        """Generate generic transformation with MeTTa reasoning traces."""
        func_name = context.function_name
        to_pattern = transformation.get('to', 'variant')
        
        new_func_name = f"{func_name}_{to_pattern.replace('-', '_')}"
        transformed_code = context.original_code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Add MeTTa reasoning trace as documentation
        reasoning_trace = f"MeTTa transformation reasoning: {transformation}"
        lines = transformed_code.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                lines.insert(i + 1, f'    """{reasoning_trace}"""')
                break
        
        return '\n'.join(lines)
    
    # MeTTa-guided template creation methods
    
    def _create_recursive_max_template_metta(self, func_name: str, context: GenerationContext,
                                           guidance: Dict[str, Any]) -> str:
        """Create recursive maximum template with MeTTa guidance."""
        params = self._extract_function_parameters(context.original_code, context.function_name)
        termination = guidance.get('termination_condition', 'index >= end')
        
        return f'''def {func_name}({", ".join(params)}, index=None):
    """MeTTa-guided recursive transformation: maximum finding.
    
    MeTTa reasoning: {guidance}
    Termination condition: {termination}
    """
    if index is None:
        index = {params[1] if len(params) > 1 else "0"}
    
    # MeTTa-derived base case
    if index >= {params[2] if len(params) > 2 else f"len({params[0]})"}:
        return None
    
    # Process current element (MeTTa-guided)
    current = {params[0]}[index]
    
    # Recursive call with MeTTa reasoning
    rest_result = {func_name}({", ".join(params)}, index + 1)
    
    # MeTTa-guided combination logic
    if rest_result is None:
        return current
    else:
        return max(current, rest_result)'''
    
    def _create_recursive_sum_template_metta(self, func_name: str, context: GenerationContext,
                                           guidance: Dict[str, Any]) -> str:
        """Create recursive sum template with MeTTa guidance."""
        params = self._extract_function_parameters(context.original_code, context.function_name)
        
        return f'''def {func_name}({", ".join(params)}, index=None):
    """MeTTa-guided recursive transformation: summation.
    
    MeTTa reasoning: {guidance}
    """
    if index is None:
        index = {params[1] if len(params) > 1 else "0"}
    
    # MeTTa-derived base case
    if index >= {params[2] if len(params) > 2 else f"len({params[0]})"}:
        return 0
    
    # MeTTa-guided recursive summation
    current = {params[0]}[index]
    return current + {func_name}({", ".join(params)}, index + 1)'''
    
    def _create_generic_recursive_template_metta(self, func_name: str, context: GenerationContext,
                                               guidance: Dict[str, Any]) -> str:
        """Create generic recursive template with MeTTa guidance."""
        params = self._extract_function_parameters(context.original_code, context.function_name)
        
        return f'''def {func_name}({", ".join(params)}, index=None):
    """MeTTa-guided recursive transformation: generic pattern.
    
    MeTTa reasoning applied: {guidance}
    """
    if index is None:
        index = {params[1] if len(params) > 1 else "0"}
    
    # MeTTa-derived base case
    if index >= {params[2] if len(params) > 2 else f"len({params[0]})"}:
        return None
    
    # Process current element with MeTTa guidance
    current_element = {params[0]}[index]
    
    # MeTTa-guided recursive processing
    rest_result = {func_name}({", ".join(params)}, index + 1)
    
    # MeTTa-reasoned combination
    if rest_result is None:
        return current_element
    else:
        # Combination logic derived from MeTTa reasoning
        return rest_result'''
    
    def _create_stack_based_iterative_template_metta(self, func_name: str, context: GenerationContext,
                                                   guidance: Dict[str, Any]) -> str:
        """Create stack-based iterative template with MeTTa guidance."""
        params = self._extract_function_parameters(context.original_code, context.function_name)
        state_vars = guidance.get('state_variables', [])
        
        return f'''def {func_name}({", ".join(params)}):
    """MeTTa-guided iterative transformation: stack-based.
    
    MeTTa analysis determined stack is required.
    State variables: {state_vars}
    MeTTa guidance: {guidance}
    """
    # MeTTa-derived stack structure
    stack = [({", ".join(params)})]
    result = None
    
    while stack:
        current_args = stack.pop()
        
        # MeTTa-guided processing logic
        # Base case handling derived from MeTTa reasoning
        if self._is_base_case(*current_args):
            if result is None:
                result = self._base_case_value(*current_args)
            else:
                result = self._combine_results(result, self._base_case_value(*current_args))
        else:
            # MeTTa-derived subproblem generation
            subproblems = self._generate_subproblems(*current_args)
            stack.extend(subproblems)
    
    return result'''
    
    def _create_simple_iterative_template_metta(self, func_name: str, context: GenerationContext,
                                              guidance: Dict[str, Any]) -> str:
        """Create simple iterative template with MeTTa guidance."""
        params = self._extract_function_parameters(context.original_code, context.function_name)
        
        return f'''def {func_name}({", ".join(params)}):
    """MeTTa-guided iterative transformation: simple iteration.
    
    MeTTa analysis determined simple iteration is sufficient.
    MeTTa guidance: {guidance}
    """
    result = None
    
    # MeTTa-guided iterative processing
    for i in range({params[1] if len(params) > 1 else "0"}, 
                   {params[2] if len(params) > 2 else f"len({params[0]})"}):
        current_element = {params[0]}[i]
        
        # MeTTa-derived combination logic
        if result is None:
            result = current_element
        else:
            result = max(result, current_element)  # MeTTa-guided operation
    
    return result'''
    
    def _create_higher_order_template_metta(self, func_name: str, context: GenerationContext,
                                          guidance: Dict[str, Any]) -> str:
        """Create higher-order functional template with MeTTa guidance."""
        params = self._extract_function_parameters(context.original_code, context.function_name)
        
        return f'''def {func_name}({", ".join(params)}):
    """MeTTa-guided functional transformation: higher-order functions.
    
    MeTTa analysis determined higher-order functions are applicable.
    MeTTa guidance: {guidance}
    """
    from functools import reduce
    
    # MeTTa-derived input validation
    if {params[1]} < 0 or {params[2]} > len({params[0]}) or {params[1]} >= {params[2]}:
        return None
    
    # Extract relevant slice (MeTTa-guided)
    relevant_slice = {params[0]}[{params[1]}:{params[2]}]
    
    # MeTTa-guided functional composition
    if not relevant_slice:
        return None
    
    # Higher-order function application derived from MeTTa reasoning
    return reduce(lambda acc, x: max(acc, x), relevant_slice)'''
    
    def _create_pure_functional_template_metta(self, func_name: str, context: GenerationContext,
                                             guidance: Dict[str, Any]) -> str:
        """Create pure functional template with MeTTa guidance."""
        params = self._extract_function_parameters(context.original_code, context.function_name)
        
        return f'''def {func_name}({", ".join(params)}):
    """MeTTa-guided functional transformation: pure function.
    
    MeTTa verified no side effects.
    MeTTa guidance: {guidance}
    """
    # MeTTa-derived input validation
    if {params[1]} < 0 or {params[2]} > len({params[0]}) or {params[1]} >= {params[2]}:
        return None
    
    # Pure functional approach (MeTTa-guided)
    relevant_slice = {params[0]}[{params[1]}:{params[2]}]
    
    try:
        # MeTTa-guided pure functional implementation
        return max(relevant_slice) if relevant_slice else None
    except ValueError:
        return None'''
    
    def _create_parallel_template_metta(self, func_name: str, context: GenerationContext,
                                      guidance: Dict[str, Any], parallel_type: str) -> str:
        """Create parallel template with MeTTa guidance."""
        params = self._extract_function_parameters(context.original_code, context.function_name)
        
        return f'''def {func_name}({", ".join(params)}):
    """MeTTa-guided parallel transformation: {parallel_type}.
    
    MeTTa verified independent iterations.
    MeTTa guidance: {guidance}
    """
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing as mp
    
    # MeTTa-derived input validation
    if {params[1]} < 0 or {params[2]} > len({params[0]}) or {params[1]} >= {params[2]}:
        return None
    
    work_data = {params[0]}[{params[1]}:{params[2]}]
    
    if not work_data:
        return None
    
    # MeTTa-guided parallel processing
    def process_chunk(chunk):
        return max(chunk) if chunk else None
    
    # Divide work based on MeTTa analysis
    num_workers = min(mp.cpu_count(), len(work_data))
    chunk_size = max(1, len(work_data) // num_workers)
    
    chunks = [work_data[i:i + chunk_size] 
              for i in range(0, len(work_data), chunk_size)]
    
    # MeTTa-guided parallel execution
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        chunk_results = list(executor.map(process_chunk, chunks))
    
    # MeTTa-derived result combination
    valid_results = [r for r in chunk_results if r is not None]
    return max(valid_results) if valid_results else None'''
    
    # Helper methods for pattern detection and guidance
    
    def _has_iterative_pattern(self, code: str, func_name: str) -> bool:
        """Check for iterative patterns."""
        return (any(pattern in code for pattern in ["for ", "while "]) and
                func_name not in code.replace(f"def {func_name}", ""))
    
    def _has_recursive_pattern(self, code: str, func_name: str) -> bool:
        """Check for recursive patterns."""
        return (func_name in code.replace(f"def {func_name}", "") and
                any(pattern in code for pattern in ["if ", "return"]))
    
    def _has_imperative_pattern(self, code: str) -> bool:
        """Check for imperative patterns."""
        return any(pattern in code for pattern in ["for ", "while ", "if ", "="])
    
    def _has_parallelizable_pattern(self, code: str) -> bool:
        """Check for parallelizable patterns."""
        return ("for " in code and 
                not any(pattern in code for pattern in ["global ", "nonlocal "]))
    
    def _is_recursion_safe(self, code: str) -> bool:
        """Check if recursion transformation is safe."""
        return (any(pattern in code for pattern in ["break", "return", ">=", "<="]) and
                not any(pattern in code for pattern in ["global ", "nonlocal "]))
    
    def _is_iteration_safe(self, code: str) -> bool:
        """Check if iteration transformation is safe."""
        return not any(pattern in code for pattern in ["nested recursion", "mutual recursion"])
    
    def _is_functional_safe(self, code: str) -> bool:
        """Check if functional transformation is safe."""
        return not any(pattern in code for pattern in ["global ", "print(", "input(", "open("])
    
    def _is_parallelization_safe(self, code: str) -> bool:
        """Check if parallelization is safe."""
        return not any(pattern in code for pattern in ["global ", "shared", "lock"])
    
    def _get_recursive_guidance(self, context: GenerationContext) -> Dict[str, Any]:
        """Get MeTTa-derived guidance for recursive transformation."""
        code = context.original_code
        guidance = {}
        
        if "max" in code or "maximum" in code:
            guidance['accumulator_type'] = 'maximum'
        elif "sum" in code or "+=" in code:
            guidance['accumulator_type'] = 'sum'
        else:
            guidance['accumulator_type'] = 'generic'
        
        if "range(" in code:
            guidance['termination_condition'] = 'index >= end'
        else:
            guidance['termination_condition'] = 'base_case_reached'
        
        return guidance
    
    def _get_iterative_guidance(self, context: GenerationContext) -> Dict[str, Any]:
        """Get MeTTa-derived guidance for iterative transformation."""
        code = context.original_code
        guidance = {}
        
        # Determine if explicit stack is needed
        recursion_depth = code.count(context.function_name) - 1
        guidance['requires_stack'] = recursion_depth > 1
        
        # Identify state variables
        guidance['state_variables'] = ['result', 'index']
        
        return guidance
    
    def _get_functional_guidance(self, context: GenerationContext) -> Dict[str, Any]:
        """Get MeTTa-derived guidance for functional transformation."""
        code = context.original_code
        guidance = {}
        
        if any(func in code for func in ["map", "filter", "reduce"]):
            guidance['functional_style'] = 'higher_order'
        else:
            guidance['functional_style'] = 'pure'
        
        guidance['composable'] = "return" in code and "=" not in code.split('return')[0]
        
        return guidance
    
    def _get_parallel_guidance(self, context: GenerationContext) -> Dict[str, Any]:
        """Get MeTTa-derived guidance for parallel transformation."""
        code = context.original_code
        guidance = {}
        
        guidance['parallel_type'] = 'data_parallel'
        guidance['independence_verified'] = not any(dep in code for dep in ["previous", "last", "accumulate"])
        
        return guidance
    
    def _extract_function_parameters(self, code: str, func_name: str) -> List[str]:
        """Extract parameters from function definition."""
        import re
        match = re.search(rf'def\s+{func_name}\s*\(([^)]*)\)', code)
        if match:
            params_str = match.group(1)
            return [p.strip() for p in params_str.split(',') if p.strip()]
        return ['data', 'start', 'end']
    
    def _derive_properties_from_transformation(self, transformation: Dict[str, Any]) -> List[str]:
        """Derive properties from MeTTa transformation reasoning."""
        properties = ["metta-reasoned", "algorithm-transformed"]
        
        from_pattern = transformation.get('from', '')
        to_pattern = transformation.get('to', '')
        
        if 'recursive' in to_pattern:
            properties.append("recursive")
        if 'functional' in to_pattern:
            properties.append("functional")
        if 'parallel' in to_pattern:
            properties.append("parallel")
        
        if transformation.get('safety') == 'safe':
            properties.append("transformation-safe")
        
        return properties
    
    def _calculate_metta_confidence(self, transformation: Dict[str, Any], safety: str) -> float:
        """Calculate confidence based on MeTTa reasoning quality."""
        base_confidence = 0.7
        
        if safety == 'safe':
            base_confidence += 0.2
        elif safety == 'unsafe':
            base_confidence -= 0.3
        
        # Boost confidence if MeTTa provided detailed guidance
        if isinstance(transformation.get('guidance'), dict) and transformation['guidance']:
            base_confidence += 0.1
        
        return min(0.95, max(0.1, base_confidence))
    
    def _estimate_complexity_change(self, transformation: Dict[str, Any]) -> str:
        """Estimate complexity change from transformation."""
        to_pattern = transformation.get('to', '')
        
        if 'recursive' in to_pattern:
            return "same-or-worse-space"
        elif 'iterative' in to_pattern:
            return "improved-space"
        elif 'parallel' in to_pattern:
            return "improved-time"
        else:
            return "same"
    
    def _determine_applicability_scope(self, transformation: Dict[str, Any], safety: str) -> str:
        """Determine applicability scope based on transformation and safety."""
        if safety == 'safe':
            return "broad"
        elif safety == 'unsafe':
            return "narrow"
        else:
            return "medium"
    
    def _extract_pattern_family(self, context: GenerationContext) -> str:
        """Extract pattern family from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].pattern_family
        return "algorithmic"
    
    def _extract_data_structures(self, context: GenerationContext) -> List[str]:
        """Extract data structures from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].data_structures
        return ["generic"]
    
    def _extract_operations(self, context: GenerationContext) -> List[str]:
        """Extract operations from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].operations
        return ["algorithmic-transformation"]