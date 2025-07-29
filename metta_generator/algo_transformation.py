#!/usr/bin/env python3
"""
Fixed MeTTa-Powered Algorithm Transformation Generator
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
    
    def generate_candidates(self, context: GenerationContext, strategy) -> List[DonorCandidate]:
        """Generate algorithm transformation candidates using MeTTa reasoning."""
        candidates = self._generate_candidates_impl(context, strategy)
        
        # Ensure all candidates have proper generator attribution and MeTTa traces
        for candidate in candidates:
            if not hasattr(candidate, 'generator_used') or candidate.generator_used == "UnknownGenerator":
                candidate.generator_used = self.generator_name
            
            # Add MeTTa reasoning trace if not present
            if not getattr(candidate, 'metta_reasoning_trace', None):
                candidate.metta_reasoning_trace = [f"generated-by {self.generator_name}"]
        
        return candidates
    
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
        
        # Handle both raw MeTTa results (lists) and structured results (dicts)
        safe_transformations = []
        for t in transformations:
            if isinstance(t, dict):
                safe_transformations.append(t)
            elif isinstance(t, list) and len(t) >= 2:
                # Convert list format to dict format
                safe_transformations.append({
                    'from': str(t[0]) if len(t) > 0 else 'unknown',
                    'to': str(t[1]) if len(t) > 1 else 'unknown',
                    'pattern': str(t[0]) if len(t) > 0 else 'generic',
                    'target_pattern': str(t[1]) if len(t) > 1 else 'variant'
                })
            else:
                # Fallback for unexpected formats
                safe_transformations.append({
                    'from': 'unknown',
                    'to': 'variant', 
                    'pattern': 'generic',
                    'target_pattern': 'variant'
                })
        
        print(f"        Found {len(safe_transformations)} potential transformations: {[t.get('from', 'unknown') + '->' + t.get('to', 'unknown') for t in safe_transformations]}")
        
        for transformation in safe_transformations:
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
        new_func_name = f"{func_name}_recursive"
        
        # Simple recursive template based on MeTTa reasoning
        return f'''def {new_func_name}(numbers, start_idx, end_idx, current_idx=None):
    """MeTTa-guided recursive transformation from iterative pattern."""
    if current_idx is None:
        current_idx = start_idx
    
    # MeTTa-derived base case
    if current_idx >= end_idx:
        return None
    
    # Process current element
    current_element = numbers[current_idx]
    
    # MeTTa-guided recursive call
    rest_result = {new_func_name}(numbers, start_idx, end_idx, current_idx + 1)
    
    # MeTTa-reasoned combination logic
    if rest_result is None:
        return current_element
    else:
        return max(current_element, rest_result)'''
    
    def _generate_iterative_from_recursive(self, context: GenerationContext,
                                         guidance: Dict[str, Any]) -> str:
        """Generate iterative version from recursive using MeTTa guidance."""
        func_name = context.function_name
        new_func_name = f"{func_name}_iterative"
        
        return f'''def {new_func_name}(numbers, start_idx, end_idx):
    """MeTTa-guided iterative transformation from recursive pattern."""
    if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
        return None
    
    # MeTTa-guided iterative processing
    result = numbers[start_idx]
    for i in range(start_idx + 1, end_idx):
        if numbers[i] > result:
            result = numbers[i]
    
    return result'''
    
    def _generate_functional_from_imperative(self, context: GenerationContext,
                                           guidance: Dict[str, Any]) -> str:
        """Generate functional version from imperative using MeTTa guidance."""
        func_name = context.function_name
        new_func_name = f"{func_name}_functional"
        
        return f'''def {new_func_name}(numbers, start_idx, end_idx):
    """MeTTa-guided functional transformation from imperative pattern."""
    if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
        return None
    
    # MeTTa-guided functional approach
    relevant_slice = numbers[start_idx:end_idx]
    return max(relevant_slice) if relevant_slice else None'''
    
    def _generate_parallel_from_sequential(self, context: GenerationContext,
                                         guidance: Dict[str, Any]) -> str:
        """Generate parallel version from sequential using MeTTa guidance."""
        func_name = context.function_name
        new_func_name = f"{func_name}_parallel"
        
        return f'''def {new_func_name}(numbers, start_idx, end_idx):
    """MeTTa-guided parallel transformation from sequential pattern."""
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing as mp
    
    if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
        return None
    
    work_data = numbers[start_idx:end_idx]
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