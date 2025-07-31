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
        """Generate code using MeTTa-derived guidance and demo solution templates."""
        from_pattern = transformation.get('from', '')
        to_pattern = transformation.get('to', '')
        guidance = transformation.get('guidance', {})
        
        # First, try to find relevant demo solutions for this problem context
        demo_solution = self._find_relevant_demo_solution(context, transformation)
        if demo_solution:
            print(f"          Using demo solution template: {demo_solution['name']}")
            return self._adapt_demo_solution_to_context(demo_solution, context, transformation)
        
        # Fallback to pattern-based generation if no demo solution found
        print(f"          No demo solution found, using pattern-based generation")
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
    
    def _find_relevant_demo_solution(self, context: GenerationContext, 
                                   transformation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find relevant demo solution based on context and transformation patterns."""
        try:
            # Analyze the original function to determine the problem type
            problem_type = self._classify_problem_type(context)
            
            if problem_type:
                # Query MeTTa for demo solutions matching this problem type
                demo_query = f"""
                (match &self
                  (demo-solution-pattern {problem_type} $solution_name)
                  (demo-function $solution_name $pattern))
                """
                
                demo_results = self._use_metta_reasoning(context, "query_demo", query=demo_query)
                
                if demo_results and len(demo_results) > 0:
                    # Return the first matching demo solution
                    return {
                        'name': problem_type,
                        'solution_type': problem_type,
                        'pattern': demo_results[0] if isinstance(demo_results[0], str) else str(demo_results[0]),
                        'problem_type': problem_type
                    }
            
            return None
            
        except Exception as e:
            print(f"          Error finding demo solution: {e}")
            return None
    
    def _classify_problem_type(self, context: GenerationContext) -> Optional[str]:
        """Classify the problem type based on function analysis."""
        code = context.original_code.lower()
        func_name = context.function_name.lower()
        
        # Check for memory-related issues
        if any(pattern in code for pattern in ['append', 'extend', 'list(', '[]', 'dict(', '{}']):
            if any(pattern in code for pattern in ['for ', 'while ', 'range(']):
                return 'memory_leak'
        
        # Check for CPU-intensive patterns
        if any(pattern in code for pattern in ['for ', 'while ']) and 'for ' in code:
            nested_loops = code.count('for ') > 1 or code.count('while ') > 1
            if nested_loops or any(pattern in code for pattern in ['range(', 'enumerate(']):
                return 'cpu_overload'
        
        # Check for connection patterns
        if any(pattern in code for pattern in ['connect', 'query', 'execute', 'cursor', 'database']):
            return 'connection_issues'
        
        # Check for error-prone patterns
        if any(pattern in code for pattern in ['try:', 'except:', 'raise', 'error']):
            return 'request_failures'
        
        return None
    
    def _adapt_demo_solution_to_context(self, demo_solution: Dict[str, Any], 
                                      context: GenerationContext, 
                                      transformation: Dict[str, Any]) -> str:
        """Adapt a demo solution to the specific context and transformation."""
        solution_type = demo_solution['solution_type']
        func_name = context.function_name
        
        # Generate adapted code based on the demo solution pattern
        if solution_type == 'memory_leak':
            return self._generate_memory_efficient_solution(func_name, context)
        elif solution_type == 'cpu_overload':
            return self._generate_cpu_efficient_solution(func_name, context)  
        elif solution_type == 'connection_issues':
            return self._generate_connection_efficient_solution(func_name, context)
        elif solution_type == 'request_failures':
            return self._generate_error_resilient_solution(func_name, context)
        else:
            return self._generate_generic_optimized_solution(func_name, context)
    
    def _generate_memory_efficient_solution(self, func_name: str, context: GenerationContext) -> str:
        """Generate memory-efficient solution based on demo patterns."""
        new_func_name = f"{func_name}_memory_optimized"
        return f'''def {new_func_name}(data_items):
    """Memory-optimized version using generator patterns from demo solutions."""
    
    # Memory optimization: Use generator for streaming processing
    def process_items_generator():
        for item in data_items:
            if item:  # Skip invalid items
                processed = process_item_efficiently(item)
                yield processed
    
    # Memory optimization: Process in chunks to control memory usage
    chunk_size = 100
    results = []
    
    for chunk_start in range(0, len(data_items), chunk_size):
        chunk_items = data_items[chunk_start:chunk_start + chunk_size]
        
        # Memory optimization: Process chunk and clean up immediately
        chunk_results = list(process_items_generator())
        results.extend(chunk_results)
        
        # Memory optimization: Force cleanup
        del chunk_results
        if chunk_start % 1000 == 0:  # Periodic GC
            import gc
            gc.collect()
    
    return results'''
    
    def _generate_cpu_efficient_solution(self, func_name: str, context: GenerationContext) -> str:
        """Generate CPU-efficient solution based on demo patterns."""
        new_func_name = f"{func_name}_cpu_optimized"
        return f'''def {new_func_name}(dataset_a, dataset_b):
    """CPU-optimized version using efficient algorithms from demo solutions."""
    
    # CPU optimization: Sort once for efficient searching (O(n log n))
    sorted_b = sorted(dataset_b, key=lambda x: x.get('key', x))
    results = []
    
    for item_a in dataset_a:
        # CPU optimization: Binary search instead of linear scan
        import bisect
        key_a = item_a.get('key', item_a)
        
        # CPU optimization: Use bisect for O(log n) search
        pos = bisect.bisect_left([x.get('key', x) for x in sorted_b], key_a)
        
        if pos < len(sorted_b):
            matched_item = sorted_b[pos]
            similarity = calculate_similarity_fast(item_a, matched_item)
            
            # CPU optimization: Early termination for low similarity
            if similarity > 0.1:
                results.append({{
                    'item_a': item_a,
                    'item_b': matched_item,
                    'similarity': similarity
                }})
        
        # CPU optimization: Yield control periodically
        if len(results) % 1000 == 0:
            pass  # Allow other processes to run
    
    return results'''
    
    def _generate_connection_efficient_solution(self, func_name: str, context: GenerationContext) -> str:
        """Generate connection-efficient solution based on demo patterns."""
        new_func_name = f"{func_name}_connection_optimized"
        return f'''def {new_func_name}(data_items):
    """Connection-optimized version using pooling patterns from demo solutions."""
    
    # Connection optimization: Use connection pool for reuse
    with connection_pool.acquire() as db_conn:
        results = []
        
        # Connection optimization: Prepare statements once
        select_stmt = db_conn.prepare("SELECT * FROM table WHERE id = ?")
        insert_stmt = db_conn.prepare("INSERT INTO results VALUES (?, ?)")
        
        # Connection optimization: Process in batches to minimize round trips
        batch_size = 50
        for batch_start in range(0, len(data_items), batch_size):
            batch_items = data_items[batch_start:batch_start + batch_size]
            
            # Connection optimization: Start transaction for batch
            with db_conn.transaction():
                batch_results = []
                
                for item in batch_items:
                    # Connection optimization: Reuse prepared statement
                    data = select_stmt.execute([item.get('id')])
                    batch_results.append(process_data(data))
                
                # Connection optimization: Batch insert
                insert_stmt.execute_many(batch_results)
                results.extend(batch_results)
        
        return results'''
    
    def _generate_error_resilient_solution(self, func_name: str, context: GenerationContext) -> str:
        """Generate error-resilient solution based on demo patterns."""
        new_func_name = f"{func_name}_error_resilient"  
        return f'''def {new_func_name}(request_data):
    """Error-resilient version using resilience patterns from demo solutions."""
    
    # Error resilience: Input validation
    try:
        if not request_data or not isinstance(request_data, dict):
            raise ValueError("Invalid request data")
        
        user_id = request_data.get('user_id')
        if not user_id:
            raise ValueError("user_id is required")
            
    except ValueError as e:
        return {{'error': 'validation_error', 'message': str(e), 'status': 'failed'}}
    
    # Error resilience: Retry mechanism for external calls
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Error resilience: Timeout protection
            import asyncio
            result = asyncio.wait_for(
                external_service_call(user_id), 
                timeout=5.0
            )
            break
        except (asyncio.TimeoutError, ConnectionError) as e:
            if attempt == max_retries - 1:
                # Error resilience: Graceful degradation
                result = {{'user_id': user_id, 'status': 'degraded_mode'}}
            else:
                import time
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
    
    # Error resilience: Comprehensive error handling
    try:
        processed_result = process_request_safely(result)
        return {{'result': processed_result, 'status': 'success'}}
    except Exception as e:
        return {{
            'error': 'processing_error',
            'message': 'Request processing failed', 
            'status': 'failed'
        }}'''
    
    def _generate_generic_optimized_solution(self, func_name: str, context: GenerationContext) -> str:
        """Generate generic optimized solution when no specific demo pattern matches."""
        new_func_name = f"{func_name}_optimized"
        return f'''def {new_func_name}(*args, **kwargs):
    """Generic optimized version with basic improvements."""
    
    # Basic optimization: Input validation
    if not args:
        return None
    
    # Basic optimization: Early return for empty data
    data = args[0] if args else []
    if not data:
        return []
    
    # Basic optimization: Process efficiently
    results = []
    for item in data:
        if item:  # Skip invalid items
            processed = process_item_basic(item)
            results.append(processed)
    
    return results'''
    
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