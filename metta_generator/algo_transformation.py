#!/usr/bin/env python3
"""
Algorithm Transformation Generator Module
Generates donor candidates by transforming algorithmic approaches
"""

from typing import List, Optional
import re
import ast
from metta_generator.base import BaseDonorGenerator, GenerationContext, DonorCandidate, GenerationStrategy

class AlgorithmTransformationGenerator(BaseDonorGenerator):
    """Generator that creates variants by transforming algorithmic approaches."""
    
    def __init__(self):
        self.transformation_types = {
            "iterative_to_recursive": self._transform_iterative_to_recursive,
            "recursive_to_iterative": self._transform_recursive_to_iterative,
            "imperative_to_functional": self._transform_imperative_to_functional,
            "functional_to_imperative": self._transform_functional_to_imperative,
            "sequential_to_parallel": self._transform_sequential_to_parallel,
            "eager_to_lazy": self._transform_eager_to_lazy,
            "divide_and_conquer": self._transform_to_divide_and_conquer,
            "dynamic_programming": self._transform_to_dynamic_programming,
            "greedy_approach": self._transform_to_greedy,
            "backtracking": self._transform_to_backtracking
        }
        
        self.confidence_scores = {
            "iterative_to_recursive": 0.8,
            "recursive_to_iterative": 0.85,
            "imperative_to_functional": 0.9,
            "functional_to_imperative": 0.75,
            "sequential_to_parallel": 0.7,
            "eager_to_lazy": 0.8,
            "divide_and_conquer": 0.7,
            "dynamic_programming": 0.6,
            "greedy_approach": 0.65,
            "backtracking": 0.6
        }
    
    def can_generate(self, context: GenerationContext, strategy: GenerationStrategy) -> bool:
        """Check if this generator can handle the given context and strategy."""
        if strategy != GenerationStrategy.ALGORITHM_TRANSFORMATION:
            return False
        
        # Check if function has algorithmic patterns that can be transformed
        code = context.original_code
        
        # Check for iterative patterns
        has_loops = any(pattern in code for pattern in ["for ", "while "])
        
        # Check for recursive patterns
        func_name = context.function_name
        has_recursion = func_name in code.replace(f"def {func_name}", "")
        
        # Check for functional patterns
        has_functional = any(pattern in code for pattern in ["map(", "filter(", "reduce(", "lambda"])
        
        # Check for sequential processing
        has_sequential = "for " in code and "append" in code
        
        return has_loops or has_recursion or has_functional or has_sequential
    
    def generate_candidates(self, context: GenerationContext, strategy: GenerationStrategy) -> List[DonorCandidate]:
        """Generate algorithm transformation candidates."""
        candidates = []
        
        # Determine applicable transformations
        applicable_transformations = self._get_applicable_transformations(context)
        
        for transformation_name in applicable_transformations:
            if transformation_name in self.transformation_types:
                transformer_func = self.transformation_types[transformation_name]
                candidate = self._create_transformation_candidate(
                    context, transformation_name, transformer_func
                )
                if candidate:
                    candidates.append(candidate)
        
        return candidates
    
    def get_supported_strategies(self) -> List[GenerationStrategy]:
        """Get list of strategies this generator supports."""
        return [GenerationStrategy.ALGORITHM_TRANSFORMATION]
    
    def _get_applicable_transformations(self, context: GenerationContext) -> List[str]:
        """Determine which transformations are applicable to the given context."""
        applicable = []
        code = context.original_code
        func_name = context.function_name
        
        # Check for iterative patterns
        if any(pattern in code for pattern in ["for ", "while "]):
            applicable.append("iterative_to_recursive")
            applicable.append("imperative_to_functional")
            applicable.append("sequential_to_parallel")
            applicable.append("eager_to_lazy")
        
        # Check for recursive patterns
        if func_name in code.replace(f"def {func_name}", ""):
            applicable.append("recursive_to_iterative")
            applicable.append("dynamic_programming")  # Often good for recursive functions
        
        # Check for functional patterns
        if any(pattern in code for pattern in ["map(", "filter(", "reduce(", "lambda"]):
            applicable.append("functional_to_imperative")
        else:
            applicable.append("imperative_to_functional")
        
        # Check for search/optimization patterns
        if any(pattern in func_name.lower() for pattern in ["find", "search", "optimal", "best"]):
            applicable.extend(["divide_and_conquer", "greedy_approach", "backtracking"])
        
        # Check for computation-heavy patterns
        if "calculation" in func_name.lower() or "compute" in func_name.lower():
            applicable.append("dynamic_programming")
        
        return list(set(applicable))  # Remove duplicates
    
    def _create_transformation_candidate(self, context: GenerationContext,
                                       transformation_name: str,
                                       transformer_func: callable) -> Optional[DonorCandidate]:
        """Create an algorithm transformation candidate."""
        try:
            # Apply the transformation
            transformed_code = transformer_func(context.original_code, context.function_name)
            
            if not transformed_code or transformed_code == context.original_code:
                return None
            
            confidence = self.confidence_scores.get(transformation_name, 0.7)
            
            return DonorCandidate(
                name=f"{context.function_name}_{transformation_name}",
                description=f"Algorithm transformation: {transformation_name.replace('_', ' ')}",
                code=transformed_code,
                strategy="algorithm_transformation",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=self._get_data_structures_from_context(context),
                operations_used=self._get_operations_from_context(context) + [transformation_name],
                metta_derivation=[
                    f"(algorithm-transformation {context.function_name} {transformation_name})"
                ],
                confidence=confidence,
                properties=["algorithm-transformed", transformation_name.replace("_", "-")],
                complexity_estimate=self._estimate_complexity_change(transformation_name),
                applicability_scope=self._estimate_applicability_scope(transformation_name)
            )
            
        except Exception as e:
            print(f"      Failed to create transformation candidate {transformation_name}: {e}")
            return None
    
    # Transformation methods
    
    def _transform_iterative_to_recursive(self, code: str, func_name: str) -> str:
        """Transform iterative code to recursive implementation."""
        new_func_name = f"{func_name}_recursive"
        
        # Analyze the iterative structure
        if "for i in range(" in code:
            return self._create_range_based_recursive(code, func_name, new_func_name)
        elif "for " in code and "in " in code:
            return self._create_collection_based_recursive(code, func_name, new_func_name)
        elif "while " in code:
            return self._create_while_based_recursive(code, func_name, new_func_name)
        else:
            return self._create_generic_recursive(code, func_name, new_func_name)
    
    def _create_range_based_recursive(self, code: str, func_name: str, new_func_name: str) -> str:
        """Create recursive version for range-based loops."""
        params = self._extract_parameters(code, func_name)
        
        return f'''def {new_func_name}({', '.join(params)}, index=None):
    """Recursive implementation of {func_name}."""
    if index is None:
        index = {params[1] if len(params) > 1 else '0'}
    
    # Base case
    if index >= {params[2] if len(params) > 2 else 'len(' + params[0] + ')'}:
        return None  # or appropriate base case result
    
    # Process current element
    current_result = {self._extract_loop_body_operation(code)}
    
    # Recursive case
    remaining_result = {new_func_name}({', '.join(params)}, index + 1)
    
    # Combine results
    return current_result if remaining_result is None else {self._get_combination_logic(code)}

# Helper function to combine results
def combine_recursive_results(current, remaining):
    \"\"\"Combine current result with remaining results.\"\"\"
    if remaining is None:
        return current
    return current + remaining  # Adjust based on actual operation'''
    
    def _create_collection_based_recursive(self, code: str, func_name: str, new_func_name: str) -> str:
        """Create recursive version for collection-based loops."""
        params = self._extract_parameters(code, func_name)
        
        return f'''def {new_func_name}({', '.join(params)}):
    """Recursive implementation of {func_name}."""
    # Base case
    if not {params[0]}:
        return {self._get_base_case_value(code)}
    
    # Process first element
    first = {params[0]}[0]
    rest = {params[0]}[1:]
    
    # Process current element
    current_result = {self._extract_loop_body_operation(code, 'first')}
    
    # Recursive call on rest
    rest_result = {new_func_name}(rest{', ' + ', '.join(params[1:]) if len(params) > 1 else ''})
    
    # Combine results
    return {self._get_combination_logic(code, 'current_result', 'rest_result')}'''
    
    def _create_while_based_recursive(self, code: str, func_name: str, new_func_name: str) -> str:
        """Create recursive version for while loops."""
        params = self._extract_parameters(code, func_name)
        condition = self._extract_while_condition(code)
        
        return f'''def {new_func_name}({', '.join(params)}, state=None):
    """Recursive implementation of {func_name}."""
    if state is None:
        state = {self._get_initial_state(code)}
    
    # Base case (negation of while condition)
    if not ({condition}):
        return state
    
    # Process current iteration
    new_state = {self._extract_while_body_operation(code)}
    
    # Recursive call
    return {new_func_name}({', '.join(params)}, new_state)'''
    
    def _transform_recursive_to_iterative(self, code: str, func_name: str) -> str:
        """Transform recursive code to iterative implementation."""
        new_func_name = f"{func_name}_iterative"
        params = self._extract_parameters(code, func_name)
        
        return f'''def {new_func_name}({', '.join(params)}):
    """Iterative implementation of {func_name}."""
    # Use explicit stack to simulate recursion
    stack = [({', '.join(params)})]
    result = None
    
    while stack:
        current_params = stack.pop()
        
        # Base case check
        if {self._extract_base_case_condition(code)}:
            result = {self._extract_base_case_value(code)}
            continue
        
        # Process current level
        {self._extract_recursive_operation(code)}
        
        # Add recursive calls to stack
        {self._generate_stack_operations(code)}
    
    return result'''
    
    def _transform_imperative_to_functional(self, code: str, func_name: str) -> str:
        """Transform imperative code to functional style."""
        new_func_name = f"{func_name}_functional"
        
        # Detect the pattern and create appropriate functional version
        if self._is_search_pattern(code):
            return self._create_functional_search(code, func_name, new_func_name)
        elif self._is_transform_pattern(code):
            return self._create_functional_transform(code, func_name, new_func_name)
        elif self._is_aggregate_pattern(code):
            return self._create_functional_aggregate(code, func_name, new_func_name)
        else:
            return self._create_generic_functional(code, func_name, new_func_name)
    
    def _create_functional_search(self, code: str, func_name: str, new_func_name: str) -> str:
        """Create functional search implementation."""
        params = self._extract_parameters(code, func_name)
        
        return f'''def {new_func_name}({', '.join(params)}):
    """Functional style search implementation."""
    from typing import Optional, Callable
    
    def predicate(item):
        # Extract search condition from original code
        return {self._extract_search_condition(code)}
    
    # Use functional approach
    try:
        return next(filter(predicate, {params[0]}))
    except StopIteration:
        return None'''
    
    def _create_functional_transform(self, code: str, func_name: str, new_func_name: str) -> str:
        """Create functional transform implementation."""
        params = self._extract_parameters(code, func_name)
        
        return f'''def {new_func_name}({', '.join(params)}):
    """Functional style transform implementation."""
    from typing import Callable, Iterable
    
    def transform_fn(item):
        # Extract transformation logic from original code
        return {self._extract_transform_operation(code)}
    
    # Use functional approach
    return list(map(transform_fn, {params[0]}))'''
    
    def _create_functional_aggregate(self, code: str, func_name: str, new_func_name: str) -> str:
        """Create functional aggregate implementation."""
        params = self._extract_parameters(code, func_name)
        
        return f'''def {new_func_name}({', '.join(params)}):
    """Functional style aggregate implementation."""
    from functools import reduce
    from typing import Callable
    
    def combine_fn(acc, item):
        # Extract combination logic from original code
        return {self._extract_aggregate_operation(code)}
    
    # Use functional approach
    initial_value = {self._get_initial_aggregate_value(code)}
    return reduce(combine_fn, {params[0]}, initial_value)'''
    
    def _transform_functional_to_imperative(self, code: str, func_name: str) -> str:
        """Transform functional code to imperative style."""
        new_func_name = f"{func_name}_imperative"
        
        # Replace functional constructs with loops
        imperative_code = code.replace(f"def {func_name}(", f"def {new_func_name}(")
        
        # Replace map with for loop
        imperative_code = re.sub(
            r'list\(map\(([^,]+),\s*([^)]+)\)\)',
            r'[\\1(item) for item in \\2]',
            imperative_code
        )
        
        # Replace filter with for loop
        imperative_code = re.sub(
            r'list\(filter\(([^,]+),\s*([^)]+)\)\)',
            r'[item for item in \\2 if \\1(item)]',
            imperative_code
        )
        
        # Replace reduce with for loop
        imperative_code = re.sub(
            r'reduce\(([^,]+),\s*([^,]+),\s*([^)]+)\)',
            self._create_reduce_loop,
            imperative_code
        )
        
        # Add docstring
        imperative_code = self._add_transformation_docstring(
            imperative_code, "functional", "imperative"
        )
        
        return imperative_code
    
    def _transform_sequential_to_parallel(self, code: str, func_name: str) -> str:
        """Transform sequential code to parallel processing."""
        new_func_name = f"{func_name}_parallel"
        
        params = self._extract_parameters(code, func_name)
        
        return f'''def {new_func_name}({', '.join(params)}, num_workers=None):
    """Parallel processing implementation of {func_name}."""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    import multiprocessing
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    def process_chunk(chunk):
        \"\"\"Process a chunk of data.\"\"\"
        result = []
        for item in chunk:
            # Extract processing logic from original code
            processed = {self._extract_processing_operation(code)}
            result.append(processed)
        return result
    
    # Split data into chunks
    chunk_size = max(1, len({params[0]}) // num_workers)
    chunks = [
        {params[0]}[i:i + chunk_size] 
        for i in range(0, len({params[0]}), chunk_size)
    ]
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        chunk_results = list(executor.map(process_chunk, chunks))
    
    # Combine results
    final_result = []
    for chunk_result in chunk_results:
        final_result.extend(chunk_result)
    
    return {self._get_final_result_processing(code)}'''
    
    def _transform_eager_to_lazy(self, code: str, func_name: str) -> str:
        """Transform eager evaluation to lazy evaluation."""
        new_func_name = f"{func_name}_lazy"
        
        params = self._extract_parameters(code, func_name)
        
        return f'''def {new_func_name}({', '.join(params)}):
    """Lazy evaluation implementation of {func_name}."""
    from typing import Iterator, Generator
    
    def lazy_processor():
        \"\"\"Generator that yields results lazily.\"\"\"
        for item in {params[0]}:
            # Extract processing logic from original code
            result = {self._extract_processing_operation(code)}
            yield result
    
    # Return generator instead of computing all results
    return lazy_processor()

def materialize_lazy_result(lazy_result):
    \"\"\"Helper function to materialize lazy results.\"\"\"
    return list(lazy_result)'''
    
    def _transform_to_divide_and_conquer(self, code: str, func_name: str) -> str:
        """Transform to divide and conquer approach."""
        new_func_name = f"{func_name}_divide_conquer"
        
        params = self._extract_parameters(code, func_name)
        
        return f'''def {new_func_name}({', '.join(params)}):
    """Divide and conquer implementation of {func_name}."""
    # Base case
    if len({params[0]}) <= 1:
        return {self._get_base_case_for_divide_conquer(code)}
    
    # Divide
    mid = len({params[0]}) // 2
    left_half = {params[0]}[:mid]
    right_half = {params[0]}[mid:]
    
    # Conquer
    left_result = {new_func_name}(left_half{', ' + ', '.join(params[1:]) if len(params) > 1 else ''})
    right_result = {new_func_name}(right_half{', ' + ', '.join(params[1:]) if len(params) > 1 else ''})
    
    # Combine
    return {self._get_combine_logic_for_divide_conquer(code)}'''
    
    def _transform_to_dynamic_programming(self, code: str, func_name: str) -> str:
        """Transform to dynamic programming approach."""
        new_func_name = f"{func_name}_dp"
        
        params = self._extract_parameters(code, func_name)
        
        return f'''def {new_func_name}({', '.join(params)}, memo=None):
    """Dynamic programming implementation of {func_name}."""
    if memo is None:
        memo = {{}}
    
    # Create key for memoization
    key = {self._create_memoization_key(params)}
    
    # Check if already computed
    if key in memo:
        return memo[key]
    
    # Base case
    if {self._extract_base_case_condition(code)}:
        result = {self._extract_base_case_value(code)}
        memo[key] = result
        return result
    
    # Recursive case with memoization
    result = {self._extract_dp_recursive_logic(code, new_func_name)}
    memo[key] = result
    return result'''
    
    def _transform_to_greedy(self, code: str, func_name: str) -> str:
        """Transform to greedy algorithm approach."""
        new_func_name = f"{func_name}_greedy"
        
        params = self._extract_parameters(code, func_name)
        
        return f'''def {new_func_name}({', '.join(params)}):
    """Greedy algorithm implementation of {func_name}."""
    result = []
    remaining = list({params[0]})
    
    while remaining:
        # Greedy choice: select best option at each step
        best_option = {self._get_greedy_selection_logic(code)}
        result.append(best_option)
        remaining.remove(best_option)
        
        # Update state if necessary
        {self._get_greedy_state_update(code)}
    
    return {self._get_greedy_final_result(code)}'''
    
    def _transform_to_backtracking(self, code: str, func_name: str) -> str:
        """Transform to backtracking approach."""
        new_func_name = f"{func_name}_backtrack"
        
        params = self._extract_parameters(code, func_name)
        
        return f'''def {new_func_name}({', '.join(params)}):
    """Backtracking implementation of {func_name}."""
    def backtrack(current_solution, remaining_choices):
        # Base case: solution is complete
        if {self._get_backtrack_complete_condition(code)}:
            return current_solution
        
        # Try each possible choice
        for choice in remaining_choices:
            if {self._get_backtrack_valid_condition(code)}:
                # Make choice
                new_solution = current_solution + [choice]
                new_remaining = [c for c in remaining_choices if c != choice]
                
                # Recursively solve
                result = backtrack(new_solution, new_remaining)
                if result is not None:
                    return result
                
                # Backtrack (undo choice) - implicit in recursion
        
        return None  # No solution found
    
    # Start backtracking
    initial_solution = []
    all_choices = list({params[0]})
    return backtrack(initial_solution, all_choices)'''
    
    # Helper methods for code analysis and extraction
    
    def _extract_parameters(self, code: str, func_name: str) -> List[str]:
        """Extract parameter names from function definition."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    return [arg.arg for arg in node.args.args]
        except:
            pass
        return ["data", "param1", "param2"]
    
    def _extract_loop_body_operation(self, code: str, item_name: str = "item") -> str:
        """Extract the main operation from a loop body."""
        # This is a simplified extraction - in practice would need AST analysis
        if "max_val" in code:
            return f"max(current_result, {item_name}) if 'current_result' in locals() else {item_name}"
        elif "sum" in code or "+=" in code:
            return f"current_result + {item_name} if 'current_result' in locals() else {item_name}"
        elif "count" in code:
            return f"current_result + 1 if 'current_result' in locals() else 1"
        else:
            return item_name
    
    def _get_combination_logic(self, code: str, current: str = "current_result", remaining: str = "remaining_result") -> str:
        """Get logic for combining recursive results."""
        if "max" in code:
            return f"max({current}, {remaining}) if {remaining} is not None else {current}"
        elif "sum" in code or "+=" in code:
            return f"{current} + {remaining} if {remaining} is not None else {current}"
        elif "count" in code:
            return f"{current} + {remaining} if {remaining} is not None else {current}"
        else:
            return f"{current} if {remaining} is None else {current} + {remaining}"
    
    def _get_base_case_value(self, code: str) -> str:
        """Get appropriate base case value."""
        if "max" in code:
            return "float('-inf')"
        elif "min" in code:
            return "float('inf')"
        elif "sum" in code or "count" in code:
            return "0"
        elif "list" in code or "append" in code:
            return "[]"
        else:
            return "None"
    
    def _is_search_pattern(self, code: str) -> bool:
        """Check if code follows a search pattern."""
        return any(pattern in code for pattern in ["find", "search", "return", "if"])
    
    def _is_transform_pattern(self, code: str) -> bool:
        """Check if code follows a transform pattern."""
        return any(pattern in code for pattern in ["append", "transform", "map", "convert"])
    
    def _is_aggregate_pattern(self, code: str) -> bool:
        """Check if code follows an aggregate pattern."""
        return any(pattern in code for pattern in ["sum", "count", "total", "+=", "accumulate"])
    
    def _extract_search_condition(self, code: str) -> str:
        """Extract search condition from code."""
        # Simplified extraction
        if ">" in code:
            return "item > threshold"  # placeholder
        elif "==" in code:
            return "item == target"
        else:
            return "True"  # fallback
    
    def _extract_transform_operation(self, code: str) -> str:
        """Extract transformation operation from code."""
        # Simplified extraction
        if "*" in code:
            return "item * 2"  # placeholder
        elif "upper" in code:
            return "item.upper()"
        else:
            return "item"  # identity transformation
    
    def _extract_aggregate_operation(self, code: str) -> str:
        """Extract aggregation operation from code."""
        # Simplified extraction
        if "sum" in code or "+=" in code:
            return "acc + item"
        elif "max" in code:
            return "max(acc, item)"
        elif "count" in code:
            return "acc + 1"
        else:
            return "acc + item"  # default
    
    def _get_initial_aggregate_value(self, code: str) -> str:
        """Get initial value for aggregation."""
        if "max" in code:
            return "float('-inf')"
        elif "min" in code:
            return "float('inf')"
        else:
            return "0"
    
    def _estimate_complexity_change(self, transformation_name: str) -> str:
        """Estimate how complexity changes with transformation."""
        complexity_changes = {
            "iterative_to_recursive": "same",
            "recursive_to_iterative": "same",
            "imperative_to_functional": "same",
            "sequential_to_parallel": "parallel-speedup",
            "eager_to_lazy": "space-optimized",
            "divide_and_conquer": "logarithmic-improvement",
            "dynamic_programming": "time-optimized",
            "greedy_approach": "linear-time",
            "backtracking": "exponential-worst-case"
        }
        return complexity_changes.get(transformation_name, "same")
    
    def _estimate_applicability_scope(self, transformation_name: str) -> str:
        """Estimate applicability scope of transformation."""
        scope_mapping = {
            "iterative_to_recursive": "broad",
            "recursive_to_iterative": "broad", 
            "imperative_to_functional": "broad",
            "sequential_to_parallel": "medium",
            "eager_to_lazy": "medium",
            "divide_and_conquer": "narrow",
            "dynamic_programming": "narrow",
            "greedy_approach": "narrow",
            "backtracking": "narrow"
        }
        return scope_mapping.get(transformation_name, "medium")
    
    def _add_transformation_docstring(self, code: str, from_style: str, to_style: str) -> str:
        """Add documentation about the transformation."""
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                if i + 1 < len(lines) and '"""' in lines[i + 1]:
                    lines[i + 1] = lines[i + 1].replace('"""', f'"""Transformed from {from_style} to {to_style}. ')
                else:
                    lines.insert(i + 1, f'    """Transformed from {from_style} to {to_style}."""')
                break
        
        return '\n'.join(lines)
    
    def _get_primary_pattern_family(self, context: GenerationContext) -> str:
        """Get the primary pattern family from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].pattern_family
        return "generic"
    
    def _get_data_structures_from_context(self, context: GenerationContext) -> List[str]:
        """Get data structures from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].data_structures
        return ["generic"]
    
    def _get_operations_from_context(self, context: GenerationContext) -> List[str]:
        """Get operations from context."""
        if context.detected_patterns:
            return context.detected_patterns[0].operations
        return ["transformation"]
    
    # Additional helper methods would be implemented here for the more complex transformations
    # These are simplified placeholders for demonstration