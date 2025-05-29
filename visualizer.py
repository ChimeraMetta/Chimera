#!/usr/bin/env python3
"""
Donor Generation Process Visualizer
Shows how the system iteratively generates candidates and tests them against 
the original function's constraints until it finds successful solutions.
"""

import time
import json
import inspect
import os
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Any, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Try to import our components
try:
    from executors.metta_generator import integrate_metta_generation, GenerationStrategy
    from reflectors.static_analyzer import decompose_function
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    print("Warning: MeTTa components not available, using simulation")

class TestResult(Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"

@dataclass
class CandidateTest:
    """Represents a test case and its result."""
    test_input: Tuple
    expected_output: Any
    actual_output: Any
    result: TestResult
    error_message: str = ""

@dataclass
class GenerationEvent:
    """Represents a candidate generation and testing event."""
    timestamp: float
    iteration: int
    candidate_name: str
    strategy: str
    confidence: float
    test_results: List[CandidateTest]
    constraints_satisfied: int
    total_constraints: int
    success_rate: float
    execution_time: float

class ConstraintBasedTester:
    """Tests candidates against the original function's constraints."""
    
    def __init__(self, original_function):
        self.original_function = original_function
        self.test_cases = self._generate_test_cases()
        
    def _generate_test_cases(self):
        """Generate comprehensive test cases based on the original function."""
        test_cases = []
        
        # Test data sets
        test_data = [
            [1, 5, 3, 9, 2, 7, 4],      # Normal case
            [10, 20, 30, 40, 50],       # Ascending
            [50, 40, 30, 20, 10],       # Descending  
            [5, 5, 5, 5, 5],            # All equal
            [42],                        # Single element
            [1, 2],                     # Two elements
            list(range(100))            # Large dataset
        ]
        
        for data in test_data:
            # Valid range tests
            test_cases.extend([
                ((data, 0, len(data)), self.original_function(data, 0, len(data))),
                ((data, 1, len(data)-1), self.original_function(data, 1, len(data)-1)),
                ((data, 0, len(data)//2), self.original_function(data, 0, len(data)//2)),
            ])
            
            # Edge cases
            if len(data) > 2:
                test_cases.extend([
                    ((data, 0, 1), self.original_function(data, 0, 1)),  # Single element
                    ((data, len(data)-2, len(data)), self.original_function(data, len(data)-2, len(data))),
                ])
            
            # Boundary tests  
            test_cases.extend([
                ((data, -1, 2), self.original_function(data, -1, 2)),    # Negative start
                ((data, 0, len(data)+5), self.original_function(data, 0, len(data)+5)),  # Beyond end
                ((data, 5, 2), self.original_function(data, 5, 2)),      # Start > end
            ])
        
        return test_cases
    
    def test_candidate(self, candidate_func) -> List[CandidateTest]:
        """Test a candidate function against all constraints."""
        results = []
        
        for test_input, expected_output in self.test_cases:
            try:
                actual_output = candidate_func(*test_input)
                
                if actual_output == expected_output:
                    result = TestResult.PASSED
                    error_msg = ""
                else:
                    result = TestResult.FAILED
                    error_msg = f"Expected {expected_output}, got {actual_output}"
                    
            except Exception as e:
                result = TestResult.ERROR
                actual_output = None
                error_msg = str(e)
            
            results.append(CandidateTest(
                test_input=test_input,
                expected_output=expected_output,
                actual_output=actual_output,
                result=result,
                error_message=error_msg
            ))
        
        return results

class DonorGenerationVisualizer:
    """Visualizes the iterative donor generation and constraint satisfaction process."""
    
    def __init__(self, original_function):
        self.original_function = original_function
        self.original_function_code = self._get_function_source(original_function)
        self.tester = ConstraintBasedTester(original_function)
        self.events: List[GenerationEvent] = []
        self.generation_start_time = time.time()
        self.successful_candidate_codes = []
        
        # Setup visualization
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle(f"Donor Generation Evolution: {original_function.__name__}", fontsize=16)
        self.save_plots = True
        self.plot_save_dir = "evolution_plots"
        self._ensure_plot_directory()
        
        # Initialize plots
        self._setup_plots()
        
    def _setup_plots(self):
        """Setup the visualization plots."""
        # Plot 1: Constraint satisfaction over time
        self.axes[0, 0].set_title("Constraint Satisfaction Progress")
        self.axes[0, 0].set_xlabel("Iteration")
        self.axes[0, 0].set_ylabel("Constraints Satisfied (%)")
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Success rate by strategy
        self.axes[0, 1].set_title("Success Rate by Strategy")
        self.axes[0, 1].set_xlabel("Strategy")
        self.axes[0, 1].set_ylabel("Success Rate (%)")
        
        # Plot 3: Confidence vs Performance
        self.axes[1, 0].set_title("Confidence vs Actual Performance")  
        self.axes[1, 0].set_xlabel("Predicted Confidence")
        self.axes[1, 0].set_ylabel("Actual Success Rate")
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Evolution timeline
        self.axes[1, 1].set_title("Generation Timeline")
        self.axes[1, 1].set_xlabel("Time (seconds)")
        self.axes[1, 1].set_ylabel("Candidates")
    
    def _ensure_plot_directory(self):
        """Create directory for saving plots if it doesn't exist."""
        if not os.path.exists(self.plot_save_dir):
            os.makedirs(self.plot_save_dir)
            print(f"{Fore.YELLOW}Created directory: {self.plot_save_dir}")
    
    def _get_function_source(self, func):
        """Get the source code of a function."""
        try:
            return inspect.getsource(func)
        except (OSError, TypeError):
            # Fallback for functions defined in REPL or other edge cases
            return f"def {func.__name__}(...):\n    # Source code not available"
    
    def _format_code_for_display(self, code, title=""):
        """Format code for nice display with proper indentation."""
        lines = code.strip().split('\n')
        formatted_lines = []
        
        if title:
            formatted_lines.append(f"# {title}")
            formatted_lines.append("#" + "="*50)
        
        for line in lines:
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
        
    def run_evolution_process(self, max_iterations=8, target_success_rate=0.8):
        """
        Run the iterative evolution process until we find good solutions.
        """
        print(f"{Fore.CYAN}STARTING CONSTRAINT-BASED DONOR EVOLUTION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Original function: {self.original_function.__name__}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Test cases: {len(self.tester.test_cases)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Target success rate: {target_success_rate:.1%}{Style.RESET_ALL}")
        
        successful_candidates = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            iteration_start = time.time()
            
            print(f"\\n{Fore.MAGENTA}ITERATION {iteration}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}{'-' * 40}{Style.RESET_ALL}")
            
            # Generate candidates for this iteration
            candidates = self._generate_candidates_for_iteration(iteration)
            print(f"Generated {len(candidates)} candidates")
            
            # Test each candidate
            for candidate in candidates:
                event = self._test_and_record_candidate(candidate, iteration, iteration_start)
                self.events.append(event)
                
                success_rate = event.success_rate
                constraints_met = event.constraints_satisfied
                
                print(f"  {candidate['name']}: {Fore.BLUE}{constraints_met}{Style.RESET_ALL}/{event.total_constraints} constraints ({Fore.BLUE}{success_rate:.1%}{Style.RESET_ALL})")
                
                if success_rate >= target_success_rate:
                    successful_candidates.append((candidate, event))
                    # Store the successful candidate's code
                    self.successful_candidate_codes.append({
                        'name': candidate['name'],
                        'code': candidate['code'],
                        'strategy': candidate['strategy'],
                        'success_rate': success_rate,
                        'iteration': iteration,
                        'constraints_satisfied': constraints_met,
                        'total_constraints': event.total_constraints
                    })
                    print(f"  {Fore.GREEN}SUCCESS: {candidate['name']} meets target!{Style.RESET_ALL}")
                
            # Update visualization
            self._update_plots()
            
            # Check if we should continue
            best_success_rate = max([e.success_rate for e in self.events], default=0)
            
            print(f"\\n{Fore.YELLOW}Iteration {iteration} Summary:{Style.RESET_ALL}")
            print(f"  Best success rate so far: {Fore.YELLOW}{best_success_rate:.1%}{Style.RESET_ALL}")
            print(f"  Successful candidates: {Fore.YELLOW}{len(successful_candidates)}{Style.RESET_ALL}")
            
            if len(successful_candidates) >= 2:
                print(f"\\n{Fore.GREEN}Found {len(successful_candidates)} successful candidates!{Style.RESET_ALL}")
                break
                
            # Brief pause for visualization
            plt.pause(1.0)
        
        print(f"\\n{Fore.CYAN}EVOLUTION COMPLETE{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Iterations: {iteration}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Total candidates tested: {len(self.events)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Successful candidates: {len(successful_candidates)}{Style.RESET_ALL}")
        
        return successful_candidates
    
    def _generate_candidates_for_iteration(self, iteration):
        """Generate syntactically valid candidates for testing."""
        
        # Try real generation first
        if COMPONENTS_AVAILABLE and iteration <= 2:
            try:
                real_candidates = integrate_metta_generation(self.original_function)
                if real_candidates:
                    # Fix any syntax issues in real candidates
                    fixed_candidates = []
                    for candidate in real_candidates:
                        if self._is_syntactically_valid(candidate['code']):
                            fixed_candidates.append(candidate)
                        else:
                            # Create a fixed version
                            fixed_code = self._fix_candidate_code(candidate)
                            if fixed_code:
                                candidate['code'] = fixed_code
                                fixed_candidates.append(candidate)
                    
                    if fixed_candidates:
                        return fixed_candidates
            except Exception as e:
                print(f"  {Fore.RED}Real generation failed: {e}{Style.RESET_ALL}")
        
        # Generate synthetic candidates with increasing sophistication
        return self._generate_synthetic_candidates(iteration)
    
    def _generate_synthetic_candidates(self, iteration):
        """Generate synthetic candidates that become more sophisticated over time."""
        strategies = [
            "operation_substitution", "accumulator_variation", "structure_preservation",
            "condition_variation", "property_guided", "pattern_expansion"
        ]
        
        candidates = []
        num_candidates = min(3 + iteration, 6)  # 4, 5, 6, 6, 6, 6...
        
        for i in range(num_candidates):
            strategy = strategies[i % len(strategies)]
            
            # Generate increasingly sophisticated candidates
            candidate = {
                "name": f"{self.original_function.__name__}_{strategy}_v{iteration}_{i}",
                "description": f"Iteration {iteration}: {strategy} approach",  
                "strategy": strategy,
                "confidence": 0.2 + (iteration * 0.1) + (i * 0.05),
                "final_score": 0.3 + (iteration * 0.08),
                "properties": [strategy, "synthetic"],
                "code": self._generate_valid_code(strategy, iteration, i),
                "metta_derivation": [f"(synthetic-generation {strategy} iter-{iteration})"]
            }
            candidates.append(candidate)
        
        return candidates
    
    def _generate_valid_code(self, strategy, iteration, variant):
        """Generate syntactically valid code that attempts to solve the same problem."""
        func_name = f"{self.original_function.__name__}_{strategy}_v{iteration}_{variant}"
        
        # Get original function parameters
        import inspect
        try:
            sig = inspect.signature(self.original_function)
            params = list(sig.parameters.keys())
        except:
            params = ["numbers", "start_idx", "end_idx"]
        
        param_str = ", ".join(params)
        data_param, start_param, end_param = params[0], params[1], params[2]
        
        # Generate different approaches based on strategy and iteration
        templates = {
            "operation_substitution": self._template_operation_substitution,
            "accumulator_variation": self._template_accumulator_variation,
            "structure_preservation": self._template_structure_preservation,
            "condition_variation": self._template_condition_variation,
            "property_guided": self._template_property_guided,
            "pattern_expansion": self._template_pattern_expansion
        }
        
        template_func = templates.get(strategy, self._template_basic)
        return template_func(func_name, param_str, data_param, start_param, end_param, iteration, variant)
    
    def _template_operation_substitution(self, func_name, param_str, data_param, start_param, end_param, iteration, variant):
        """Template for operation substitution candidates."""
        operations = [
            ("min", "<", "float('inf')"),
            ("max", ">", "float('-inf')"),  
            ("sum", "+", "0"),
            ("product", "*", "1")
        ]
        
        op_name, comparator, init_val = operations[variant % len(operations)]
        
        if iteration >= 3:  # More sophisticated versions in later iterations
            return f'''def {func_name}({param_str}):
    """Find {op_name} value using operation substitution."""
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return None if "{op_name}" in ["min", "max"] else {init_val}
    
    result = {data_param}[{start_param}]
    for i in range({start_param} + 1, {end_param}):
        if "{op_name}" == "sum":
            result += {data_param}[i]
        elif "{op_name}" == "product":
            result *= {data_param}[i]
        elif {data_param}[i] {comparator} result:
            result = {data_param}[i]
    
    return result'''
        else:
            # Simpler early versions that may not handle all edge cases
            return f'''def {func_name}({param_str}):
    """Simple {op_name} finder."""
    result = {init_val}
    for i in range({start_param}, {end_param}):
        if "{op_name}" == "sum":
            result += {data_param}[i]
        elif "{op_name}" == "product":
            result *= {data_param}[i]
        elif {data_param}[i] {comparator} result:
            result = {data_param}[i]
    return result'''
    
    def _template_accumulator_variation(self, func_name, param_str, data_param, start_param, end_param, iteration, variant):
        """Template for accumulator variation candidates."""
        accumulators = ["sum", "count", "average", "product"]
        acc_type = accumulators[variant % len(accumulators)]
        
        if acc_type == "sum":
            return f'''def {func_name}({param_str}):
    """Sum accumulator variation."""
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return 0
    
    total = 0
    for i in range({start_param}, {end_param}):
        total += {data_param}[i]
    return total'''
        
        elif acc_type == "count":
            return f'''def {func_name}({param_str}):
    """Count accumulator variation."""
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return 0
    
    return {end_param} - {start_param}'''
        
        elif acc_type == "average":
            return f'''def {func_name}({param_str}):
    """Average accumulator variation."""
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return None
    
    total = sum({data_param}[{start_param}:{end_param}])
    count = {end_param} - {start_param}
    return total / count if count > 0 else None'''
        
        else:  # product
            return f'''def {func_name}({param_str}):
    """Product accumulator variation."""
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return 1
    
    result = 1
    for i in range({start_param}, {end_param}):
        result *= {data_param}[i]
    return result'''
    
    def _template_structure_preservation(self, func_name, param_str, data_param, start_param, end_param, iteration, variant):
        """Template for structure preservation candidates."""
        variations = ["index", "position", "element_info", "range_info"]
        var_type = variations[variant % len(variations)]
        
        if var_type == "index":
            return f'''def {func_name}({param_str}):
    """Return index instead of value."""
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return -1
    
    max_val = {data_param}[{start_param}]
    max_idx = {start_param}
    for i in range({start_param} + 1, {end_param}):
        if {data_param}[i] > max_val:
            max_val = {data_param}[i]
            max_idx = i
    return max_idx'''
        
        else:
            return f'''def {func_name}({param_str}):
    """Structure preservation with {var_type}."""
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return None
    
    # Preserve original structure but return different info
    result = {data_param}[{start_param}]
    for i in range({start_param} + 1, {end_param}):
        if {data_param}[i] > result:
            result = {data_param}[i]
    
    return (result, {start_param}, {end_param})  # Value with range info'''
    
    def _template_condition_variation(self, func_name, param_str, data_param, start_param, end_param, iteration, variant):
        """Template for condition variation candidates."""
        variations = ["threshold", "predicate", "filter", "conditional"]
        var_type = variations[variant % len(variations)]
        
        if var_type == "threshold":
            return f'''def {func_name}({param_str}, threshold=0):
    """Find elements above threshold."""
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return []
    
    result = []
    for i in range({start_param}, {end_param}):
        if {data_param}[i] > threshold:
            result.append({data_param}[i])
    
    return result'''
        
        elif var_type == "predicate":
            return f'''def {func_name}({param_str}, predicate=None):
    """Find first element matching predicate."""
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return None
    
    if predicate is None:
        predicate = lambda x: x > 0
    
    for i in range({start_param}, {end_param}):
        if predicate({data_param}[i]):
            return {data_param}[i]
    
    return None'''
        
        else:
            return f'''def {func_name}({param_str}):
    """Conditional variation."""
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return None
    
    result = {data_param}[{start_param}]
    for i in range({start_param} + 1, {end_param}):
        if {data_param}[i] > result:
            result = {data_param}[i]
    
    return result if result > 0 else None'''
    
    def _template_property_guided(self, func_name, param_str, data_param, start_param, end_param, iteration, variant):
        """Template for property-guided candidates."""
        properties = ["bounds_safe", "null_safe", "type_safe", "enhanced"]
        prop_type = properties[variant % len(properties)]
        
        if prop_type == "bounds_safe":
            return f'''def {func_name}({param_str}):
    """Enhanced bounds checking variant."""
    # Enhanced bounds checking
    if not {data_param} or not isinstance({data_param}, list):
        return None
    if not isinstance({start_param}, int) or not isinstance({end_param}, int):
        return None
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return None
    
    max_val = {data_param}[{start_param}]
    for i in range({start_param} + 1, {end_param}):
        if {data_param}[i] > max_val:
            max_val = {data_param}[i]
    
    return max_val'''
        
        elif prop_type == "null_safe":
            return f'''def {func_name}({param_str}):
    """Null-safe variant."""
    if not {data_param}:
        return None
    if {start_param} is None or {end_param} is None:
        return None
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return None
    
    max_val = {data_param}[{start_param}]
    for i in range({start_param} + 1, {end_param}):
        if {data_param}[i] is not None and (max_val is None or {data_param}[i] > max_val):
            max_val = {data_param}[i]
    
    return max_val'''
        
        else:
            return f'''def {func_name}({param_str}):
    """Enhanced property-guided variant."""
    try:
        if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
            return None
        
        max_val = {data_param}[{start_param}]
        for i in range({start_param} + 1, {end_param}):
            if {data_param}[i] > max_val:
                max_val = {data_param}[i]
        
        return max_val
    except (IndexError, TypeError, ValueError):
        return None'''
    
    def _template_pattern_expansion(self, func_name, param_str, data_param, start_param, end_param, iteration, variant):
        """Template for pattern expansion candidates."""
        patterns = ["windowed", "chunked", "multi_range", "recursive"]
        pattern_type = patterns[variant % len(patterns)]
        
        if pattern_type == "windowed":
            return f'''def {func_name}({param_str}, window_size=3):
    """Windowed pattern expansion."""
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return None
    
    if window_size <= 0 or window_size > ({end_param} - {start_param}):
        window_size = {end_param} - {start_param}
    
    max_val = None
    for start in range({start_param}, {end_param} - window_size + 1):
        window_max = {data_param}[start]
        for i in range(start + 1, start + window_size):
            if {data_param}[i] > window_max:
                window_max = {data_param}[i]
        
        if max_val is None or window_max > max_val:
            max_val = window_max
    
    return max_val'''
        
        elif pattern_type == "chunked":
            return f'''def {func_name}({param_str}, chunk_size=5):
    """Chunked pattern expansion."""
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return None
    
    chunk_maxes = []
    for i in range({start_param}, {end_param}, chunk_size):
        chunk_end = min(i + chunk_size, {end_param})
        chunk_max = {data_param}[i]
        for j in range(i + 1, chunk_end):
            if {data_param}[j] > chunk_max:
                chunk_max = {data_param}[j]
        chunk_maxes.append(chunk_max)
    
    return max(chunk_maxes) if chunk_maxes else None'''
        
        else:
            return f'''def {func_name}({param_str}):
    """Multi-range pattern expansion."""
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return None
    
    # Process in multiple sub-ranges
    range_size = {end_param} - {start_param}
    if range_size <= 2:
        max_val = {data_param}[{start_param}]
        for i in range({start_param} + 1, {end_param}):
            if {data_param}[i] > max_val:
                max_val = {data_param}[i]
        return max_val
    
    mid = {start_param} + range_size // 2
    left_max = {func_name}({data_param}, {start_param}, mid)
    right_max = {func_name}({data_param}, mid, {end_param})
    
    return max(left_max, right_max) if left_max is not None and right_max is not None else None'''
    
    def _template_basic(self, func_name, param_str, data_param, start_param, end_param, iteration, variant):
        """Basic template fallback."""
        return f'''def {func_name}({param_str}):
    """Basic iteration {iteration} candidate."""
    if {start_param} < 0 or {end_param} > len({data_param}) or {start_param} >= {end_param}:
        return None
    
    return {data_param}[{start_param}]  # Simple fallback'''
    
    def _is_syntactically_valid(self, code):
        """Check if code is syntactically valid."""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def _fix_candidate_code(self, candidate):
        """Attempt to fix syntactic issues in candidate code."""
        # This would implement simple fixes for common syntax errors
        # For now, return None to skip broken candidates
        return None
    
    def _test_and_record_candidate(self, candidate, iteration, iteration_start):
        """Test a candidate and record the results."""
        # Execute the candidate code
        try:
            exec_namespace = {}
            exec(candidate['code'], exec_namespace)
            candidate_func = exec_namespace[candidate['name']]
        except Exception as e:
            # If we can't even execute the code, mark all tests as errors
            error_results = [
                CandidateTest(
                    test_input=test_case[0],
                    expected_output=test_case[1], 
                    actual_output=None,
                    result=TestResult.ERROR,
                    error_message=f"Execution error: {e}"
                ) for test_case in self.tester.test_cases
            ]
            
            return GenerationEvent(
                timestamp=time.time() - self.generation_start_time,
                iteration=iteration,
                candidate_name=candidate['name'],
                strategy=candidate['strategy'],
                confidence=candidate['confidence'],
                test_results=error_results,
                constraints_satisfied=0,
                total_constraints=len(self.tester.test_cases),
                success_rate=0.0,
                execution_time=time.time() - iteration_start
            )
        
        # Test the candidate
        test_results = self.tester.test_candidate(candidate_func)
        
        # Calculate metrics
        passed_tests = sum(1 for r in test_results if r.result == TestResult.PASSED)
        total_tests = len(test_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        return GenerationEvent(
            timestamp=time.time() - self.generation_start_time,
            iteration=iteration,
            candidate_name=candidate['name'],
            strategy=candidate['strategy'],
            confidence=candidate['confidence'],
            test_results=test_results,
            constraints_satisfied=passed_tests,
            total_constraints=total_tests,
            success_rate=success_rate,
            execution_time=time.time() - iteration_start
        )
    
    def _update_plots(self):
        """Update the visualization plots with current data."""
        if not self.events:
            return
            
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
        
        self._setup_plots()
        
        # Plot 1: Constraint satisfaction progress
        iterations = [e.iteration for e in self.events]
        success_rates = [e.success_rate * 100 for e in self.events]
        strategies = [e.strategy for e in self.events]
        
        # Color by strategy
        strategy_colors = {
            'operation_substitution': 'red',
            'accumulator_variation': 'blue', 
            'structure_preservation': 'green',
            'condition_variation': 'orange',
            'property_guided': 'purple',
            'pattern_expansion': 'brown'
        }
        
        colors = [strategy_colors.get(s, 'gray') for s in strategies]
        
        self.axes[0, 0].scatter(iterations, success_rates, c=colors, alpha=0.7, s=50)
        
        # Add trend line
        if len(iterations) > 1:
            z = np.polyfit(iterations, success_rates, 1)
            p = np.poly1d(z)
            self.axes[0, 0].plot(iterations, p(iterations), "r--", alpha=0.8)
        
        # Plot 2: Success rate by strategy
        strategy_stats = {}
        for event in self.events:
            if event.strategy not in strategy_stats:
                strategy_stats[event.strategy] = []
            strategy_stats[event.strategy].append(event.success_rate)
        
        strategies = list(strategy_stats.keys())
        avg_success_rates = [np.mean(strategy_stats[s]) * 100 for s in strategies]
        
        bars = self.axes[0, 1].bar(range(len(strategies)), avg_success_rates, 
                                  color=[strategy_colors.get(s, 'gray') for s in strategies])
        self.axes[0, 1].set_xticks(range(len(strategies)))
        self.axes[0, 1].set_xticklabels([s.replace('_', '\n') for s in strategies], rotation=45, ha='right')
        
        # Plot 3: Confidence vs Performance
        confidences = [e.confidence for e in self.events]
        actual_performance = [e.success_rate for e in self.events]
        
        self.axes[1, 0].scatter(confidences, actual_performance, 
                               c=colors, alpha=0.7, s=50)
        
        # Add diagonal line (perfect prediction)
        min_val, max_val = 0, 1
        self.axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Plot 4: Timeline
        timestamps = [e.timestamp for e in self.events]
        
        self.axes[1, 1].scatter(timestamps, range(len(timestamps)), 
                               c=success_rates, cmap='RdYlGn', s=60, alpha=0.8)
        
        plt.tight_layout()

        if self.save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            iteration_num = max(iterations) if iterations else 0
            filename = f"{self.plot_save_dir}/evolution_iter_{iteration_num:02d}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"    {Fore.GREEN}Plot saved: {filename}{Style.RESET_ALL}")

        plt.draw()
        
    def show_final_summary(self, successful_candidates):
        """Show final summary with original function code and successful candidate codes."""
        print("\\n" + Fore.CYAN + "="*80 + Style.RESET_ALL)
        print(Fore.CYAN + "FINAL EVOLUTION SUMMARY WITH CODE" + Style.RESET_ALL)
        print(Fore.CYAN + "="*80 + Style.RESET_ALL)
        
        # Display original function code
        print("\\n" + Fore.YELLOW + "ORIGINAL FUNCTION:" + Style.RESET_ALL)
        print(Fore.YELLOW + "="*50 + Style.RESET_ALL)
        print(self._format_code_for_display(self.original_function_code))
        
        print(f"\\n{Fore.YELLOW}Successful Candidates Found: {len(successful_candidates)}{Style.RESET_ALL}")
        
        # Display each successful candidate's code
        for i, candidate_info in enumerate(self.successful_candidate_codes, 1):
            print(f"\\n" + Fore.GREEN + "="*80 + Style.RESET_ALL)
            print(f"{Fore.GREEN}SUCCESSFUL CANDIDATE #{i}: {candidate_info['name']}{Style.RESET_ALL}")
            print(Fore.GREEN + "="*80 + Style.RESET_ALL)
            print(f"{Fore.GREEN}Strategy: {candidate_info['strategy']}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Success Rate: {candidate_info['success_rate']:.1%}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Constraints Satisfied: {candidate_info['constraints_satisfied']}/{candidate_info['total_constraints']}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Generated in Iteration: {candidate_info['iteration']}{Style.RESET_ALL}")
            print("\\n" + Fore.GREEN + "CODE:" + Style.RESET_ALL)
            print(Fore.GREEN + "-"*50 + Style.RESET_ALL)
            print(candidate_info['code'])
            print(Fore.GREEN + "-"*50 + Style.RESET_ALL)
        
        # Show evolution statistics (existing code)
        if self.events:
            initial_success = self.events[0].success_rate
            final_success = max(e.success_rate for e in self.events)
            
            print(f"\\n" + Fore.BLUE + "="*80 + Style.RESET_ALL)
            print(Fore.BLUE + "EVOLUTION STATISTICS" + Style.RESET_ALL)
            print(Fore.BLUE + "="*80 + Style.RESET_ALL)
            print(f"{Fore.BLUE}Initial success rate: {initial_success:.1%}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Final success rate: {final_success:.1%}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Improvement: {final_success - initial_success:.1%}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Total candidates tested: {len(self.events)}{Style.RESET_ALL}")
            
            strategy_performance = {}
            for event in self.events:
                if event.strategy not in strategy_performance:
                    strategy_performance[event.strategy] = []
                strategy_performance[event.strategy].append(event.success_rate)
            
            print(f"\\n{Fore.BLUE}Strategy Performance:{Style.RESET_ALL}")
            for strategy, rates in strategy_performance.items():
                avg_rate = np.mean(rates)
                best_rate = max(rates)
                count = len(rates)
                print(f"  {Fore.BLUE}{strategy}: avg={avg_rate:.1%}, best={best_rate:.1%} ({count} candidates){Style.RESET_ALL}")
        
        # Show constraint analysis (existing code)
        print(f"\\n{Fore.BLUE}Constraint Analysis:{Style.RESET_ALL}")
        print(f"  {Fore.BLUE}Total test cases: {len(self.tester.test_cases)}{Style.RESET_ALL}")
        
        # Find most commonly failed constraints
        if self.events:
            constraint_failures = {}
            for event in self.events:
                for test in event.test_results:
                    if test.result == TestResult.FAILED:
                        key = str(test.test_input)
                        constraint_failures[key] = constraint_failures.get(key, 0) + 1
            
            if constraint_failures:
                print("  Most challenging constraints:")
                sorted_failures = sorted(constraint_failures.items(), key=lambda x: x[1], reverse=True)
                for constraint, failure_count in sorted_failures[:3]:
                    print(f"    {Fore.RED}{constraint}: {failure_count} failures{Style.RESET_ALL}")

    def save_evolution_data_with_code(self, filename="evolution_data_with_code.json"):
        """Save evolution data including successful candidate codes."""
        data = {
            "original_function": {
                "name": self.original_function.__name__,
                "code": self.original_function_code
            },
            "total_events": len(self.events),
            "successful_candidates": self.successful_candidate_codes,
            "events": []
        }
        
        for event in self.events:
            event_data = {
                "timestamp": event.timestamp,
                "iteration": event.iteration,
                "candidate_name": event.candidate_name,
                "strategy": event.strategy,
                "confidence": event.confidence,
                "success_rate": event.success_rate,
                "constraints_satisfied": event.constraints_satisfied,
                "total_constraints": event.total_constraints,
                "execution_time": event.execution_time
            }
            data["events"].append(event_data)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\\n{Fore.GREEN}Evolution data with code saved to {filename}{Style.RESET_ALL}")

    def save_final_plot(self, filename=None):
        """Save a final comprehensive plot with all evolution data."""
        if not self.events:
            print("No data to plot")
            return
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.plot_save_dir}/final_evolution_{self.original_function.__name__}_{timestamp}.png"
        
        # Update plots one final time
        self._update_plots()
        
        # Add title with summary statistics
        if self.successful_candidate_codes:
            success_count = len(self.successful_candidate_codes)
            best_rate = max(c['success_rate'] for c in self.successful_candidate_codes)
            self.fig.suptitle(f"Donor Evolution: {self.original_function.__name__} "
                            f"({success_count} successful, best: {best_rate:.1%})", 
                            fontsize=16)
        
        # Save high-quality version
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"{Fore.GREEN}Final comprehensive plot saved: {filename}{Style.RESET_ALL}")
        
        return filename

    def save_strategy_analysis_plots(self):
        """Save detailed plots analyzing each strategy's performance."""
        if not self.events:
            return
        
        # Group events by strategy
        strategy_data = {}
        for event in self.events:
            if event.strategy not in strategy_data:
                strategy_data[event.strategy] = []
            strategy_data[event.strategy].append(event)
        
        # Create a plot for each strategy
        for strategy, events in strategy_data.items():
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Strategy Analysis: {strategy.replace('_', ' ').title()}", fontsize=14)
            
            # Success rates over iterations
            iterations = [e.iteration for e in events]
            success_rates = [e.success_rate * 100 for e in events]
            ax1.plot(iterations, success_rates, 'o-', alpha=0.7)
            ax1.set_title("Success Rate Over Iterations")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Success Rate (%)")
            ax1.grid(True, alpha=0.3)
            
            # Confidence vs actual performance
            confidences = [e.confidence for e in events]
            actual_perf = [e.success_rate for e in events]
            ax2.scatter(confidences, actual_perf, alpha=0.7)
            ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax2.set_title("Confidence vs Actual Performance")
            ax2.set_xlabel("Predicted Confidence")
            ax2.set_ylabel("Actual Success Rate")
            ax2.grid(True, alpha=0.3)
            
            # Constraints satisfied distribution
            constraints_satisfied = [e.constraints_satisfied for e in events]
            ax3.hist(constraints_satisfied, bins=10, alpha=0.7, edgecolor='black')
            ax3.set_title("Distribution of Constraints Satisfied")
            ax3.set_xlabel("Constraints Satisfied")
            ax3.set_ylabel("Frequency")
            
            # Execution time over iterations
            exec_times = [e.execution_time for e in events]
            ax4.plot(iterations, exec_times, 'o-', alpha=0.7, color='orange')
            ax4.set_title("Execution Time Over Iterations")
            ax4.set_xlabel("Iteration")
            ax4.set_ylabel("Execution Time (seconds)")
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the strategy-specific plot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.plot_save_dir}/strategy_{strategy}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()  # Close to save memory
            
            print(f"{Fore.GREEN}Strategy analysis plot saved: {filename}{Style.RESET_ALL}")

# Demo function
def run_donor_evolution_demo():
    """Run the donor evolution visualization demo."""
    
    # Define our target function
    def find_max_in_range(numbers, start_idx, end_idx):
        """Find the maximum value in a list within a specific range."""
        if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
            return None
        
        max_val = numbers[start_idx]
        for i in range(start_idx + 1, end_idx):
            if numbers[i] > max_val:
                max_val = numbers[i]
        
        return max_val
    
    # Create and run the visualizer
    visualizer = DonorGenerationVisualizer(find_max_in_range)
    
    print(f"{Fore.CYAN}DONOR EVOLUTION VISUALIZATION DEMO{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}This demo shows how the system iteratively generates{Style.RESET_ALL}")
    print(f"{Fore.CYAN}and tests donor candidates against the original function's{Style.RESET_ALL}")
    print(f"{Fore.CYAN}constraints until it finds successful solutions.{Style.RESET_ALL}")
    print(f"\\n{Fore.YELLOW}Press Ctrl+C to stop early if desired.{Style.RESET_ALL}")
    
    try:
        # Run the evolution process
        successful_candidates = visualizer.run_evolution_process(
            max_iterations=8,
            target_success_rate=0.8
        )
        
        # Show final results
        visualizer.show_final_summary(successful_candidates)

        # Save comprehensive plots
        final_plot_file = visualizer.save_final_plot()
        visualizer.save_strategy_analysis_plots()
        
        # Save data for analysis
        visualizer.save_evolution_data_with_code("general_evolution_data.json")
        
        # Keep the plot open
        print(f"\\n{Fore.CYAN}Visualization complete - close the plot window to exit{Style.RESET_ALL}")
        plt.show()
        
        return successful_candidates
        
    except KeyboardInterrupt:
        print(f"\\n\\n{Fore.RED}Evolution stopped by user{Style.RESET_ALL}")
        visualizer.show_final_summary([])
        return []

def run_comparative_evolution():
    """Run evolution on multiple target functions for comparison."""
    
    # Define multiple target functions
    def find_max_in_range(numbers, start_idx, end_idx):
        if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
            return None
        max_val = numbers[start_idx]
        for i in range(start_idx + 1, end_idx):
            if numbers[i] > max_val:
                max_val = numbers[i]
        return max_val
    
    def find_min_in_range(numbers, start_idx, end_idx):
        if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
            return None
        min_val = numbers[start_idx]
        for i in range(start_idx + 1, end_idx):
            if numbers[i] < min_val:
                min_val = numbers[i]
        return min_val
    
    def sum_in_range(numbers, start_idx, end_idx):
        if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
            return 0
        return sum(numbers[start_idx:end_idx])
    
    functions = [find_max_in_range, find_min_in_range, sum_in_range]
    results = {}
    
    for func in functions:
        print(f"\\n{'='*60}")
        print(f"{Fore.CYAN}EVOLVING DONORS FOR: {func.__name__}{Style.RESET_ALL}")
        print('='*60)
        
        visualizer = DonorGenerationVisualizer(func)
        successful_candidates = visualizer.run_evolution_process(
            max_iterations=5,
            target_success_rate=0.7
        )

        # Show final results
        visualizer.show_final_summary(successful_candidates)

        # Save comprehensive plots
        final_plot_file = visualizer.save_final_plot()
        visualizer.save_strategy_analysis_plots()
        
        # Save data for analysis
        visualizer.save_evolution_data_with_code(f"{func.__name__}_evolution_data.json")
        
        results[func.__name__] = {
            'successful_count': len(successful_candidates),
            'total_events': len(visualizer.events),
            'best_success_rate': max([e.success_rate for e in visualizer.events], default=0)
        }
    
    # Comparative summary
    print(f"\\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}COMPARATIVE EVOLUTION RESULTS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    for func_name, data in results.items():
        print(f"{Fore.CYAN}{func_name}:{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}Successful candidates: {data['successful_count']}{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}Total candidates tested: {data['total_events']}{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}Best success rate: {data['best_success_rate']:.1%}{Style.RESET_ALL}")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        run_comparative_evolution()
    else:
        run_donor_evolution_demo()