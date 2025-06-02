#!/usr/bin/env python3
"""
Enhanced Donor Generation Process Visualizer
Updated to work with the ModularMettaDonorGenerator system used by the "generate" command
"""

import time
import json
import inspect
import os
import ast
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Any, Tuple, Dict, Optional, Union, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Import the proper MeTTa generator system
try:
    from metta_generator.base import ModularMettaDonorGenerator, GenerationStrategy
    from reflectors.dynamic_monitor import DynamicMonitor
    METTA_COMPONENTS_AVAILABLE = True
except ImportError:
    METTA_COMPONENTS_AVAILABLE = False
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
    generator_used: str = "unknown"

class ConstraintBasedTester:
    """Tests candidates against the original function's constraints."""
    
    def __init__(self, original_function):
        self.original_function = original_function
        self.test_cases = self._generate_test_cases()
        
    def _generate_test_cases(self):
        """Generate comprehensive test cases based on the original function."""
        test_cases = []
        
        # Get function signature to adapt test cases accordingly
        sig = inspect.signature(self.original_function)
        param_count = len(sig.parameters)
        param_names = list(sig.parameters.keys())
        
        print(f"    Generating test cases for {self.original_function.__name__} with {param_count} parameters: {param_names}")
        
        if param_count == 3 and all(param in str(sig) for param in ['start', 'end', 'idx']):
            # Range-based function (like find_max_in_range)
            test_cases = self._generate_range_based_test_cases()
        elif param_count == 1:
            # Single parameter functions
            test_cases = self._generate_single_param_test_cases()
        elif param_count == 2:
            # Two parameter functions
            test_cases = self._generate_two_param_test_cases()
        else:
            # Generic test cases
            test_cases = self._generate_generic_test_cases()
            
        print(f"    Generated {len(test_cases)} test cases")
        return test_cases
    
    def _generate_range_based_test_cases(self):
        """Generate test cases for range-based functions like find_max_in_range."""
        test_cases = []
        
        # Test data sets
        test_data = [
            [1, 5, 3, 9, 2, 7, 4],      # Normal case
            [10, 20, 30, 40, 50],       # Ascending
            [50, 40, 30, 20, 10],       # Descending  
            [5, 5, 5, 5, 5],            # All equal
            [42],                        # Single element
            [1, 2],                     # Two elements
            list(range(20))             # Larger dataset (reduced from 100 for faster testing)
        ]
        
        for data in test_data:
            # Valid range tests
            try:
                test_cases.extend([
                    ((data, 0, len(data)), self.original_function(data, 0, len(data))),
                    ((data, 1, len(data)-1 if len(data) > 1 else 1), 
                     self.original_function(data, 1, len(data)-1 if len(data) > 1 else 1)),
                ])
                
                if len(data) > 2:
                    test_cases.extend([
                        ((data, 0, len(data)//2), self.original_function(data, 0, len(data)//2)),
                        ((data, 0, 1), self.original_function(data, 0, 1)),
                        ((data, len(data)-2, len(data)), self.original_function(data, len(data)-2, len(data))),
                    ])
                
                # Boundary tests  
                test_cases.extend([
                    ((data, -1, 2), self.original_function(data, -1, 2)),
                    ((data, 0, len(data)+5), self.original_function(data, 0, len(data)+5)),
                    ((data, 5, 2), self.original_function(data, 5, 2)),
                ])
            except Exception as e:
                print(f"    Warning: Failed to generate test case for data {data[:3]}...: {e}")
        
        return test_cases
    
    def _generate_single_param_test_cases(self):
        """Generate test cases for single parameter functions."""
        test_cases = []
        
        # Different types of single parameter inputs
        test_inputs = [
            # Lists
            [1, 2, 3, 4, 5],
            [10, 5, 8, 3, 1],
            [],
            [42],
            
            # Strings  
            "hello world",
            "Testing String",
            "",
            "a",
            
            # Numbers
            42,
            0,
            -5,
            3.14,
            
            # Other data structures
            {"a": 1, "b": 2, "c": 3},
            {1, 2, 3, 4, 5},
            (1, 2, 3),
        ]
        
        for test_input in test_inputs:
            try:
                expected_output = self.original_function(test_input)
                test_cases.append(((test_input,), expected_output))
            except Exception as e:
                # Add error cases as well
                test_cases.append(((test_input,), f"ERROR: {str(e)}"))
        
        return test_cases
    
    def _generate_two_param_test_cases(self):
        """Generate test cases for two parameter functions."""
        test_cases = []
        
        # Common two-parameter scenarios
        test_pairs = [
            # String operations
            ("hello world", "world"),
            ("testing", "test"),
            ("", "anything"),
            
            # Number operations
            (10, 5),
            (0, 1),
            (100, 50),
            
            # List/collection operations
            ([1, 2, 3], 2),
            ([1, 2, 3], [4, 5, 6]),
            ([], []),
        ]
        
        for param1, param2 in test_pairs:
            try:
                expected_output = self.original_function(param1, param2)
                test_cases.append(((param1, param2), expected_output))
            except Exception as e:
                test_cases.append(((param1, param2), f"ERROR: {str(e)}"))
        
        return test_cases
    
    def _generate_generic_test_cases(self):
        """Generate generic test cases for functions with unknown signatures."""
        test_cases = []
        
        # Try some basic test cases with various argument counts
        basic_tests = [
            ((),),
            ((1,),),
            ((1, 2),),
            ((1, 2, 3),),
            (([1, 2, 3],),),
            (("test",),),
            (({"key": "value"},),),
        ]
        
        for test_args in basic_tests:
            try:
                expected_output = self.original_function(*test_args)
                test_cases.append((test_args, expected_output))
            except Exception:
                # Skip test cases that don't match the function signature
                continue
        
        return test_cases
    
    def test_candidate(self, candidate_func) -> List[CandidateTest]:
        """Test a candidate function against all constraints."""
        results = []
        
        for test_input, expected_output in self.test_cases:
            try:
                # Handle error cases from test generation
                if isinstance(expected_output, str) and expected_output.startswith("ERROR:"):
                    # Expected an error, so if candidate doesn't error, it's a failure
                    try:
                        actual_output = candidate_func(*test_input)
                        result = TestResult.FAILED
                        error_msg = f"Expected error but got {actual_output}"
                    except Exception:
                        result = TestResult.PASSED
                        actual_output = "ERROR (as expected)"
                        error_msg = ""
                else:
                    # Normal test case
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

class EnhancedDonorGenerationVisualizer:
    """Enhanced visualizer using the ModularMettaDonorGenerator system."""
    
    def __init__(self, original_function, ontology_file: Optional[str] = None):
        self.original_function = original_function
        self.original_function_code = self._get_function_source(original_function)
        self.tester = ConstraintBasedTester(original_function)
        self.events: List[GenerationEvent] = []
        self.generation_start_time = time.time()
        self.successful_candidate_codes = []
        
        # Initialize MeTTa components
        if METTA_COMPONENTS_AVAILABLE:
            self._initialize_metta_components(ontology_file)
        else:
            self.metta_generator = None
            print(f"{Fore.YELLOW}Warning: MeTTa components not available, using simulation mode{Style.RESET_ALL}")
        
        # Setup visualization
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle(f"Enhanced Donor Generation Evolution: {original_function.__name__}", fontsize=16)
        self.save_plots = True
        self.plot_save_dir = "evolution_plots"
        self._ensure_plot_directory()
        
        # Initialize plots
        self._setup_plots()
        
    def _initialize_metta_components(self, ontology_file: Optional[str]):
        """Initialize the MeTTa generator components."""
        try:
            print(f"  Initializing Enhanced MeTTa Donor Generator...")
            
            # Create local monitor for this visualization
            self.local_monitor = DynamicMonitor()
            
            # Create the modular generator
            self.metta_generator = ModularMettaDonorGenerator(metta_space=self.local_monitor.metta_space)
            
            # Register the specialized generators
            from metta_generator.operation_substitution import OperationSubstitutionGenerator
            from metta_generator.data_struct_adaptation import DataStructureAdaptationGenerator  
            from metta_generator.algo_transformation import AlgorithmTransformationGenerator
            
            op_sub_generator = OperationSubstitutionGenerator()
            data_adapt_generator = DataStructureAdaptationGenerator()
            algo_transform_generator = AlgorithmTransformationGenerator()
            
            self.metta_generator.registry.register_generator(op_sub_generator)
            self.metta_generator.registry.register_generator(data_adapt_generator)
            self.metta_generator.registry.register_generator(algo_transform_generator)
            
            print(f"  ✓ Registered {len(self.metta_generator.registry.generators)} specialized generators")
            
            # Load ontology
            if ontology_file and os.path.exists(ontology_file):
                ontology_loaded = self.metta_generator.load_ontology(ontology_file)
                if ontology_loaded:
                    print(f"  ✓ Loaded ontology: {ontology_file}")
                else:
                    print(f"  ⚠ Failed to load ontology: {ontology_file}")
            else:
                print(f"  ⚠ No ontology file provided or file not found")
            
        except Exception as e:
            print(f"  ✗ Failed to initialize MeTTa components: {e}")
            self.metta_generator = None
    
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
        
        # Plot 4: Generator attribution
        self.axes[1, 1].set_title("Generator Attribution")
        self.axes[1, 1].set_xlabel("Generator Type")
        self.axes[1, 1].set_ylabel("Candidates Generated")
    
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
            return f"def {func.__name__}(...):\n    # Source code not available"
    
    def run_evolution_process(self, max_iterations=8, target_success_rate=0.8):
        """Run the iterative evolution process using the enhanced MeTTa system."""
        print(f"{Fore.CYAN}STARTING ENHANCED CONSTRAINT-BASED DONOR EVOLUTION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Original function: {self.original_function.__name__}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Test cases: {len(self.tester.test_cases)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Target success rate: {target_success_rate:.1%}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}MeTTa generator available: {self.metta_generator is not None}{Style.RESET_ALL}")
        
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
                
                print(f"  {candidate['name']}: {Fore.BLUE}{constraints_met}{Style.RESET_ALL}/{event.total_constraints} constraints ({Fore.BLUE}{success_rate:.1%}{Style.RESET_ALL}) [{event.generator_used}]")
                
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
                        'total_constraints': event.total_constraints,
                        'generator_used': event.generator_used
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
        
        print(f"\\n{Fore.CYAN}ENHANCED EVOLUTION COMPLETE{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Iterations: {iteration}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Total candidates tested: {len(self.events)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Successful candidates: {len(successful_candidates)}{Style.RESET_ALL}")
        
        return successful_candidates
    
    def _generate_candidates_for_iteration(self, iteration):
        """Generate candidates using the enhanced MeTTa system."""
        
        # Try real MeTTa generation first
        if self.metta_generator and iteration <= 3:
            try:
                print(f"  Using ModularMettaDonorGenerator...")
                real_candidates = self.metta_generator.generate_donors_from_function(self.original_function)
                
                if real_candidates:
                    # Add generator attribution
                    for candidate in real_candidates:
                        if 'generator_used' not in candidate:
                            candidate['generator_used'] = candidate.get('strategy', 'unknown')
                    
                    print(f"  ✓ Generated {len(real_candidates)} real MeTTa candidates")
                    
                    # Take a subset to avoid overwhelming the visualization
                    max_candidates = min(4, len(real_candidates))
                    return real_candidates[:max_candidates]
                
            except Exception as e:
                print(f"  {Fore.RED}MeTTa generation failed: {e}{Style.RESET_ALL}")
        
        # Generate synthetic candidates as fallback
        print(f"  Using synthetic generation...")
        return self._generate_synthetic_candidates(iteration)
    
    def _generate_synthetic_candidates(self, iteration):
        """Generate synthetic candidates for visualization purposes."""
        strategies = [
            "operation_substitution", "data_structure_adaptation", "algorithm_transformation",
            "accumulator_variation", "structure_preservation", "condition_variation"
        ]
        
        candidates = []
        num_candidates = min(3 + iteration, 5)
        
        for i in range(num_candidates):
            strategy = strategies[i % len(strategies)]
            
            candidate = {
                "name": f"{self.original_function.__name__}_{strategy}_iter{iteration}_{i}",
                "description": f"Iteration {iteration}: {strategy} approach",  
                "strategy": strategy,
                "confidence": 0.3 + (iteration * 0.1) + (i * 0.05),
                "final_score": 0.4 + (iteration * 0.08),
                "properties": [strategy, "synthetic"],
                "code": self._generate_synthetic_code(strategy, iteration, i),
                "metta_derivation": [f"(synthetic-generation {strategy} iter-{iteration})"],
                "generator_used": f"synthetic-{strategy}"
            }
            candidates.append(candidate)
        
        return candidates
    
    def _generate_synthetic_code(self, strategy, iteration, variant):
        """Generate synthetic code that attempts to solve the same problem."""
        func_name = f"{self.original_function.__name__}_{strategy}_iter{iteration}_{variant}"
        
        # Get original function parameters
        try:
            sig = inspect.signature(self.original_function)
            params = list(sig.parameters.keys())
        except:
            params = ["data", "param1", "param2"]
        
        param_str = ", ".join(params)
        
        # Generate different approaches based on strategy
        if strategy == "operation_substitution":
            return self._generate_op_sub_synthetic(func_name, param_str, params, iteration)
        elif strategy == "data_structure_adaptation":
            return self._generate_ds_adapt_synthetic(func_name, param_str, params, iteration)
        elif strategy == "algorithm_transformation":
            return self._generate_algo_transform_synthetic(func_name, param_str, params, iteration)
        else:
            return self._generate_generic_synthetic(func_name, param_str, params, strategy)
    
    def _generate_op_sub_synthetic(self, func_name, param_str, params, iteration):
        """Generate operation substitution synthetic code."""
        if len(params) >= 3:  # Range-based function
            return f'''def {func_name}({param_str}):
    """Operation substitution variant - using min instead of max."""
    if {params[1]} < 0 or {params[2]} > len({params[0]}) or {params[1]} >= {params[2]}:
        return None
    
    result = {params[0]}[{params[1]}]
    for i in range({params[1]} + 1, {params[2]}):
        if {params[0]}[i] < result:  # Changed from > to <
            result = {params[0]}[i]
    
    return result'''
        else:
            return f'''def {func_name}({param_str}):
    """Operation substitution variant."""
    # Synthetic operation substitution logic
    return "substituted_" + str({params[0]})'''
    
    def _generate_ds_adapt_synthetic(self, func_name, param_str, params, iteration):
        """Generate data structure adaptation synthetic code."""
        return f'''def {func_name}({param_str}):
    """Data structure adaptation variant - using set instead of list."""
    if len({params[0] if params else 'data'}) == 0:
        return None
    
    # Convert to set for unique processing
    data_set = set({params[0] if params else 'data'})
    
    # Process as set
    return max(data_set) if data_set else None'''
    
    def _generate_algo_transform_synthetic(self, func_name, param_str, params, iteration):
        """Generate algorithm transformation synthetic code."""
        return f'''def {func_name}({param_str}):
    """Algorithm transformation variant - recursive approach."""
    def recursive_helper(data, start, end):
        if start >= end or start >= len(data):
            return None
        
        if start == end - 1:
            return data[start]
        
        mid = (start + end) // 2
        left_result = recursive_helper(data, start, mid)
        right_result = recursive_helper(data, mid, end)
        
        if left_result is None:
            return right_result
        if right_result is None:
            return left_result
        
        return max(left_result, right_result)
    
    if len(params) >= 3:
        return recursive_helper({params[0]}, {params[1]}, {params[2]})
    else:
        return recursive_helper({params[0] if params else '[]'}, 0, len({params[0] if params else '[]'}))'''
    
    def _generate_generic_synthetic(self, func_name, param_str, params, strategy):
        """Generate generic synthetic code."""
        return f'''def {func_name}({param_str}):
    """Generic {strategy} variant."""
    # Synthetic {strategy} implementation
    try:
        if not {params[0] if params else 'data'}:
            return None
        return {params[0] if params else 'data'}[0]  # Simple fallback
    except (IndexError, TypeError):
        return None'''
    
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
                execution_time=time.time() - iteration_start,
                generator_used=candidate.get('generator_used', 'unknown')
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
            execution_time=time.time() - iteration_start,
            generator_used=candidate.get('generator_used', 'unknown')
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
            'data_structure_adaptation': 'blue', 
            'algorithm_transformation': 'green',
            'accumulator_variation': 'orange',
            'structure_preservation': 'purple',
            'condition_variation': 'brown'
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
        
        strategies_list = list(strategy_stats.keys())
        avg_success_rates = [np.mean(strategy_stats[s]) * 100 for s in strategies_list]
        
        bars = self.axes[0, 1].bar(range(len(strategies_list)), avg_success_rates, 
                                  color=[strategy_colors.get(s, 'gray') for s in strategies_list])
        self.axes[0, 1].set_xticks(range(len(strategies_list)))
        self.axes[0, 1].set_xticklabels([s.replace('_', '\n') for s in strategies_list], rotation=45, ha='right')
        
        # Plot 3: Confidence vs Performance
        confidences = [e.confidence for e in self.events]
        actual_performance = [e.success_rate for e in self.events]
        
        self.axes[1, 0].scatter(confidences, actual_performance, 
                               c=colors, alpha=0.7, s=50)
        
        # Add diagonal line (perfect prediction)
        min_val, max_val = 0, 1
        self.axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Plot 4: Generator attribution
        generator_stats = {}
        for event in self.events:
            generator = event.generator_used
            if generator not in generator_stats:
                generator_stats[generator] = 0
            generator_stats[generator] += 1
        
        if generator_stats:
            generators = list(generator_stats.keys())
            counts = list(generator_stats.values())
            
            bars = self.axes[1, 1].bar(range(len(generators)), counts)
            self.axes[1, 1].set_xticks(range(len(generators)))
            self.axes[1, 1].set_xticklabels([g.replace('_', '\n') for g in generators], rotation=45, ha='right')

        plt.tight_layout()

        if self.save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            iteration_num = max(iterations) if iterations else 0
            filename = f"{self.plot_save_dir}/enhanced_evolution_iter_{iteration_num:02d}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"    {Fore.GREEN}Enhanced plot saved: {filename}{Style.RESET_ALL}")

        plt.draw()
        
    def show_final_summary(self, successful_candidates):
        """Show final summary with original function code and successful candidate codes."""
        print("\\n" + Fore.CYAN + "="*80 + Style.RESET_ALL)
        print(Fore.CYAN + "ENHANCED EVOLUTION SUMMARY WITH CODE" + Style.RESET_ALL)
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
            print(f"{Fore.GREEN}Generator: {candidate_info.get('generator_used', 'unknown')}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Success Rate: {candidate_info['success_rate']:.1%}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Constraints Satisfied: {candidate_info['constraints_satisfied']}/{candidate_info['total_constraints']}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Generated in Iteration: {candidate_info['iteration']}{Style.RESET_ALL}")
            print("\\n" + Fore.GREEN + "CODE:" + Style.RESET_ALL)
            print(Fore.GREEN + "-"*50 + Style.RESET_ALL)
            print(candidate_info['code'])
            print(Fore.GREEN + "-"*50 + Style.RESET_ALL)
        
        # Show evolution statistics
        if self.events:
            initial_success = self.events[0].success_rate
            final_success = max(e.success_rate for e in self.events)
            
            print(f"\\n" + Fore.BLUE + "="*80 + Style.RESET_ALL)
            print(Fore.BLUE + "ENHANCED EVOLUTION STATISTICS" + Style.RESET_ALL)
            print(Fore.BLUE + "="*80 + Style.RESET_ALL)
            print(f"{Fore.BLUE}Initial success rate: {initial_success:.1%}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Final success rate: {final_success:.1%}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Improvement: {final_success - initial_success:.1%}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Total candidates tested: {len(self.events)}{Style.RESET_ALL}")
            
            # Strategy performance
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
            
            # Generator performance
            generator_performance = {}
            for event in self.events:
                if event.generator_used not in generator_performance:
                    generator_performance[event.generator_used] = []
                generator_performance[event.generator_used].append(event.success_rate)
            
            print(f"\\n{Fore.BLUE}Generator Performance:{Style.RESET_ALL}")
            for generator, rates in generator_performance.items():
                avg_rate = np.mean(rates)
                best_rate = max(rates)
                count = len(rates)
                print(f"  {Fore.BLUE}{generator}: avg={avg_rate:.1%}, best={best_rate:.1%} ({count} candidates){Style.RESET_ALL}")
        
        # Show constraint analysis
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

    def save_evolution_data_with_code(self, filename="enhanced_evolution_data_with_code.json"):
        """Save enhanced evolution data including successful candidate codes."""
        data = {
            "original_function": {
                "name": self.original_function.__name__,
                "code": self.original_function_code
            },
            "total_events": len(self.events),
            "successful_candidates": self.successful_candidate_codes,
            "metta_generator_used": self.metta_generator is not None,
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
                "execution_time": event.execution_time,
                "generator_used": event.generator_used
            }
            data["events"].append(event_data)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\\n{Fore.GREEN}Enhanced evolution data with code saved to {filename}{Style.RESET_ALL}")

    def save_final_plot(self, filename=None):
        """Save a final comprehensive plot with all evolution data."""
        if not self.events:
            print("No data to plot")
            return
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.plot_save_dir}/final_enhanced_evolution_{self.original_function.__name__}_{timestamp}.png"
        
        # Update plots one final time
        self._update_plots()
        
        # Add title with summary statistics
        if self.successful_candidate_codes:
            success_count = len(self.successful_candidate_codes)
            best_rate = max(c['success_rate'] for c in self.successful_candidate_codes)
            metta_status = "with MeTTa" if self.metta_generator else "simulation"
            self.fig.suptitle(f"Enhanced Donor Evolution ({metta_status}): {self.original_function.__name__} "
                            f"({success_count} successful, best: {best_rate:.1%})", 
                            fontsize=16)
        
        # Save high-quality version
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"{Fore.GREEN}Final enhanced plot saved: {filename}{Style.RESET_ALL}")
        
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
            fig.suptitle(f"Enhanced Strategy Analysis: {strategy.replace('_', ' ').title()}", fontsize=14)
            
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
            
            # Generator attribution for this strategy
            generator_counts = {}
            for event in events:
                gen = event.generator_used
                generator_counts[gen] = generator_counts.get(gen, 0) + 1
            
            if generator_counts:
                generators = list(generator_counts.keys())
                counts = list(generator_counts.values())
                bars = ax4.bar(range(len(generators)), counts, alpha=0.7)
                ax4.set_title("Generator Attribution")
                ax4.set_xlabel("Generator")
                ax4.set_ylabel("Count")
                ax4.set_xticks(range(len(generators)))
                ax4.set_xticklabels([g.replace('_', '\n') for g in generators], rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save the strategy-specific plot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.plot_save_dir}/enhanced_strategy_{strategy}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()  # Close to save memory
            
            print(f"{Fore.GREEN}Enhanced strategy analysis plot saved: {filename}{Style.RESET_ALL}")


# Enhanced demo function
def run_enhanced_donor_evolution_demo(target_function=None, ontology_file=None):
    """Run the enhanced donor evolution visualization demo."""
    
    # Use provided function or default
    if target_function is None:
        def find_max_in_range(numbers, start_idx, end_idx):
            """Find the maximum value in a list within a specific range."""
            if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
                return None
            
            max_val = numbers[start_idx]
            for i in range(start_idx + 1, end_idx):
                if numbers[i] > max_val:
                    max_val = numbers[i]
            
            return max_val
        
        target_function = find_max_in_range
    
    # Create and run the enhanced visualizer
    visualizer = EnhancedDonorGenerationVisualizer(target_function, ontology_file)
    
    print(f"{Fore.CYAN}ENHANCED DONOR EVOLUTION VISUALIZATION DEMO{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}This enhanced demo shows how the ModularMettaDonorGenerator{Style.RESET_ALL}")
    print(f"{Fore.CYAN}system iteratively generates and tests donor candidates{Style.RESET_ALL}")
    print(f"{Fore.CYAN}against the original function's constraints.{Style.RESET_ALL}")
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
        data_filename = f"enhanced_{target_function.__name__}_evolution_data.json"
        visualizer.save_evolution_data_with_code(data_filename)
        
        # Keep the plot open
        print(f"\\n{Fore.CYAN}Enhanced visualization complete - close the plot window to exit{Style.RESET_ALL}")
        plt.show()
        
        return successful_candidates
        
    except KeyboardInterrupt:
        print(f"\\n\\n{Fore.RED}Enhanced evolution stopped by user{Style.RESET_ALL}")
        visualizer.show_final_summary([])
        return []


# Updated CLI integration function
def visualize_function_with_enhanced_metta(function_to_visualize, ontology_file=None):
    """
    Function that can be called from the CLI to visualize a specific function
    using the enhanced MeTTa donor generation system.
    """
    print(f"{Fore.CYAN}Starting Enhanced MeTTa Visualization for: {function_to_visualize.__name__}{Style.RESET_ALL}")
    
    # Customize plot save directory for the specific function
    visualizer = EnhancedDonorGenerationVisualizer(function_to_visualize, ontology_file)
    
    func_plot_dir = os.path.join("evolution_plots", function_to_visualize.__name__ + "_enhanced")
    visualizer.plot_save_dir = func_plot_dir
    visualizer._ensure_plot_directory()
    
    print(f"Enhanced plots and data will be saved in: {os.path.abspath(func_plot_dir)}")
    
    # Run the evolution
    successful_candidates = visualizer.run_evolution_process(
        max_iterations=8, 
        target_success_rate=0.8 
    )
    
    visualizer.show_final_summary(successful_candidates)
    
    final_plot_file = visualizer.save_final_plot()
    if final_plot_file:
         print(f"Final enhanced plot saved: {os.path.abspath(final_plot_file)}")
    
    visualizer.save_strategy_analysis_plots()
    
    data_filename = f"{function_to_visualize.__name__}_enhanced_evolution_data.json"
    full_data_path = os.path.join(func_plot_dir, data_filename)
    visualizer.save_evolution_data_with_code(full_data_path)
    print(f"Enhanced evolution data saved to: {os.path.abspath(full_data_path)}")
    
    return successful_candidates


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test-function":
        # Test with a specific function from test_file.py
        try:
            from test_file import find_max_in_range
            run_enhanced_donor_evolution_demo(find_max_in_range)
        except ImportError:
            print("Could not import from test_file.py, using default function")
            run_enhanced_donor_evolution_demo()
    else:
        run_enhanced_donor_evolution_demo()