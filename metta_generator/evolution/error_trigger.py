#!/usr/bin/env python3
"""
Error-triggered evolution system
"""

import inspect
import textwrap
from typing import Callable, List, Dict, Any
from metta_generator.evolution.semantic_evolution import SemanticEvolutionEngine
from metta_generator.evolution.error_fitness import ErrorDrivenFitnessEvaluator

class ErrorTriggeredEvolution:
    """Manages error-triggered evolution with unit test fitness"""
    
    def __init__(self, metta_space=None, reasoning_engine=None):
        self.evolution_engine = SemanticEvolutionEngine(
            metta_space=metta_space,
            reasoning_engine=reasoning_engine,
            population_size=10,  # Smaller for error response
            max_generations=5    # Faster convergence
        )
        self.fitness_evaluator = ErrorDrivenFitnessEvaluator()
        self.unit_tests = {}  # Store unit tests per function
        self.original_functions = {}  # Store original function code
    
    def _get_clean_function_source(self, func: Callable) -> str:
        """Get dedented source code for a function"""
        try:
            raw_source = inspect.getsource(func)
            return textwrap.dedent(raw_source)
        except:
            return f"# Source not available for {func.__name__ if hasattr(func, '__name__') else 'unknown'}"
    
    def register_function_tests(self, func_name: str, tests: List[Callable]):
        """Register unit tests for a function with dedented source"""
        self.unit_tests[func_name] = tests
        
        # Store dedented test sources for potential use in evolution
        test_sources = {}
        for test in tests:
            test_sources[test.__name__] = self._get_clean_function_source(test)
        
        # Store for potential debugging/analysis
        self.test_sources = getattr(self, 'test_sources', {})
        self.test_sources[func_name] = test_sources
    
    def register_original_function(self, func: Callable):
        """Register original function for evolution"""
        func_name = func.__name__
        raw_source = inspect.getsource(func)
        self.original_functions[func_name] = textwrap.dedent(raw_source)
    
    def handle_error(self, func_name: str, error_info: Dict[str, Any]):
        """Handle error by triggering evolution"""
        print(f"Error in {func_name}: {error_info['error_type']}")
        print(f"   Triggering evolution to fix the error...")
        
        # Get original function code
        if func_name not in self.original_functions:
            print(f"   No original function registered for {func_name}")
            return
        
        unit_tests = self.unit_tests.get(func_name, [])
        
        if not unit_tests:
            print(f"   No unit tests registered for {func_name}")
            return
        
        # Create custom fitness function that uses unit tests
        def error_fitness_function(genome, generated_code):
            return self.fitness_evaluator.evaluate_evolved_code(
                code=generated_code,
                unit_tests=unit_tests,
                function_name=func_name,
                error_context=error_info
            )
        
        # Replace the fitness evaluator in evolution engine
        original_evaluator = self.evolution_engine.fitness_evaluator
        self.evolution_engine.fitness_evaluator.evaluate_genome = self._create_genome_evaluator(
            error_fitness_function
        )
        
        try:
            # Determine semantics from error context
            target_semantics = self._infer_semantics_from_error(func_name, error_info)
            
            # Run evolution
            results = self.evolution_engine.evolve_solutions(target_semantics)
            
            if results:
                best_solution = results[0]
                fitness = best_solution.get('final_score', 0.0)
                print(f"   Evolution complete! Best fitness: {fitness:.3f}")
                
                if fitness > 0.8:  # High fitness threshold
                    print(f"   High-quality solution found:")
                    print(f"   {best_solution['name']}")
                    # Could auto-apply the fix here
                
            else:
                print(f"   Evolution failed to find solutions")
                
        finally:
            # Restore original fitness evaluator
            self.evolution_engine.fitness_evaluator = original_evaluator
    
    def _create_genome_evaluator(self, fitness_func):
        """Create genome evaluator that uses custom fitness function"""
        def evaluate_genome(genome, generated_code):
            # Get base evaluation
            base_results = {
                "correctness_score": 0.0,
                "efficiency_score": 0.5,  # Neutral
                "maintainability_score": 0.5,  # Neutral
                "semantic_consistency_score": 0.5,  # Neutral
                "metta_reasoning_score": 0.5   # Neutral
            }
            
            # Use our custom fitness function for correctness
            correctness = fitness_func(genome, generated_code)
            base_results["correctness_score"] = correctness
            
            # Calculate overall fitness
            base_results["overall_fitness"] = (
                0.8 * correctness +  # Heavy weight on correctness (test passing)
                0.1 * base_results["efficiency_score"] +
                0.1 * base_results["maintainability_score"]
            )
            
            return base_results
        
        return evaluate_genome
    
    def _infer_semantics_from_error(self, func_name: str, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Infer target semantics from error context"""
        error_type = error_info.get('error_type', '')
        
        # Default semantics
        semantics = {
            "purpose": "fix_error", 
            "input_constraints": ["input_received", "error_handled"],
            "output_spec": "error_free_result"
        }
        
        # Adjust based on error type
        if error_type == "IndexError":
            semantics["input_constraints"].append("bounds_checked")
            semantics["purpose"] = "bounds_safe_operation"
        elif error_type == "TypeError":
            semantics["input_constraints"].append("type_validated")
            semantics["purpose"] = "type_safe_operation"
        elif error_type == "ValueError":
            semantics["input_constraints"].append("value_validated")
            semantics["purpose"] = "value_safe_operation"
        
        return semantics