#!/usr/bin/env python3
"""
Error-driven fitness evaluator that uses unit test passing as fitness metric
"""

import sys
import textwrap
from typing import List, Dict, Any, Callable
from io import StringIO

class ErrorDrivenFitnessEvaluator:
    """Fitness evaluator based on unit test passing and error resolution"""
    
    def __init__(self):
        self.test_results = {}
        self.error_history = {}
    
    def evaluate_evolved_code(self, code: str, unit_tests: List[Callable], 
                            function_name: str, error_context: Dict[str, Any] = None) -> float:
        """
        Evaluate fitness based on unit test passing rate
        
        Args:
            code: Generated/evolved code
            unit_tests: List of unit test functions
            function_name: Name of the function being evolved
            error_context: Context from the original error
            
        Returns:
            Fitness score (0.0 to 1.0)
        """
        if not unit_tests:
            return 0.5  # Neutral fitness if no tests
        
        # Execute the code and extract the evolved function
        try:
            exec_globals = {}
            clean_code = textwrap.dedent(code)
            exec(clean_code, exec_globals)
            
            # Find the evolved function
            evolved_func = None
            for name, obj in exec_globals.items():
                if callable(obj) and (name == function_name or name.startswith(('evolved_', 'semantic_'))):
                    evolved_func = obj
                    break
            
            if not evolved_func:
                return 0.0  # No valid function found
            
        except Exception as e:
            return 0.0  # Code doesn't compile/execute
        
        # Run unit tests
        passed = 0
        total = len(unit_tests)
        
        for test in unit_tests:
            try:
                # Capture stdout/stderr to prevent test output pollution
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = StringIO()
                sys.stderr = StringIO()
                
                # Run the test with the evolved function
                test(evolved_func)
                passed += 1
                
            except Exception as test_error:
                # Test failed
                pass
            finally:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        
        # Calculate base fitness from test passing rate
        base_fitness = passed / total
        
        # Add bonus for error resolution if error context is provided
        if error_context:
            error_resolution_bonus = self._calculate_error_resolution_bonus(
                evolved_func, error_context
            )
            base_fitness = min(1.0, base_fitness + error_resolution_bonus)
        
        return base_fitness
    
    def _calculate_error_resolution_bonus(self, evolved_func: Callable, 
                                        error_context: Dict[str, Any]) -> float:
        """Calculate bonus for resolving specific error patterns"""
        bonus = 0.0
        
        # Get error type from context
        error_type = error_context.get('error_type', '')
        
        # Test with the original failing inputs if available
        if 'failing_inputs' in error_context:
            try:
                for inputs in error_context['failing_inputs']:
                    if isinstance(inputs, list):
                        evolved_func(*inputs)
                    else:
                        evolved_func(inputs)
                # If we get here, the error was resolved
                bonus += 0.1
            except:
                # Still fails - no bonus
                pass
        
        # Bonus for handling specific error types
        if error_type in ['IndexError', 'TypeError', 'ValueError'] and bonus > 0:
            bonus += 0.05
        
        return bonus