"""
Autonomous Error-Driven Evolution System

This system automatically generates and applies fixes for runtime errors without requiring
pre-defined unit tests. It uses error context and MeTTa reasoning to create improved
function implementations.
"""

import functools
import inspect
from typing import Any, Dict, List, Callable, Optional

# Import existing components
from reflectors.dynamic_monitor import DynamicMonitor
from metta_generator.base import MeTTaPoweredModularDonorGenerator
from hyperon import *


class AutonomousErrorFixer:
    """
    Autonomous system that generates fixes for runtime errors using error context alone.
    No pre-defined unit tests required - generates test cases from error context.
    """
    
    def __init__(self, metta_space=None):
        """Initialize the autonomous error fixer."""
        self.metta = MeTTa()
        self.metta_space = metta_space or self.metta.space()
        
        # Initialize MeTTa-powered donor generator
        self.donor_generator = MeTTaPoweredModularDonorGenerator(metta_space=self.metta_space)
        
        # Track original functions and their current implementations
        self.function_registry = {}  # func_name -> original_function
        self.current_implementations = {}  # func_name -> current_active_function
        self.error_history = {}  # func_name -> list of error contexts
        self.fix_attempts = {}  # func_name -> number of fix attempts
        
        # Configuration
        self.max_fix_attempts = 3
        self.auto_apply_fixes = True
        
    def register_function(self, func: Callable, context: str = None):
        """Register a function for autonomous error fixing."""
        func_name = func.__name__
        self.function_registry[func_name] = func
        self.current_implementations[func_name] = func
        self.error_history[func_name] = []
        self.fix_attempts[func_name] = 0
        
        # Add function info to MeTTa space
        if context:
            context_atom = self.metta.parse_single(f"(function-context {func_name} {context})")
            self.metta_space.add_atom(context_atom)
        
        print(f"[OK] Registered function '{func_name}' for autonomous error fixing")
        
    def generate_test_cases_from_error(self, func_name: str, error_context: Dict) -> List[Callable]:
        """
        Generate test cases based on error context to verify fixes.
        This replaces the need for pre-defined unit tests.
        """
        test_cases = []
        
        # Extract error information
        error_type = error_context.get('error_type', 'Unknown')
        error_message = error_context.get('error_message', '')
        failing_inputs = error_context.get('failing_inputs', [])
        
        # Generate basic test case from the failing input
        if failing_inputs:
            failing_args = failing_inputs[0] if failing_inputs else ()
            
            def test_no_error(func):
                """Test that the function doesn't raise the same error."""
                try:
                    result = func(*failing_args)
                    return True  # Success if no exception
                except Exception as e:
                    if type(e).__name__ == error_type:
                        return False  # Same error type = test failed
                    return True  # Different error might be acceptable
            
            test_cases.append(test_no_error)
        
        # Generate boundary condition tests based on error type
        if error_type == 'IndexError':
            def test_index_safety(func):
                """Test with various index scenarios."""
                try:
                    # Test with empty inputs
                    if failing_inputs and len(failing_args) > 0:
                        if isinstance(failing_args[0], (list, tuple, str)):
                            result = func([], *failing_args[1:])
                        return True
                except:
                    return False
            test_cases.append(test_index_safety)
            
        elif error_type == 'ValueError':
            def test_value_handling(func):
                """Test with edge case values."""
                try:
                    # Test with None if applicable
                    if failing_inputs and len(failing_args) > 0:
                        modified_args = list(failing_args)
                        modified_args[0] = None
                        result = func(*modified_args)
                    return True
                except:
                    return False
            test_cases.append(test_value_handling)
            
        elif error_type == 'TypeError':
            def test_type_safety(func):
                """Test type handling."""
                try:
                    # Test with string input if original was numeric
                    if failing_inputs and len(failing_args) > 0:
                        modified_args = list(failing_args)
                        if isinstance(modified_args[0], (int, float)):
                            modified_args[0] = str(modified_args[0])
                        result = func(*modified_args)
                    return True
                except:
                    return False
            test_cases.append(test_type_safety)
        
        # Add a general robustness test
        def test_robustness(func):
            """General robustness test."""
            try:
                # Test the original failing case but expect it to handle gracefully
                if failing_inputs:
                    result = func(*failing_args)
                    # If it returns something, that's good
                    return result is not None or result == 0 or result == False
                return True
            except:
                return False
        
        test_cases.append(test_robustness)
        
        print(f"Generated {len(test_cases)} test cases from error context for '{func_name}'")
        return test_cases
    
    def evaluate_candidate_fitness(self, candidate_func: Callable, test_cases: List[Callable]) -> float:
        """Evaluate how well a candidate function performs on generated test cases."""
        if not test_cases:
            return 0.0
            
        passed_tests = 0
        total_tests = len(test_cases)
        
        for test_case in test_cases:
            try:
                if test_case(candidate_func):
                    passed_tests += 1
            except Exception as e:
                # Test case itself failed, count as failure
                continue
        
        fitness = passed_tests / total_tests
        print(f"  Candidate fitness: {passed_tests}/{total_tests} tests passed ({fitness:.1%})")
        return fitness
    
    def generate_fix_candidates(self, func_name: str, error_context: Dict) -> List[Dict]:
        """Generate fix candidates using MeTTa-powered donor generation."""
        original_func = self.function_registry.get(func_name)
        if not original_func:
            print(f"[ERROR] No original function found for '{func_name}'")
            return []
        
        try:
            # Get function source
            func_source = inspect.getsource(original_func)
            
            print(f"[INFO] Generating fix candidates for '{func_name}' using MeTTa reasoning...")
            print(f"   Error type: {error_context.get('error_type', 'Unknown')}")
            print(f"   Error message: {error_context.get('error_message', 'No message')}")
            
            # Add error context to MeTTa space for reasoning
            error_type = error_context.get('error_type', 'Unknown')
            error_msg = error_context.get('error_message', '').replace('"', '\\"')
            
            error_atom = self.metta.parse_single(f'(function-error {func_name} {error_type} "{error_msg}")')
            self.metta_space.add_atom(error_atom)
            
            # Generate candidates with error context
            candidates = self.donor_generator.generate_donors_from_function(func_source)
            
            # Filter candidates for error-fixing potential
            filtered_candidates = []
            for candidate in candidates:
                # Prefer candidates with error-handling related strategies
                strategy = candidate.get('strategy', '').lower()
                if any(keyword in strategy for keyword in ['safety', 'robust', 'validation', 'guard', 'check']):
                    candidate['error_fix_score'] = 1.0
                else:
                    candidate['error_fix_score'] = 0.5
                    
                filtered_candidates.append(candidate)
            
            # Sort by error-fixing potential
            filtered_candidates.sort(key=lambda x: x.get('error_fix_score', 0), reverse=True)
            
            print(f"   Generated {len(filtered_candidates)} fix candidates")
            return filtered_candidates[:5]  # Return top 5 candidates
            
        except Exception as e:
            print(f"[ERROR] Error generating fix candidates: {e}")
            return []
    
    def apply_fix(self, func_name: str, candidate: Dict) -> bool:
        """Apply a fix candidate by replacing the current implementation."""
        try:
            # Execute the candidate code to get the function
            candidate_code = candidate.get('code', '')
            if not candidate_code:
                return False
            
            # Create a new namespace for execution
            exec_globals = {}
            exec_locals = {}
            
            # Execute the candidate code
            exec(candidate_code, exec_globals, exec_locals)
            
            # Find the function in the executed code
            new_func = None
            for name, obj in exec_locals.items():
                if callable(obj) and name == func_name:
                    new_func = obj
                    break
            
            if not new_func:
                # Try to find any callable that might be our function
                callables = [obj for obj in exec_locals.values() if callable(obj)]
                if callables:
                    new_func = callables[0]
            
            if new_func:
                # Update the current implementation
                self.current_implementations[func_name] = new_func
                print(f"[OK] Applied fix for '{func_name}': {candidate.get('name', 'Unknown candidate')}")
                return True
            else:
                print(f"[ERROR] Could not extract function from candidate code")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error applying fix for '{func_name}': {e}")
            return False
    
    def handle_error(self, func_name: str, error_context: Dict) -> bool:
        """
        Main error handling method that attempts to fix errors autonomously.
        Returns True if a fix was successfully applied.
        """
        print(f"\n[INFO] Autonomous error handling triggered for '{func_name}'")
        print(f"   Error: {error_context.get('error_type', 'Unknown')} - {error_context.get('error_message', 'No message')}")
        
        # Step 1: Record the error
        self.error_history[func_name].append(error_context)
        
        # Check if we should attempt a fix
        if self.fix_attempts[func_name] >= self.max_fix_attempts:
            print(f"Max fix attempts reached for '{func_name}'. No more fixes will be attempted.")
            return False
            
        self.fix_attempts[func_name] += 1
        print(f"[INFO] Autonomous error handling triggered for '{func_name}' (Attempt {self.fix_attempts[func_name]})")
        
        # Step 2: Generate test cases from the error
        test_cases = self.generate_test_cases_from_error(func_name, error_context)
        
        # Step 3: Generate potential fix candidates
        candidates = self.generate_fix_candidates(func_name, error_context)
        
        if not candidates:
            print(f"No fix candidates generated for '{func_name}'. Cannot apply fix.")
            return False
            
        # Step 4: Evaluate candidates and find the best one
        best_candidate = None
        best_fitness = -1.0
        
        for candidate in candidates:
            try:
                # Create a function from the candidate code
                exec_globals = {}
                exec_locals = {}
                exec(candidate['code'], exec_globals, exec_locals)
                
                candidate_func = None
                for name, obj in exec_locals.items():
                    if callable(obj):
                        candidate_func = obj
                        break
                
                if candidate_func:
                    fitness = self.evaluate_candidate_fitness(candidate_func, test_cases)
                    candidate['fitness'] = fitness
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_candidate = candidate
            except Exception as e:
                print(f"  Error evaluating candidate: {e}")
        
        # Step 5: Apply the best fix if it meets criteria
        if best_candidate and best_fitness > 0.5: # Fitness threshold
            print(f"[INFO] Best candidate found: {best_candidate.get('name')} with fitness {best_fitness:.1%}")
            
            if self.auto_apply_fixes:
                if self.apply_fix(func_name, best_candidate):
                    print(f"Autonomous fix for '{func_name}' was successful.")
                    return True
                else:
                    print(f"Failed to apply the best fix for '{func_name}'.")
            else:
                print(f"Fix available for '{func_name}', but auto-apply is disabled.")
                # Could add logic here to ask for user confirmation
        else:
            print(f"No suitable fix found for '{func_name}' (best fitness: {best_fitness:.1%})")
            
        return False
    
    def get_current_implementation(self, func_name: str) -> Callable:
        """Get the current (possibly fixed) implementation of a function."""
        return self.current_implementations.get(func_name, self.function_registry.get(func_name))


class AutonomousMonitor(DynamicMonitor):
    """
    Enhanced monitor that integrates with autonomous error fixing.
    Automatically attempts to fix errors when they occur.
    """
    
    def __init__(self, metta_space=None):
        super().__init__(metta_space)
        self.error_fixer = AutonomousErrorFixer(metta_space)
        self.function_call_stack = []  # Track nested function calls
        
    def autonomous_transform(self, context: Optional[str] = None, 
                           enable_auto_fix: bool = True,
                           max_fix_attempts: int = 3):
        """
        Decorator for autonomous error fixing without pre-defined unit tests.
        
        Args:
            context: Domain context for the function
            enable_auto_fix: Whether to automatically apply fixes
            max_fix_attempts: Maximum number of fix attempts per function
            
        Returns:
            Decorated function with autonomous error fixing
        """
        def decorator(func):
            # Register function for autonomous fixing
            self.error_fixer.register_function(func, context)
            self.error_fixer.auto_apply_fixes = enable_auto_fix
            self.error_fixer.max_fix_attempts = max_fix_attempts
            
            # Apply original monitoring
            monitored_func = self.hybrid_transform(context, auto_fix=False)(func)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                func_name = func.__name__
                
                # Track call stack to prevent infinite recursion
                if func_name in self.function_call_stack:
                    # Recursive call detected, use original implementation
                    return func(*args, **kwargs)
                
                self.function_call_stack.append(func_name)
                
                try:
                    # Try current implementation first
                    current_impl = self.error_fixer.get_current_implementation(func_name)
                    
                    if current_impl != func:
                        # Using a fixed implementation
                        print(f"🔄 Using fixed implementation for '{func_name}'")
                        result = current_impl(*args, **kwargs)
                    else:
                        # Using original implementation with monitoring
                        result = monitored_func(*args, **kwargs)
                    
                    return result
                    
                except Exception as e:
                    # Create error context
                    error_context = self._create_error_context(func, e, args)
                    
                    # Attempt autonomous fix
                    fix_applied = self.error_fixer.handle_error(func_name, error_context)
                    
                    if fix_applied:
                        print(f"🔄 Retrying '{func_name}' with applied fix...")
                        try:
                            # Retry with the fixed implementation
                            fixed_impl = self.error_fixer.get_current_implementation(func_name)
                            result = fixed_impl(*args, **kwargs)
                            print(f"✅ Fixed implementation succeeded for '{func_name}'")
                            return result
                        except Exception as retry_error:
                            print(f"❌ Fixed implementation still failed: {retry_error}")
                            # Fall through to re-raise original error
                    
                    # No fix applied or fix failed, re-raise original error
                    raise
                    
                finally:
                    # Remove from call stack
                    if func_name in self.function_call_stack:
                        self.function_call_stack.remove(func_name)
            
            return wrapper
        
        return decorator

    def _create_error_context(self, func, error, args) -> Dict:
        # Implementation of _create_error_context method
        # This method should return a dictionary representing the error context
        # based on the function, error, and arguments
        # This is a placeholder and should be implemented based on your specific requirements
        return {}


# Factory function for easy usage
def create_autonomous_monitor(ontology_path: str = None) -> AutonomousMonitor:
    """Create an autonomous monitor with optional ontology loading."""
    monitor = AutonomousMonitor()
    
    if ontology_path:
        try:
            monitor.load_metta_rules(ontology_path)
            print(f"✓ Loaded ontology for autonomous fixing: {ontology_path}")
        except Exception as e:
            print(f"⚠ Could not load ontology: {e}")
    
    return monitor


# Demo functions for testing
def demo_autonomous_fixing():
    """Demonstrate the autonomous error fixing system."""
    
    print("🚀 Autonomous Error Fixing Demo")
    print("="*50)
    
    # Create autonomous monitor
    monitor = create_autonomous_monitor()
    
    # Define problematic functions
    @monitor.autonomous_transform(context="array_processing", enable_auto_fix=True)
    def buggy_find_max(arr, start_idx, end_idx):
        """Find max in array range - has multiple bugs."""
        max_val = arr[start_idx]  # IndexError if start_idx invalid
        for i in range(start_idx + 1, end_idx + 1):  # Off-by-one error
            if arr[i] > max_val:  # IndexError
                max_val = arr[i]
        return max_val
    
    @monitor.autonomous_transform(context="string_processing", enable_auto_fix=True)
    def buggy_process_text(text, prefix):
        """Process text with prefix - has type errors."""
        return prefix.upper() + text.lower()  # AttributeError if prefix is None
    
    # Test cases that will trigger errors and autonomous fixes
    test_cases = [
        (buggy_find_max, ([1, 2, 3], 0, 5), "end_idx out of bounds"),
        (buggy_find_max, ([1, 2, 3], -1, 2), "negative start_idx"),
        (buggy_process_text, ("hello", None), "None prefix"),
        (buggy_find_max, ([], 0, 1), "empty array"),
    ]
    
    print(f"Testing {len(test_cases)} error scenarios...")
    
    fixed_functions = 0
    for i, (func, args, description) in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {description} ---")
        print(f"Calling: {func.__name__}{args}")
        
        try:
            result = func(*args)
            print(f"✅ Success: {result}")
            
        except Exception as e:
            print(f"❌ Error persisted: {type(e).__name__}: {e}")
            
        # Check if function was fixed
        current_impl = monitor.error_fixer.get_current_implementation(func.__name__)
        if current_impl != monitor.error_fixer.function_registry[func.__name__]:
            fixed_functions += 1
            print(f"🔧 Function '{func.__name__}' was autonomously fixed")
    
    print(f"\n🎯 Demo Results:")
    print(f"   Functions autonomously fixed: {fixed_functions}")
    print(f"   Total error scenarios tested: {len(test_cases)}")
    print(f"   Success rate: {fixed_functions/len(test_cases)*100:.1f}%")
    
    return monitor


if __name__ == "__main__":
    # Run the demonstration
    demo_monitor = demo_autonomous_fixing()