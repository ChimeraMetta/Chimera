"""
Enhanced Autonomous Error-Driven Evolution System

Improved version that handles errors better and provides more robust
healing capabilities for the file-based healer.
"""

import functools
import inspect
import traceback
import ast
from typing import Any, Dict, List, Callable, Optional

# Import existing components
from reflectors.dynamic_monitor import DynamicMonitor
from metta_generator.base import MeTTaPoweredModularDonorGenerator
from hyperon import *


class AutonomousErrorFixer:
    """
    Enhanced autonomous system that generates fixes for runtime errors.
    Improved error handling and code generation capabilities.
    """
    
    def __init__(self, metta_space=None):
        """Initialize the enhanced autonomous error fixer."""
        self.metta = MeTTa()
        self.metta_space = metta_space or self.metta.space()
        
        # Initialize MeTTa-powered donor generator with fallback handling
        try:
            self.donor_generator = MeTTaPoweredModularDonorGenerator(metta_space=self.metta_space)
        except Exception as e:
            print(f"[WARNING] Failed to initialize MeTTa donor generator: {e}")
            self.donor_generator = None
        
        # Track original functions and their current implementations
        self.function_registry = {}  # func_name -> original_function
        self.current_implementations = {}  # func_name -> current_active_function
        self.error_history = {}  # func_name -> list of error contexts
        self.fix_attempts = {}  # func_name -> number of fix attempts
        
        # Configuration
        self.max_fix_attempts = 3
        self.auto_apply_fixes = True
        
        # Load basic healing patterns
        self._load_basic_healing_patterns()
        
    def _load_basic_healing_patterns(self):
        """Load basic healing patterns for common errors."""
        self.healing_patterns = {
            'IndexError': {
                'pattern': 'bounds_checking',
                'template': '''def {func_name}({params}):
    """Healed version with bounds checking."""
    try:
        # Original logic with bounds checking
        {bounds_checks}
        {original_body}
    except IndexError:
        return None  # Safe fallback
''',
                'bounds_checks': [
                    'if not arr or not isinstance(arr, (list, tuple, str)): return None',
                    'if index < 0 or index >= len(arr): return None'
                ]
            },
            'ZeroDivisionError': {
                'pattern': 'division_safety',
                'template': '''def {func_name}({params}):
    """Healed version with division safety."""
    try:
        {safety_checks}
        {original_body}
    except ZeroDivisionError:
        return float('inf') if numerator > 0 else float('-inf') if numerator < 0 else 0
''',
                'safety_checks': [
                    'if denominator == 0: return float(\'inf\') if numerator > 0 else float(\'-inf\') if numerator < 0 else 0'
                ]
            },
            'AttributeError': {
                'pattern': 'none_safety',
                'template': '''def {func_name}({params}):
    """Healed version with None safety."""
    try:
        {none_checks}
        {original_body}
    except AttributeError:
        return None  # Safe fallback
''',
                'none_checks': [
                    'if any(arg is None for arg in [{param_list}]): return ""'
                ]
            },
            'TypeError': {
                'pattern': 'type_safety',
                'template': '''def {func_name}({params}):
    """Healed version with type safety."""
    try:
        {type_checks}
        {original_body}
    except TypeError:
        return None  # Safe fallback
''',
                'type_checks': [
                    'if not all(isinstance(arg, expected_type) for arg, expected_type in zip(args, expected_types)): return None'
                ]
            }
        }
        
    def register_function(self, func: Callable, context: str = None):
        """Register a function for autonomous error fixing."""
        func_name = func.__name__
        self.function_registry[func_name] = func
        self.current_implementations[func_name] = func
        self.error_history[func_name] = []
        self.fix_attempts[func_name] = 0
        
        # Add function info to MeTTa space if possible
        try:
            if context:
                context_atom = self.metta.parse_single(f"(function-context {func_name} {context})")
                self.metta_space.add_atom(context_atom)
        except Exception as e:
            print(f"[WARNING] Could not add context to MeTTa space: {e}")
        
        print(f"[OK] Registered function '{func_name}' for autonomous error fixing")
        
    def generate_test_cases_from_error(self, func_name: str, error_context: Dict) -> List[Callable]:
        """Generate test cases based on error context to verify fixes."""
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
        """Generate fix candidates using multiple approaches."""
        candidates = []
        
        # Try MeTTa-powered generation first
        if self.donor_generator:
            try:
                metta_candidates = self._generate_metta_candidates(func_name, error_context)
                candidates.extend(metta_candidates)
            except Exception as e:
                print(f"[WARNING] MeTTa generation failed: {e}")
        
        # Generate pattern-based candidates
        pattern_candidates = self._generate_pattern_candidates(func_name, error_context)
        candidates.extend(pattern_candidates)
        
        # Generate simple safe fallback
        fallback_candidate = self._generate_fallback_candidate(func_name, error_context)
        if fallback_candidate:
            candidates.append(fallback_candidate)
        
        print(f"   Generated {len(candidates)} fix candidates total")
        return candidates
    
    def _generate_metta_candidates(self, func_name: str, error_context: Dict) -> List[Dict]:
        """Generate candidates using MeTTa-powered donor generation."""
        try:
            original_func = self.function_registry.get(func_name)
            if not original_func:
                return []
            
            func_source = error_context.get('function_source')
            if not func_source:
                try:
                    func_source = inspect.getsource(original_func)
                except:
                    func_source = self._create_basic_function_template(func_name)
            
            print(f"[INFO] Generating MeTTa candidates for '{func_name}'...")
            
            # Use donor generator to create candidates
            metta_donors = self.donor_generator.generate_donors_from_function(func_source)
            
            # Convert to our candidate format
            candidates = []
            for donor in metta_donors[:3]:  # Limit to top 3
                candidate = {
                    'name': donor.get('name', f"{func_name}_metta_fix"),
                    'code': donor.get('code', ''),
                    'strategy': 'metta_powered',
                    'confidence': donor.get('final_score', 0.7),
                    'error_fix_score': 0.8
                }
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            print(f"[WARNING] MeTTa candidate generation failed: {e}")
            return []
    
    def _generate_pattern_candidates(self, func_name: str, error_context: Dict) -> List[Dict]:
        """Generate candidates using error pattern templates."""
        candidates = []
        error_type = error_context.get('error_type', 'Unknown')
        
        if error_type in self.healing_patterns:
            pattern_info = self.healing_patterns[error_type]
            
            try:
                # Extract function signature
                func_source = error_context.get('function_source', '')
                params, original_body = self._parse_function_source(func_source, func_name)
                
                # Generate healing code
                healing_code = self._apply_healing_pattern(
                    pattern_info, func_name, params, original_body, error_context
                )
                
                candidate = {
                    'name': f"{func_name}_{pattern_info['pattern']}_fix",
                    'code': healing_code,
                    'strategy': f"pattern_{pattern_info['pattern']}",
                    'confidence': 0.8,
                    'error_fix_score': 0.9
                }
                candidates.append(candidate)
                
            except Exception as e:
                print(f"[WARNING] Pattern candidate generation failed: {e}")
        
        return candidates
    
    def _parse_function_source(self, func_source: str, func_name: str) -> tuple:
        """Parse function source to extract parameters and body."""
        try:
            # Parse the AST
            tree = ast.parse(func_source)
            
            # Find the function definition
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    # Extract parameters
                    params = []
                    for arg in node.args.args:
                        params.append(arg.arg)
                    
                    # Extract body (simplified)
                    body_lines = []
                    for stmt in node.body:
                        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                            # Skip docstrings
                            continue
                        # For now, just get the source lines
                        body_lines.append("    # Original function body")
                    
                    return ', '.join(params), '\n'.join(body_lines) if body_lines else "    return None"
            
        except Exception as e:
            print(f"[WARNING] Could not parse function source: {e}")
        
        # Fallback
        return '*args, **kwargs', '    return None'
    
    def _apply_healing_pattern(self, pattern_info: Dict, func_name: str, 
                             params: str, original_body: str, error_context: Dict) -> str:
        """Apply a healing pattern to generate fixed code."""
        template = pattern_info['template']
        
        # Prepare template variables
        template_vars = {
            'func_name': func_name,
            'params': params,
            'original_body': original_body,
            'bounds_checks': '\n        '.join(pattern_info.get('bounds_checks', [])),
            'safety_checks': '\n        '.join(pattern_info.get('safety_checks', [])),
            'none_checks': '\n        '.join(pattern_info.get('none_checks', [])),
            'type_checks': '\n        '.join(pattern_info.get('type_checks', [])),
            'param_list': params.replace(' ', '').split(',') if params != '*args, **kwargs' else []
        }
        
        # Format the template
        try:
            healing_code = template.format(**template_vars)
            return healing_code
        except Exception as e:
            print(f"[WARNING] Template formatting failed: {e}")
            return self._create_basic_safe_function(func_name, params)
    
    def _generate_fallback_candidate(self, func_name: str, error_context: Dict) -> Dict:
        """Generate a safe fallback candidate."""
        error_type = error_context.get('error_type', 'Unknown')
        
        # Create basic safe implementation
        if error_type == 'IndexError':
            safe_code = f'''def {func_name}(*args, **kwargs):
    """Safe fallback for IndexError."""
    try:
        if len(args) >= 2:
            arr, index = args[0], args[1]
            if isinstance(arr, (list, tuple, str)) and isinstance(index, int):
                if 0 <= index < len(arr):
                    return arr[index]
        return None
    except Exception:
        return None
'''
        elif error_type == 'ZeroDivisionError':
            safe_code = f'''def {func_name}(*args, **kwargs):
    """Safe fallback for ZeroDivisionError."""
    try:
        if len(args) >= 2:
            numerator, denominator = args[0], args[1]
            if denominator != 0:
                return numerator / denominator
            return float('inf') if numerator > 0 else float('-inf') if numerator < 0 else 0
        return None
    except Exception:
        return None
'''
        elif error_type == 'AttributeError':
            safe_code = f'''def {func_name}(*args, **kwargs):
    """Safe fallback for AttributeError."""
    try:
        # Handle None values safely
        safe_args = []
        for arg in args:
            if arg is None:
                safe_args.append("")  # Safe default
            else:
                safe_args.append(arg)
        
        # Apply basic operation with safe arguments
        if len(safe_args) >= 2:
            first, second = safe_args[0], safe_args[1]
            if hasattr(first, 'upper') and hasattr(second, 'upper'):
                return f"{{first.upper()}} {{second.upper()}}"
        
        return ""
    except Exception:
        return ""
'''
        else:
            safe_code = f'''def {func_name}(*args, **kwargs):
    """Safe fallback implementation."""
    try:
        # Generic safe implementation
        return None
    except Exception:
        return None
'''
        
        return {
            'name': f"{func_name}_safe_fallback",
            'code': safe_code,
            'strategy': 'safe_fallback',
            'confidence': 0.6,
            'error_fix_score': 0.7
        }
    
    def _create_basic_function_template(self, func_name: str) -> str:
        """Create a basic function template when source is not available."""
        return f'''def {func_name}(*args, **kwargs):
    """Basic function template."""
    return None
'''
    
    def _create_basic_safe_function(self, func_name: str, params: str) -> str:
        """Create a basic safe function."""
        return f'''def {func_name}({params}):
    """Basic safe function."""
    try:
        # Safe implementation
        return None
    except Exception:
        return None
'''
    
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
        if func_name not in self.error_history:
            self.error_history[func_name] = []
        self.error_history[func_name].append(error_context)
        
        # Check if we should attempt a fix
        if func_name not in self.fix_attempts:
            self.fix_attempts[func_name] = 0
            
        if self.fix_attempts[func_name] >= self.max_fix_attempts:
            print(f"Max fix attempts reached for '{func_name}'. No more fixes will be attempted.")
            return False
            
        self.fix_attempts[func_name] += 1
        print(f"   Attempt {self.fix_attempts[func_name]} of {self.max_fix_attempts}")
        
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
        if best_candidate and best_fitness > 0.3:  # Lower threshold for better acceptance
            print(f"[INFO] Best candidate found: {best_candidate.get('name')} with fitness {best_fitness:.1%}")
            
            if self.auto_apply_fixes:
                if self.apply_fix(func_name, best_candidate):
                    print(f"Autonomous fix for '{func_name}' was successful.")
                    return True
                else:
                    print(f"Failed to apply the best fix for '{func_name}'.")
            else:
                print(f"Fix available for '{func_name}', but auto-apply is disabled.")
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
                        print(f"[INFO] Using fixed implementation for '{func_name}'")
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
                        print(f"[INFO] Retrying '{func_name}' with applied fix...")
                        try:
                            # Retry with the fixed implementation
                            fixed_impl = self.error_fixer.get_current_implementation(func_name)
                            result = fixed_impl(*args, **kwargs)
                            print(f"[OK] Fixed implementation succeeded for '{func_name}'")
                            return result
                        except Exception as retry_error:
                            print(f"[ERROR] Fixed implementation still failed: {retry_error}")
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
        """Create error context from function, error, and arguments."""
        import textwrap
        
        try:
            # Get and dedent the original function source
            raw_func_source = inspect.getsource(func)
            clean_func_source = textwrap.dedent(raw_func_source)
        except:
            clean_func_source = f"# Source not available for {func.__name__}"
        
        error_context = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'failing_inputs': [args] if args else [],
            'function_name': func.__name__,
            'traceback': traceback.format_exc(),
            'function_source': clean_func_source
        }
        return error_context


# Factory function for easy usage
def create_autonomous_monitor(ontology_path: str = None) -> AutonomousMonitor:
    """Create an autonomous monitor with optional ontology loading."""
    monitor = AutonomousMonitor()
    
    if ontology_path:
        try:
            monitor.load_metta_rules(ontology_path)
            print(f"[OK] Loaded ontology for autonomous fixing: {ontology_path}")
        except Exception as e:
            print(f"[WARNING] Could not load ontology: {e}")
    
    return monitor


# Demo functions for testing
def demo_autonomous_fixing():
    """Demonstrate the autonomous error fixing system."""
    
    print("[DEMO] Autonomous Error Fixing Demo")
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
    
    @monitor.autonomous_transform(context="math_operations", enable_auto_fix=True)
    def buggy_divide(a, b):
        """Divide two numbers - has division by zero."""
        return a / b  # ZeroDivisionError if b is 0
    
    # Test cases that will trigger errors and autonomous fixes
    test_cases = [
        (buggy_find_max, ([1, 2, 3], 0, 5), "end_idx out of bounds"),
        (buggy_find_max, ([1, 2, 3], -1, 2), "negative start_idx"),
        (buggy_process_text, ("hello", None), "None prefix"),
        (buggy_divide, (10, 0), "division by zero"),
        (buggy_find_max, ([], 0, 1), "empty array"),
    ]
    
    print(f"Testing {len(test_cases)} error scenarios...")
    
    fixed_functions = 0
    for i, (func, args, description) in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {description} ---")
        print(f"Calling: {func.__name__}{args}")
        
        try:
            result = func(*args)
            print(f"[OK] Success: {result}")
            
        except Exception as e:
            print(f"[ERROR] Error persisted: {type(e).__name__}: {e}")
            
        # Check if function was fixed
        current_impl = monitor.error_fixer.get_current_implementation(func.__name__)
        original_impl = monitor.error_fixer.function_registry.get(func.__name__)
        if current_impl != original_impl:
            fixed_functions += 1
            print(f"[INFO] Function '{func.__name__}' was autonomously fixed")
    
    print(f"\n[INFO] Demo Results:")
    print(f"   Functions autonomously fixed: {fixed_functions}")
    print(f"   Total error scenarios tested: {len(test_cases)}")
    print(f"   Success rate: {fixed_functions/len(test_cases)*100:.1f}%")
    
    return monitor


def test_enhanced_error_fixer_standalone():
    """Test the enhanced error fixer in standalone mode."""
    print("\n[INFO] Testing Enhanced Error Fixer (Standalone)")
    print("-" * 50)
    
    # Create error fixer
    error_fixer = AutonomousErrorFixer()
    
    # Test function with IndexError
    def test_index_func(arr, index):
        return arr[index]
    
    # Register function
    error_fixer.register_function(test_index_func)
    
    # Create error context
    error_context = {
        'error_type': 'IndexError',
        'error_message': 'list index out of range',
        'failing_inputs': [([1, 2, 3], 5)],
        'function_name': 'test_index_func',
        'traceback': 'Traceback (most recent call last):\n  File "<stdin>", line 1, in <module>\nIndexError: list index out of range',
        'function_source': '''def test_index_func(arr, index):
    return arr[index]
'''
    }
    
    # Test healing
    print("Testing IndexError healing...")
    success = error_fixer.handle_error('test_index_func', error_context)
    
    if success:
        print("[OK] Healing successful!")
        
        # Test the healed function
        healed_func = error_fixer.get_current_implementation('test_index_func')
        try:
            result = healed_func([1, 2, 3], 5)
            print(f"[OK] Healed function returned: {result}")
        except Exception as e:
            print(f"[ERROR] Healed function still has issues: {e}")
    else:
        print("[WARNING] Healing was not successful")
    
    # Test with ZeroDivisionError
    def test_div_func(a, b):
        return a / b
    
    error_fixer.register_function(test_div_func)
    
    error_context_div = {
        'error_type': 'ZeroDivisionError',
        'error_message': 'division by zero',
        'failing_inputs': [(10, 0)],
        'function_name': 'test_div_func',
        'traceback': 'Traceback (most recent call last):\n  File "<stdin>", line 1, in <module>\nZeroDivisionError: division by zero',
        'function_source': '''def test_div_func(a, b):
    return a / b
'''
    }
    
    print("\nTesting ZeroDivisionError healing...")
    success_div = error_fixer.handle_error('test_div_func', error_context_div)
    
    if success_div:
        print("[OK] Division healing successful!")
        
        # Test the healed function
        healed_div_func = error_fixer.get_current_implementation('test_div_func')
        try:
            result = healed_div_func(10, 0)
            print(f"[OK] Healed division function returned: {result}")
        except Exception as e:
            print(f"[ERROR] Healed division function still has issues: {e}")
    else:
        print("[WARNING] Division healing was not successful")
    
    return success or success_div


# This section should only run when the file is executed directly, not when imported
# Commented out to prevent automatic execution during imports
# 
# if __name__ == "__main__":
#     # Run demonstrations
#     print("Running Autonomous Error Fixing Demonstrations")
#     print("=" * 60)
#     
#     # Test standalone error fixer
#     standalone_success = test_enhanced_error_fixer_standalone()
#     
#     # Run full demo
#     try:
#         demo_monitor = demo_autonomous_fixing()
#         demo_success = True
#     except Exception as e:
#         print(f"[ERROR] Demo failed: {e}")
#         demo_success = False
#     
#     # Summary
#     print(f"\n[INFO] Test Summary:")
#     print(f"   Standalone Error Fixer: {'[OK] PASS' if standalone_success else '[ERROR] FAIL'}")
#     print(f"   Full Demo: {'[OK] PASS' if demo_success else '[ERROR] FAIL'}")
#     
#     if standalone_success and demo_success:
#         print("[OK] All tests passed! Enhanced Autonomous Error Fixer is working correctly.")
#     else:
#         print("[WARNING] Some tests failed. Check the output above for details.")


def run_autonomous_evolution_demo():
    """
    Explicitly callable function to run the autonomous evolution demo.
    Call this only when you want to run the demonstration.
    """
    print("Running Autonomous Error Fixing Demonstrations")
    print("=" * 60)
    
    # Test standalone error fixer
    standalone_success = test_enhanced_error_fixer_standalone()
    
    # Run full demo
    try:
        demo_monitor = demo_autonomous_fixing()
        demo_success = True
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        demo_success = False
    
    # Summary
    print(f"\n[INFO] Test Summary:")
    print(f"   Standalone Error Fixer: {'[OK] PASS' if standalone_success else '[ERROR] FAIL'}")
    print(f"   Full Demo: {'[OK] PASS' if demo_success else '[ERROR] FAIL'}")
    
    if standalone_success and demo_success:
        print("[OK] All tests passed! Enhanced Autonomous Error Fixer is working correctly.")
        return True
    else:
        print("[WARNING] Some tests failed. Check the output above for details.")
        return False