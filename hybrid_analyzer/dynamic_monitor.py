import functools
import inspect
import os
import traceback
import tempfile
import json
import time
from typing import Any, Dict, List, Callable, Optional, Union, Tuple

# Import the static analyzer and type conversion function
from static_analyzer import decompose_function, decompose_source, convert_python_type_to_metta

# MeTTa integration through hyperon
try:
    from hyperon import MeTTa
except ImportError:
    print("Hyperon MeTTa not found, will mock MeTTa functionality for demo purposes")
    class MockMeTTa:
        class Space:
            def __init__(self):
                self.atoms = []
                
            def add_atom(self, atom_str):
                """Add an atom to the space"""
                self.atoms.append(atom_str)
                
            def query(self, pattern_str):
                """Query atoms matching the pattern"""
                # Mock implementation for demonstration
                return [a for a in self.atoms if pattern_str in a]
                
            def run_ops(self, code_str):
                """Run MeTTa operations"""
                # Mock implementation
                return None
    
    MeTTa = MockMeTTa

class DynamicMonitor:
    """
    Dynamic runtime monitor for Python functions that integrates with MeTTa.
    Captures execution details and pushes them to MeTTa for reasoning.
    """
    
    def __init__(self, metta_space=None):
        """Initialize the monitor with an optional MeTTa space."""
        self.metta = MeTTa()
        self.metta_space = metta_space or self.metta.space()
    
    def hybrid_transform(self, context: Optional[str] = None, 
                        auto_fix: bool = False):
        """
        Decorator for monitoring Python functions and capturing runtime data for MeTTa.
        
        Args:
            context: Domain context for the function (e.g. "finance", "data_processing")
            auto_fix: Whether to attempt automatic fixes using MeTTa-derived suggestions
            
        Returns:
            Decorated function
        """
        def decorator(func):
            # Get initial static analysis of the function and add to MeTTa
            analysis_result = decompose_function(func)
            
            # Add static analysis atoms to MeTTa space
            if "metta_atoms" in analysis_result and len(analysis_result["metta_atoms"]) > 0:
                for atom in analysis_result["metta_atoms"]:
                    self.metta_space.add_atom(atom)
            else:
                # If static analysis failed, at least register the function
                self.metta_space.add_atom(f"(function {func.__name__})")
            
            # Add context information if provided
            if context:
                # Use proper atom format with hyphen-separated identifiers
                scope_atoms = context.replace(":", "-").split(".")
                context_expr = " ".join(scope_atoms)
                self.metta_space.add_atom(f"(function-context {func.__name__} {context_expr})")
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create a unique execution ID for this call
                exec_id = f"exec-{func.__name__}-{int(time.time()*1000)}"
                
                # Record call start
                start_time = time.time()
                self.metta_space.add_atom(f"(execution-start {exec_id} {func.__name__} {start_time})")
                
                # Capture and record input parameters
                try:
                    input_info = self._capture_inputs(func, exec_id, args, kwargs)
                except Exception as e:
                    # If input capture fails, log it but continue
                    self.metta_space.add_atom(f"(execution-input-error {exec_id} \"{str(e)}\")")
                
                try:
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Record successful execution
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    # Record success in MeTTa
                    self.metta_space.add_atom(f"(execution-success {exec_id} {execution_time})")
                    
                    # Capture and record output value
                    try:
                        self._capture_output(exec_id, result)
                    except Exception as e:
                        # If output capture fails, log it but return the result
                        self.metta_space.add_atom(f"(execution-output-error {exec_id} \"{str(e)}\")")
                    
                    return result
                    
                except Exception as e:
                    # Record failure in MeTTa
                    end_time = time.time()
                    execution_time = end_time - start_time
                    error_type = type(e).__name__
                    error_msg = str(e)
                    
                    # Create unique error ID
                    error_id = f"error-{func.__name__}-{int(time.time()*1000)}"
                    
                    # Record error details in MeTTa
                    self.metta_space.add_atom(f"(execution-error {exec_id} {error_id} {execution_time})")
                    self.metta_space.add_atom(f"(error-type {error_id} {error_type})")
                    self.metta_space.add_atom(f"(error-message {error_id} \"{error_msg}\")")
                    
                    # Capture stack trace information
                    trace = traceback.format_exc()
                    for i, line in enumerate(trace.splitlines()):
                        self.metta_space.add_atom(f"(error-trace {error_id} {i} \"{line}\")")
                    
                    # If auto_fix is enabled, query MeTTa for potential fixes
                    if auto_fix:
                        # Ask MeTTa if it has any fixes for this error type
                        fix_query = f"(match &self (error-fix {error_type} $fix) $fix)"
                        potential_fixes = self.metta_space.query(fix_query)
                        
                        if potential_fixes:
                            # Let MeTTa select the best fix based on context
                            apply_fix_query = f"(match &self (select-fix-for {error_id} {func.__name__} $fix) $fix)"
                            selected_fix = self.metta_space.query(apply_fix_query)
                            
                            if selected_fix:
                                fix_to_apply = selected_fix[0]
                                
                                # Request the fix code from MeTTa
                                fix_code_query = f"(match &self (fix-code {fix_to_apply} $code) $code)"
                                fix_code_result = self.metta_space.query(fix_code_query)
                                
                                if fix_code_result:
                                    # Apply the fix (this would need to be implemented)
                                    # For now, just record that we attempted a fix
                                    self.metta_space.add_atom(f"(fix-attempted {error_id} {fix_to_apply})")
                    
                    # Re-raise the exception
                    raise
            
            return wrapper
        
        return decorator
    
    def _capture_inputs(self, func: Callable, exec_id: str, args: tuple, kwargs: dict) -> Dict:
        """
        Capture function input parameters and add them to MeTTa.
        
        Returns a dictionary with the captured input information.
        """
        # Bind args and kwargs to parameter names
        sig = inspect.signature(func)
        try:
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
        except TypeError as e:
            # If binding fails, record that but continue with what we can capture
            self.metta_space.add_atom(f"(input-binding-error {exec_id} \"{str(e)}\")")
            return {}
        
        # Record each parameter
        input_info = {}
        for i, (param_name, param_value) in enumerate(bound_args.arguments.items()):
            # Capture type and create a MeTTa-compatible type representation
            param_type = type(param_value).__name__
            metta_type = convert_python_type_to_metta(param_type)
            
            # Record parameter in MeTTa
            self.metta_space.add_atom(f"(input-param {exec_id} {i} {param_name} {metta_type})")
            
            # Try to capture sample value for debugging (safely)
            try:
                sample = self._safe_str_sample(param_value)
                if sample:
                    self.metta_space.add_atom(f"(input-sample {exec_id} {param_name} \"{sample}\")")
            except Exception:
                # If sampling fails, just continue
                pass
            
            # Store info for return
            input_info[param_name] = {
                "index": i,
                "type": param_type,
                "metta_type": metta_type
            }
        
        return input_info
    
    def _capture_output(self, exec_id: str, result: Any) -> None:
        """Capture function output value and add it to MeTTa."""
        # Capture result type
        result_type = type(result).__name__
        metta_type = convert_python_type_to_metta(result_type)
        
        # Record output type in MeTTa
        self.metta_space.add_atom(f"(output-type {exec_id} {metta_type})")
        
        # Try to capture sample output value (safely)
        try:
            sample = self._safe_str_sample(result)
            if sample:
                self.metta_space.add_atom(f"(output-sample {exec_id} \"{sample}\")")
        except Exception:
            # If sampling fails, just continue
            pass
    
    def _safe_str_sample(self, value: Any, max_length: int = 100) -> Optional[str]:
        """Create a safe string representation of a value for logging."""
        try:
            str_val = str(value)
            if len(str_val) > max_length:
                return str_val[:max_length] + "..."
            return str_val
        except:
            return None
    
    def query_metta(self, query_pattern: str) -> List[str]:
        """
        Query MeTTa for information.
        
        This allows users to directly query the MeTTa space for insights and recommendations.
        """
        return self.metta_space.query(query_pattern)
    
    def get_function_recommendations(self, func_name: str) -> List[Dict]:
        """
        Get improvement recommendations for a function by querying MeTTa.
        """
        # Query MeTTa for recommendations instead of calculating in Python
        recommendation_query = f"(match &self (function-recommendation {func_name} $type $description $confidence) ($type $description $confidence))"
        raw_recommendations = self.metta_space.query(recommendation_query)
        
        # Parse the results
        recommendations = []
        for raw in raw_recommendations:
            # Parse the raw string result into components
            # This parsing depends on the exact format returned by MeTTa
            # For demo purposes, we'll just create a simple dictionary
            try:
                parts = raw.strip('()').split(' ', 2)
                rec_type = parts[0]
                confidence = float(parts[-1])
                description = parts[1] if len(parts) == 3 else "No description available"
                
                recommendations.append({
                    "type": rec_type,
                    "description": description,
                    "confidence": confidence
                })
            except Exception as e:
                print(f"Error parsing recommendation: {e}")
        
        return recommendations
    
    def get_error_patterns(self, func_name: str) -> List[Dict]:
        """
        Get error patterns for a function by querying MeTTa.
        """
        # Query MeTTa for error patterns
        pattern_query = f"(match &self (function-error-pattern {func_name} $error_type $frequency $description) ($error_type $frequency $description))"
        raw_patterns = self.metta_space.query(pattern_query)
        
        # Parse the results
        patterns = []
        for raw in raw_patterns:
            try:
                parts = raw.strip('()').split(' ', 2)
                error_type = parts[0]
                frequency = int(parts[1]) if len(parts) > 1 else 0
                description = parts[2] if len(parts) > 2 else "Unknown pattern"
                
                patterns.append({
                    "error_type": error_type,
                    "frequency": frequency,
                    "description": description
                })
            except Exception as e:
                print(f"Error parsing error pattern: {e}")
        
        return patterns
    
    def load_metta_rules(self, rules_file: str) -> bool:
        """
        Load MeTTa rules from a file.
        
        This allows users to add domain-specific reasoning rules to the MeTTa space.
        """
        try:
            with open(rules_file, 'r') as f:
                rules = f.read()
            
            # Add each line as a separate atom
            for line in rules.strip().split('\n'):
                if line.strip() and not line.strip().startswith(';'):  # Skip empty lines and comments
                    self.metta_space.add_atom(line.strip())
            
            return True
        except Exception as e:
            print(f"Error loading MeTTa rules: {e}")
            return False


# Create a global instance for easy access
monitor = DynamicMonitor()

# Decorator for convenience
def hybrid_transform(context=None, auto_fix=False):
    """Decorator for monitoring Python functions and pushing data to MeTTa."""
    return monitor.hybrid_transform(context, auto_fix)