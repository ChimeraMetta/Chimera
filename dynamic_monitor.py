import functools
import inspect
import os
import ast
import traceback
import tempfile
import json
import time
from typing import Any, Dict, List, Callable, Optional, Union, Tuple

# Import the static analyzer
from static_analyzer import decompose_function, decompose_source

# MeTTa integration (needs metta package or custom implementation)
try:
    import metta
except ImportError:
    print("MeTTa not found, will mock MeTTa functionality for demo purposes")
    class MockMeTTa:
        def __init__(self):
            self.atoms = []
            
        def add_atom(self, atom: str):
            self.atoms.append(atom)
            
        def query(self, pattern: str) -> List[str]:
            # Mock implementation for demonstration
            return [a for a in self.atoms if pattern in a]
            
        def execute(self, code: str) -> Any:
            # Mock implementation
            return None
    
    metta = MockMeTTa()

class DynamicMonitor:
    """
    Dynamic runtime monitor for Python functions that integrates with MeTTa reasoning.
    Captures execution details and provides feedback based on MeTTa analysis.
    """
    
    def __init__(self, metta_space=None):
        """Initialize the monitor with an optional MeTTa space."""
        self.metta_space = metta_space or metta.MeTTaSpace()
        self.success_records = {}
        self.execution_history = {}
        self.error_patterns = {}
        self.function_metrics = {}
        
        # Ensure we have the donor system loaded
        self._ensure_donor_system_loaded()
    
    def _ensure_donor_system_loaded(self):
        """Load the donor system into MeTTa if it's not already loaded."""
        # Check if donor system is loaded (by querying for a known atom)
        donor_check = self.metta_space.query("(fragment-donor $name $code)")
        
        if not donor_check:
            # Load donor system from a file or define basic donors inline
            donors = [
                "(= (fragment-donor \"python_string_concat\") \"str1 + str2\")",
                "(= (fragment-donor \"python_f_string\") \"f\\\"{variable}\\\"\")",
                "(= (fragment-donor \"python_format_string\") \"\\\"{}\\\".format(variable)\")",
                "(= (fragment-donor \"python_string_join\") \"\\\", \\\".join(items)\")",
                "(= (fragment-donor \"python_list_comprehension\") \"[x for x in items]\")",
                "(= (fragment-donor \"python_zero_division_check\") \"if divisor != 0:\\n    result = dividend / divisor\\nelse:\\n    result = float('inf')\")",
                
                # Operation donors
                "(= (operation-donor Add String String) (fragment-donor \"python_string_concat\"))",
                "(= (operation-donor Add String Number) (fragment-donor \"python_f_string\"))",
                "(= (operation-donor Add Number String) (fragment-donor \"python_f_string\"))",
                "(= (operation-donor Div Number Number) (fragment-donor \"python_zero_division_check\"))",
                
                # Type conversion donors
                "(= (type-donor String Number) \"float(string_value)\")",
                "(= (type-donor Number String) \"str(number_value)\")",
                
                # Function donors
                "(= (function-donor \"safe_division\") \"def safe_divide(a: float, b: float, default: float = float('inf')) -> float:\\n    try:\\n        return a / b\\n    except ZeroDivisionError:\\n        return default\")",
                
                # Success tracking
                "(= (record-fix-attempt $error_signature $fix $success) (case (match-atom &self (success-record $error_signature $fix $timestamp $success_count $failure_count)) (Empty (case $success (True (add-atom &self (success-record $error_signature $fix (current-time) 1 0))) (False (add-atom &self (success-record $error_signature $fix (current-time) 0 1))))) ((success-record $error_signature $fix $timestamp $success_count $failure_count) (case $success (True (update-atom &self (success-record $error_signature $fix $timestamp $success_count $failure_count) (success-record $error_signature $fix (current-time) (+ $success_count 1) $failure_count))) (False (update-atom &self (success-record $error_signature $fix $timestamp $success_count $failure_count) (success-record $error_signature $fix (current-time) $success_count (+ $failure_count 1)))))))"
            ]
            
            for donor in donors:
                self.metta_space.add_atom(donor)
    
    def hybrid_transform(self, context: Optional[str] = None, 
                        auto_fix: bool = False, 
                        collect_metrics: bool = True):
        """
        Decorator for monitoring and potentially transforming Python functions.
        
        Args:
            context: Domain context for the function (e.g. "finance", "data_processing")
            auto_fix: Whether to automatically apply suggested fixes
            collect_metrics: Whether to collect performance and usage metrics
            
        Returns:
            Decorated function
        """
        def decorator(func):
            # Get initial static analysis of the function
            analysis_result = decompose_function(func)
            
            # Add static analysis atoms to MeTTa space
            if "metta_atoms" in analysis_result:
                for atom in analysis_result["metta_atoms"]:
                    self.metta_space.add_atom(atom)
            
            # Add context information if provided
            if context:
                self.metta_space.add_atom(f"(function-context {func.__name__} \"{context}\")")
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                input_signature = self._get_input_signature(func, args, kwargs)
                error_occurred = None
                exception_info = None
                result = None
                
                try:
                    # Execute the function
                    result = func(*args, **kwargs)
                    return_type = type(result).__name__
                    
                    # Record successful execution
                    self._record_execution(
                        func_name=func.__name__,
                        input_signature=input_signature,
                        return_type=return_type,
                        execution_time=time.time() - start_time,
                        success=True
                    )
                    
                    return result
                    
                except Exception as e:
                    # Capture the exception
                    error_occurred = True
                    exception_info = {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc()
                    }
                    
                    # Record failed execution
                    self._record_execution(
                        func_name=func.__name__,
                        input_signature=input_signature,
                        return_type=None,
                        execution_time=time.time() - start_time,
                        success=False,
                        error=exception_info
                    )
                    
                    # Attempt to fix the error if auto_fix is enabled
                    if auto_fix:
                        fixed_result = self._attempt_fix(
                            func=func,
                            args=args,
                            kwargs=kwargs,
                            error=exception_info
                        )
                        if fixed_result["success"]:
                            # Log the successful fix
                            print(f"Auto-fixed error in {func.__name__}: {exception_info['type']}")
                            return fixed_result["result"]
                    
                    # Re-raise the exception if not fixed
                    raise
                
                finally:
                    # Update metrics regardless of success/failure
                    if collect_metrics:
                        self._update_metrics(
                            func_name=func.__name__,
                            execution_time=time.time() - start_time,
                            success=not error_occurred
                        )
            
            return wrapper
        
        return decorator
    
    def _get_input_signature(self, func: Callable, args: tuple, kwargs: dict) -> Dict:
        """Extract a signature of function inputs for pattern matching."""
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        input_types = {}
        for param_name, param_value in bound_args.arguments.items():
            input_types[param_name] = {
                "type": type(param_value).__name__,
                "sample": self._safe_str_sample(param_value)
            }
        
        return input_types
    
    def _safe_str_sample(self, value: Any, max_length: int = 100) -> str:
        """Create a safe string representation of a value for logging."""
        try:
            str_val = str(value)
            if len(str_val) > max_length:
                return str_val[:max_length] + "..."
            return str_val
        except:
            return f"<unprintable {type(value).__name__}>"
    
    def _record_execution(self, func_name: str, input_signature: Dict, 
                         return_type: Optional[str], execution_time: float,
                         success: bool, error: Optional[Dict] = None) -> None:
        """Record function execution details for MeTTa reasoning."""
        # Create a unique execution ID
        exec_id = f"exec_{func_name}_{int(time.time()*1000)}"
        
        # Add execution record to MeTTa
        self.metta_space.add_atom(f"(execution {exec_id} {func_name} {success})")
        
        # Add input information
        for param_name, param_info in input_signature.items():
            self.metta_space.add_atom(
                f"(execution-param {exec_id} {param_name} {param_info['type']})"
            )
        
        # Add output information if successful
        if success and return_type:
            self.metta_space.add_atom(f"(execution-return {exec_id} {return_type})")
        
        # Add error information if failed
        if not success and error:
            error_id = f"error_{func_name}_{int(time.time()*1000)}"
            self.metta_space.add_atom(f"(execution-error {exec_id} {error_id})")
            self.metta_space.add_atom(f"(error {error_id} {error['type']})")
            
            # Parse error information to identify patterns
            self._analyze_error(func_name, error, input_signature)
        
        # Store in internal history
        if func_name not in self.execution_history:
            self.execution_history[func_name] = []
        
        self.execution_history[func_name].append({
            "id": exec_id,
            "timestamp": time.time(),
            "inputs": input_signature,
            "return_type": return_type,
            "execution_time": execution_time,
            "success": success,
            "error": error
        })
    
    def _analyze_error(self, func_name: str, error: Dict, input_signature: Dict) -> None:
        """Analyze an error to identify patterns and potential fixes."""
        error_type = error["type"]
        error_msg = error["message"]
        
        # Create an error signature for pattern matching
        error_sig = f"{func_name}:{error_type}"
        
        # Add to error patterns
        if error_sig not in self.error_patterns:
            self.error_patterns[error_sig] = {
                "count": 0,
                "samples": []
            }
        
        self.error_patterns[error_sig]["count"] += 1
        self.error_patterns[error_sig]["samples"].append({
            "message": error_msg,
            "inputs": input_signature,
            "timestamp": time.time()
        })
        
        # Limit sample size
        if len(self.error_patterns[error_sig]["samples"]) > 10:
            self.error_patterns[error_sig]["samples"] = self.error_patterns[error_sig]["samples"][-10:]
        
        # Add to MeTTa for reasoning
        self.metta_space.add_atom(f"(error-pattern {error_sig} {self.error_patterns[error_sig]['count']})")
        
        # Look for common patterns in error messages
        if "division by zero" in error_msg:
            self.metta_space.add_atom(f"(zero-division-error {error_sig})")
        elif "index out of range" in error_msg or "list index out of range" in error_msg:
            self.metta_space.add_atom(f"(index-error {error_sig})")
        elif "NoneType" in error_msg and "has no attribute" in error_msg:
            self.metta_space.add_atom(f"(none-attribute-error {error_sig})")
        elif "KeyError" in error_type:
            self.metta_space.add_atom(f"(key-error {error_sig})")
    
    def _update_metrics(self, func_name: str, execution_time: float, success: bool) -> None:
        """Update performance metrics for a function."""
        if func_name not in self.function_metrics:
            self.function_metrics[func_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_time": 0,
                "avg_time": 0,
                "min_time": float('inf'),
                "max_time": 0
            }
        
        metrics = self.function_metrics[func_name]
        metrics["calls"] += 1
        if success:
            metrics["successes"] += 1
        else:
            metrics["failures"] += 1
        
        metrics["total_time"] += execution_time
        metrics["avg_time"] = metrics["total_time"] / metrics["calls"]
        metrics["min_time"] = min(metrics["min_time"], execution_time)
        metrics["max_time"] = max(metrics["max_time"], execution_time)
        
        # Add metric info to MeTTa
        self.metta_space.add_atom(
            f"(function-metrics {func_name} {metrics['calls']} {metrics['successes']} {metrics['failures']} {metrics['avg_time']:.6f})"
        )
    
    def _attempt_fix(self, func: Callable, args: tuple, kwargs: dict, error: Dict) -> Dict:
        """Attempt to automatically fix an error based on MeTTa reasoning."""
        result = {
            "success": False,
            "result": None,
            "fix_applied": None
        }
        
        func_name = func.__name__
        error_type = error["type"]
        error_sig = f"{func_name}:{error_type}"
        
        # Get function source code
        try:
            source = inspect.getsource(func)
        except Exception as e:
            return result
        
        # Query MeTTa for possible fixes
        fix_query = f"(find-fix {error_sig})"
        fixes = self.metta_space.query(fix_query)
        
        if not fixes:
            # If no specific fix is found, try general fixes based on error type
            if "ZeroDivisionError" in error_type:
                # For zero division errors, apply safe division
                fix = self.metta_space.query("(function-donor \"safe_division\")")
                if fix:
                    # Extract the safe division function
                    safe_div_code = fix[0].split("\"def ")[1].rsplit("\"", 1)[0]
                    safe_div_code = "def " + safe_div_code
                    
                    # Try to apply the fix
                    modified_source = self._apply_zero_division_fix(source, safe_div_code)
                    if modified_source != source:
                        fix_to_apply = "safe_division"
                    else:
                        return result
            elif "IndexError" in error_type:
                # For index errors, add bounds checking
                modified_source = self._apply_index_check_fix(source)
                if modified_source != source:
                    fix_to_apply = "index_check"
                else:
                    return result
            elif "TypeError" in error_type and ("str" in error["message"] or "string" in error["message"]):
                # For type errors involving strings
                modified_source = self._apply_string_conversion_fix(source)
                if modified_source != source:
                    fix_to_apply = "string_conversion"
                else:
                    return result
            else:
                return result
        else:
            # Use the first suggested fix
            fix_to_apply = fixes[0]
            # TODO: Extract fix code from MeTTa result
            # For now, return as not implemented
            return result
        
        # Execute the modified function
        try:
            # Create a temporary module to execute the fixed function
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
                temp_file_name = temp_file.name
                temp_file.write(modified_source.encode('utf-8'))
            
            # Import the temporary module
            import importlib.util
            spec = importlib.util.spec_from_file_location("fixed_module", temp_file_name)
            fixed_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fixed_module)
            
            # Get the fixed function
            fixed_func = getattr(fixed_module, func_name)
            
            # Execute the fixed function
            fixed_result = fixed_func(*args, **kwargs)
            
            # Record the successful fix
            self._record_fix_success(error_sig, fix_to_apply)
            
            result = {
                "success": True,
                "result": fixed_result,
                "fix_applied": fix_to_apply
            }
            
            # Clean up the temporary file
            os.unlink(temp_file_name)
            
        except Exception as e:
            # Record the failed fix attempt
            self._record_fix_failure(error_sig, fix_to_apply)
        
        return result
    
    def _apply_zero_division_fix(self, source: str, safe_div_code: str) -> str:
        """Apply a zero division fix to the source code."""
        try:
            # Parse the source
            tree = ast.parse(source)
            
            # Create a new module with the safe division function
            safe_div_tree = ast.parse(safe_div_code)
            
            # Create a transformer to replace division operations
            class DivisionTransformer(ast.NodeTransformer):
                def visit_BinOp(self, node):
                    self.generic_visit(node)
                    if isinstance(node.op, ast.Div):
                        # Replace with safe_divide call
                        return ast.Call(
                            func=ast.Name(id='safe_divide', ctx=ast.Load()),
                            args=[node.left, node.right],
                            keywords=[]
                        )
                    return node
            
            # Apply the transformation
            transformer = DivisionTransformer()
            transformed_tree = transformer.visit(tree)
            
            # Add the safe_divide function to the module
            transformed_tree.body = safe_div_tree.body + transformed_tree.body
            
            # Convert back to source
            return ast.unparse(transformed_tree)
        except Exception as e:
            # If transformation fails, return original source
            return source
    
    def _apply_index_check_fix(self, source: str) -> str:
        """Apply an index check fix to the source code."""
        try:
            # Parse the source
            tree = ast.parse(source)
            
            # Create a transformer to add index checks
            class IndexCheckTransformer(ast.NodeTransformer):
                def visit_Subscript(self, node):
                    self.generic_visit(node)
                    
                    # Only process simple list or dict subscripts
                    if not isinstance(node.value, ast.Name):
                        return node
                    
                    # Create a check node
                    if isinstance(node.slice, ast.Index):  # Python 3.8
                        index = node.slice.value
                    else:  # Python 3.9+
                        index = node.slice
                    
                    # Create a try-except block
                    try_body = [ast.Return(
                        value=ast.Subscript(
                            value=node.value,
                            slice=node.slice,
                            ctx=node.ctx
                        )
                    )]
                    
                    except_body = [ast.Return(
                        value=ast.Constant(value=None)
                    )]
                    
                    except_handler = ast.ExceptHandler(
                        type=ast.Name(id='IndexError', ctx=ast.Load()),
                        name=None,
                        body=except_body
                    )
                    
                    return ast.Try(
                        body=try_body,
                        handlers=[except_handler],
                        orelse=[],
                        finalbody=[]
                    )
            
            # Apply the transformation - this approach is limited
            # and would need more sophisticated context analysis 
            # to be practically useful
            return source
        except Exception as e:
            # If transformation fails, return original source
            return source
    
    def _apply_string_conversion_fix(self, source: str) -> str:
        """Apply string conversion fix to the source code."""
        try:
            # Parse the source
            tree = ast.parse(source)
            
            # Create a transformer to add string conversions
            class StringConversionTransformer(ast.NodeTransformer):
                def visit_BinOp(self, node):
                    self.generic_visit(node)
                    if isinstance(node.op, ast.Add):
                        # Add str() around non-string operands when string concatenation is likely
                        # This is simplified for demonstration
                        return ast.BinOp(
                            left=ast.Call(
                                func=ast.Name(id='str', ctx=ast.Load()),
                                args=[node.left],
                                keywords=[]
                            ),
                            op=ast.Add(),
                            right=ast.Call(
                                func=ast.Name(id='str', ctx=ast.Load()),
                                args=[node.right],
                                keywords=[]
                            )
                        )
                    return node
            
            # Apply the transformation
            transformer = StringConversionTransformer()
            transformed_tree = transformer.visit(tree)
            
            # Convert back to source
            return ast.unparse(transformed_tree)
        except Exception as e:
            # If transformation fails, return original source
            return source
    
    def _record_fix_success(self, error_sig: str, fix: str) -> None:
        """Record a successful fix in MeTTa."""
        # Add to success records
        if error_sig not in self.success_records:
            self.success_records[error_sig] = {}
        
        if fix not in self.success_records[error_sig]:
            self.success_records[error_sig][fix] = {
                "successes": 0,
                "failures": 0,
                "last_used": 0
            }
        
        self.success_records[error_sig][fix]["successes"] += 1
        self.success_records[error_sig][fix]["last_used"] = time.time()
        
        # Record in MeTTa
        self.metta_space.execute(f"(record-fix-attempt {error_sig} {fix} True)")
    
    def _record_fix_failure(self, error_sig: str, fix: str) -> None:
        """Record a failed fix in MeTTa."""
        # Add to success records
        if error_sig not in self.success_records:
            self.success_records[error_sig] = {}
        
        if fix not in self.success_records[error_sig]:
            self.success_records[error_sig][fix] = {
                "successes": 0,
                "failures": 0,
                "last_used": 0
            }
        
        self.success_records[error_sig][fix]["failures"] += 1
        self.success_records[error_sig][fix]["last_used"] = time.time()
        
        # Record in MeTTa
        self.metta_space.execute(f"(record-fix-attempt {error_sig} {fix} False)")
    
    def get_function_recommendations(self, func_name: str) -> List[Dict]:
        """
        Get improvement recommendations for a function based on execution history
        and MeTTa reasoning.
        """
        if func_name not in self.execution_history:
            return []
        
        # Query MeTTa for recommendations
        recommendations = []
        
        # Check error patterns
        error_sig = f"{func_name}:"
        error_patterns = [pattern for pattern in self.error_patterns if pattern.startswith(error_sig)]
        
        for pattern in error_patterns:
            error_info = self.error_patterns[pattern]
            error_type = pattern.split(":")[-1]
            
            if error_info["count"] >= 3:  # Threshold for recommendation
                # Common error patterns warrant recommendations
                if error_type == "ZeroDivisionError":
                    recommendations.append({
                        "type": "error_pattern",
                        "error": "ZeroDivisionError",
                        "count": error_info["count"],
                        "recommendation": "Add a check for zero before division",
                        "sample_code": "if divisor != 0:\n    result = dividend / divisor\nelse:\n    result = default_value"
                    })
                elif error_type == "IndexError":
                    recommendations.append({
                        "type": "error_pattern",
                        "error": "IndexError",
                        "count": error_info["count"],
                        "recommendation": "Add bounds checking before accessing list elements",
                        "sample_code": "if 0 <= index < len(my_list):\n    value = my_list[index]\nelse:\n    value = default_value"
                    })
                elif error_type == "TypeError" and any("str" in s["message"] for s in error_info["samples"]):
                    recommendations.append({
                        "type": "error_pattern",
                        "error": "TypeError (string related)",
                        "count": error_info["count"],
                        "recommendation": "Ensure consistent types with explicit conversion",
                        "sample_code": "result = str(value1) + str(value2)"
                    })
        
        # Check successful vs failed execution ratio
        if func_name in self.function_metrics:
            metrics = self.function_metrics[func_name]
            failure_rate = metrics["failures"] / max(1, metrics["calls"])
            
            if failure_rate > 0.1 and metrics["calls"] >= 10:
                # High failure rate warrants general robustness recommendation
                recommendations.append({
                    "type": "failure_rate",
                    "rate": f"{failure_rate:.2%}",
                    "calls": metrics["calls"],
                    "recommendation": "Add comprehensive error handling to improve robustness",
                    "sample_code": "try:\n    # Original code\nexcept ExceptionType as e:\n    # Handle specific error\n    logger.error(f\"Error: {e}\")\n    return default_value"
                })
        
        # Add additional MeTTa-derived recommendations
        # This would involve more complex querying of the MeTTa space
        
        return recommendations
    
    def export_knowledge(self, file_path: str) -> None:
        """Export MeTTa knowledge and monitoring data to a file."""
        export_data = {
            "success_records": self.success_records,
            "execution_history": self.execution_history,
            "error_patterns": self.error_patterns,
            "function_metrics": self.function_metrics,
            "timestamp": time.time()
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"Successfully exported knowledge to {file_path}")
        except Exception as e:
            print(f"Failed to export knowledge: {e}")
    
    def import_knowledge(self, file_path: str) -> bool:
        """Import MeTTa knowledge and monitoring data from a file."""
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)
            
            # Validate data structure
            required_keys = ["success_records", "execution_history", 
                           "error_patterns", "function_metrics"]
            
            if not all(key in import_data for key in required_keys):
                print("Invalid knowledge file format")
                return False
            
            # Update internal state
            self.success_records.update(import_data["success_records"])
            self.execution_history.update(import_data["execution_history"])
            self.error_patterns.update(import_data["error_patterns"])
            self.function_metrics.update(import_data["function_metrics"])
            
            # Update MeTTa space with imported knowledge
            self._sync_metta_with_imported_knowledge()
            
            print(f"Successfully imported knowledge from {file_path}")
            return True
            
        except Exception as e:
            print(f"Failed to import knowledge: {e}")
            return False
    
    def _sync_metta_with_imported_knowledge(self) -> None:
        """Sync MeTTa space with imported knowledge."""
        # Add error patterns
        for error_sig, info in self.error_patterns.items():
            self.metta_space.add_atom(f"(error-pattern {error_sig} {info['count']})")
        
        # Add function metrics
        for func_name, metrics in self.function_metrics.items():
            self.metta_space.add_atom(
                f"(function-metrics {func_name} {metrics['calls']} {metrics['successes']} {metrics['failures']} {metrics['avg_time']:.6f})"
            )
        
        # Add success records
        for error_sig, fixes in self.success_records.items():
            for fix, stats in fixes.items():
                self.metta_space.add_atom(
                    f"(success-record {error_sig} {fix} {stats['last_used']} {stats['successes']} {stats['failures']})"
                )


# Create a global instance for easy access
monitor = DynamicMonitor()

# Decorator for convenience
def hybrid_transform(context=None, auto_fix=False, collect_metrics=True):
    """Decorator for monitoring and potentially transforming Python functions."""
    return monitor.hybrid_transform(context, auto_fix, collect_metrics)