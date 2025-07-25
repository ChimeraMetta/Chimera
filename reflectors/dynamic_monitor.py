import functools
import inspect
import traceback
import time
import textwrap
from typing import Any, Dict, List, Callable, Optional

# Import the static analyzer and type conversion function
from reflectors.static_analyzer import decompose_function, convert_python_type_to_metta

# MeTTa integration through hyperon
from hyperon import *


class DynamicMonitor:
    """
    Dynamic runtime monitor for Python functions that integrates with MeTTa.
    Captures execution details and pushes them to MeTTa for reasoning.
    """
    
    def __init__(self, metta_space=None):
        """Initialize the monitor with an optional MeTTa space."""
        self.metta = MeTTa()
        self.metta_space = metta_space or self.metta.space()

        self.E = E
        self.S = S
        self.Atoms = Atoms
        self.AtomType = AtomType
        self.G = G
        self.interpret = interpret

        # Evolutionary components
        self.evolution_callback = None
        self.error_context = {}
    
    def set_evolution_callback(self, callback: Callable):
        """Set callback function to trigger evolution on errors"""
        self.evolution_callback = callback
    
    def _trigger_evolution_on_error(self, func_name: str, error_info: Dict[str, Any]):
        """Trigger evolution when an error occurs"""
        if self.evolution_callback:
            try:
                self.evolution_callback(func_name, error_info)
            except Exception as e:
                print(f"Evolution callback failed: {e}")
    
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
                func_atom = self.metta.parse_single(f"(function {func.__name__})")
                self.metta_space.add_atom(func_atom)
            
            # Add context information if provided
            if context:
                # Use proper atom format with hyphen-separated identifiers
                scope_atoms = context.replace(":", "-").split(".")
                context_expr = " ".join(scope_atoms)
                context_atom = self.metta.parse_single(f"(function-context {func.__name__} {context_expr})")
                self.metta_space.add_atom(context_atom)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create a unique execution ID for this call
                exec_id = f"exec-{func.__name__}-{int(time.time()*1000)}"
                
                # Record call start
                start_time = time.time()
                start_atom = self.metta.parse_single(f"(execution-start {exec_id} {func.__name__} {start_time})")
                self.metta_space.add_atom(start_atom)
                
                # Capture and record input parameters
                try:
                    input_info = self._capture_inputs(func, exec_id, args, kwargs)
                except Exception as e:
                    # If input capture fails, log it but continue
                    error_atom = self.metta.parse_single(f"(execution-input-error {exec_id} \"{str(e)}\")")
                    self.metta_space.add_atom(error_atom)
                
                try:
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Record successful execution
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    # Record success in MeTTa
                    success_atom = self.metta.parse_single(f"(execution-success {exec_id} {execution_time})")
                    self.metta_space.add_atom(success_atom)
                    
                    # Capture and record output value
                    try:
                        self._capture_output(exec_id, result)
                    except Exception as e:
                        # If output capture fails, log it but return the result
                        output_error_atom = self.metta.parse_single(f"(execution-output-error {exec_id} \"{str(e)}\")")
                        self.metta_space.add_atom(output_error_atom)
                    
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
                    error_atom = self.metta.parse_single(f"(execution-error {exec_id} {error_id} {execution_time})")
                    self.metta_space.add_atom(error_atom)
                    
                    error_type_atom = self.metta.parse_single(f"(error-type {error_id} {error_type})")
                    self.metta_space.add_atom(error_type_atom)
                    
                    # Escape error message for MeTTa
                    escaped_msg = error_msg.replace('"', '\\"').replace('\\', '\\\\')
                    error_msg_atom = self.metta.parse_single(f"(error-message {error_id} \"{escaped_msg}\")")
                    self.metta_space.add_atom(error_msg_atom)
                    
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
                    
                        error_context = self._create_error_context(func, e, args)
                    
                    # Store error context
                    self.error_context[func.__name__] = error_context
                    
                    # Trigger evolution if callback is set
                    self._trigger_evolution_on_error(func.__name__, error_context)
                    
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
            param_atom = self.metta.parse_single(f"(input-param {exec_id} {i} {param_name} {metta_type})")
            self.metta_space.add_atom(param_atom)
            
            # Try to capture sample value for debugging (safely)
            try:
                sample = self._safe_str_sample(param_value)
                if sample:
                    escaped_sample = sample.replace('"', '\\"').replace('\\', '\\\\')
                    sample_atom = self.metta.parse_single(f"(input-sample {exec_id} {param_name} \"{escaped_sample}\")")
                    self.metta_space.add_atom(sample_atom)
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
        output_type_atom = self.metta.parse_single(f"(output-type {exec_id} {metta_type})")
        self.metta_space.add_atom(output_type_atom)
        
        # Try to capture sample output value (safely)
        try:
            sample = self._safe_str_sample(result)
            if sample:
                escaped_sample = sample.replace('"', '\\"').replace('\\', '\\\\')
                sample_atom = self.metta.parse_single(f"(output-sample {exec_id} \"{escaped_sample}\")")
                self.metta_space.add_atom(sample_atom)
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
    
    def _create_error_context(self, func: Callable, e: Exception, args: tuple) -> Dict[str, Any]:
        """Creates a dictionary containing detailed context about an error."""
        try:
            # Get and dedent the original function source
            raw_func_source = inspect.getsource(func)
            clean_func_source = textwrap.dedent(raw_func_source)
        except:
            clean_func_source = f"# Source not available for {func.__name__}"
        
        error_context = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'failing_inputs': [args] if args else [],
            'function_name': func.__name__,
            'traceback': traceback.format_exc(),
            'function_source': clean_func_source
        }
        return error_context
    
    def query(self, query_pattern: str) -> List:
        """
        Query MeTTa for information using the interpret function.
        """
        try:
            # Parse the query string into a proper MeTTa expression
            parsed_query = self.metta.parse_single(query_pattern)
            
            # Use interpret function from hyperon
            results = self.interpret(self.metta_space, 
                                    self.E(self.Atoms.METTA, parsed_query,
                                           self.AtomType.UNDEFINED, 
                                           self.G(self.metta_space)))
            return results
        except Exception as e:
            print(f"Error executing query: {query_pattern}")
            print(f"  Error details: {e}")
            
            # Fallback to direct query if interpret fails
            try:
                print("Trying fallback query approach...")
                results = self.metta_space.query(parsed_query)
                return results
            except Exception as e2:
                print(f"  Fallback query failed: {e2}")
                return []
    
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

    def add_atom(self, atom_str: str) -> bool:
        """
        Add a MeTTa atom to the space.
        """
        try:
            # Skip processing for simple rule atoms that don't contain code strings
            if not ('"' in atom_str and any(keyword in atom_str for keyword in ['def ', 'class ', 'import ', 'return '])):
                # Simple atom - parse directly
                parsed_atom = self.metta.parse_single(atom_str)
                self.metta_space.add_atom(parsed_atom)
                return True
            
            # Handle function code atoms specially
            if "= (" in atom_str and '"' in atom_str:
                # Extract the code part
                code_start = atom_str.find('"')
                code_end = atom_str.rfind('"')
                if code_start != -1 and code_end != -1 and code_start < code_end:
                    code = atom_str[code_start+1:code_end]
                    
                    # Escape backslashes and quotes
                    escaped_code = code.replace('\\', '\\\\').replace('"', '\\"')
                    
                    # Preserve newlines
                    escaped_code = escaped_code.replace('\n', '\\n')
                    
                    # Reconstruct the atom string
                    atom_str = atom_str[:code_start+1] + escaped_code + atom_str[code_end:]
            
            parsed_atom = self.metta.parse_single(atom_str)
            self.metta_space.add_atom(parsed_atom)
            return True
            
        except Exception as e:
            # More informative error handling - but suppress the noisy output unless debug mode
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"Error adding atom: {atom_str[:200]}...")  # Show more chars for debugging
                print(f"Error type: {type(e)}")
                print(f"Error details: {str(e)}")
            return False
    
    def load_metta_rules(self, rules_file: str) -> bool:
        """
        Load MeTTa rules from a file by properly parsing the content first.
        
        This follows the Hyperon MeTTa implementation's expected usage pattern.
        """
        try:
            # Open and read the file
            with open(rules_file, 'r') as f:
                file_content = f.read()
            
            # Parse the content to get actual MeTTa atoms
            parsed_atoms = self.metta.parse_all(file_content)
            
            # Add each parsed atom to our space
            atom_count = 0
            for atom in parsed_atoms:
                try:
                    self.metta_space.add_atom(atom)
                    atom_count += 1
                except Exception as atom_err:
                    # Suppress individual atom errors during bulk loading unless debug mode
                    if hasattr(self, 'debug_mode') and self.debug_mode:
                        print(f"Error adding atom: {atom}")
                        print(f"  Error details: {atom_err}")
            
            print(f"Successfully loaded {atom_count}/{len(parsed_atoms)} rules from {rules_file}")
            return atom_count > 0
            
        except Exception as e:
            print(f"Error loading MeTTa rules: {e}")
            
            # Fallback approach using run and load-ascii
            try:
                print("Trying alternate approach with load-ascii...")
                
                # Create a temporary binding for our space
                space_name = f"&rules_space_{int(time.time())}"
                
                # Bind and load
                self.metta.run(f'''
                    !(bind! {space_name} {self.metta_space})
                    !(load-ascii {space_name} "{rules_file}")
                ''')
                
                print(f"Successfully loaded rules using load-ascii approach")
                return True
            except Exception as e2:
                print(f"Error in alternate approach: {e2}")
                return False

# Create a global instance for easy access
monitor = DynamicMonitor()

# Decorator for convenience
def hybrid_transform(context=None, auto_fix=False):
    """Decorator for monitoring Python functions and pushing data to MeTTa."""
    return monitor.hybrid_transform(context, auto_fix)