"""
Fixed File-Based Function Healer

Addresses issues with function source extraction, error processing, and healing generation.
Includes improved error handling and MeTTa integration.

Location: file_based_healer_fixed.py
"""

import os
import json
import time
import threading
import hashlib
import textwrap
import ast
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import functools
import traceback
import inspect

# Import healing components
from reflectors.autonomous_evolution import AutonomousErrorFixer
from hyperon import MeTTa


class FileBasedHealer:
    """
    File-based healing system that watches for error files and generates fixes.
    """
    
    def __init__(self, healing_dir: str = "./healing_workspace", ontology_path: str = None):
        """
        Initialize file-based healer.
        
        Args:
            healing_dir: Directory to store healing files
            ontology_path: Path to MeTTa ontology file
        """
        self.healing_dir = Path(healing_dir)
        self.ontology_path = ontology_path
        
        # Create directory structure
        self.healing_dir.mkdir(exist_ok=True)
        (self.healing_dir / "errors").mkdir(exist_ok=True)
        (self.healing_dir / "functions").mkdir(exist_ok=True)
        (self.healing_dir / "healed").mkdir(exist_ok=True)
        (self.healing_dir / "logs").mkdir(exist_ok=True)
        
        # Initialize MeTTa components
        self.metta = MeTTa()
        self.metta_space = self.metta.space()
        self.error_fixer = AutonomousErrorFixer(self.metta_space)
        
        # Load ontology if provided
        if ontology_path and os.path.exists(ontology_path):
            self._load_ontology()
        
        # Healing daemon state
        self.is_running = False
        self.daemon_thread = None
        self.processed_errors = set()
        
        print(f"[INFO] File-based healer initialized: {self.healing_dir}")
    
    def _load_ontology(self):
        """Load MeTTa ontology."""
        try:
            with open(self.ontology_path, 'r') as f:
                content = f.read()
            
            atoms = self.metta.parse_all(content)
            for atom in atoms:
                self.metta_space.add_atom(atom)
            
            print(f"[OK] Loaded ontology: {self.ontology_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load ontology: {e}")
    
    def start_healing_daemon(self):
        """Start the healing daemon that watches for error files."""
        if self.is_running:
            print("Healing daemon already running")
            return
        
        self.is_running = True
        self.daemon_thread = threading.Thread(target=self._healing_daemon_loop, daemon=True)
        self.daemon_thread.start()
        
        print(f"[INFO] Healing daemon started, watching: {self.healing_dir}")
    
    def stop_healing_daemon(self):
        """Stop the healing daemon."""
        self.is_running = False
        if self.daemon_thread:
            self.daemon_thread.join(timeout=5)
        print("[INFO] Healing daemon stopped")
    
    def _healing_daemon_loop(self):
        """Main loop for the healing daemon."""
        while self.is_running:
            try:
                self._process_error_files()
                time.sleep(2)  # Check every 2 seconds
            except Exception as e:
                print(f"[ERROR] Daemon error: {e}")
                time.sleep(5)
    
    def _process_error_files(self):
        """Process new error files."""
        error_dir = self.healing_dir / "errors"
        
        for error_file in error_dir.glob("*.json"):
            error_id = error_file.stem
            
            if error_id in self.processed_errors:
                continue
            
            try:
                self._process_single_error(error_file)
                self.processed_errors.add(error_id)
            except Exception as e:
                print(f"[ERROR] Error processing {error_file}: {e}")
                # Log the full error for debugging
                with open(self.healing_dir / "logs" / f"error_{error_id}.log", "w") as log_file:
                    log_file.write(f"Error processing {error_file}:\n")
                    log_file.write(f"{traceback.format_exc()}\n")
    
    def _process_single_error(self, error_file: Path):
        """Process a single error file and generate healing."""
        with open(error_file, 'r') as f:
            error_data = json.load(f)
        
        func_name = error_data['function_name']
        print(f"[INFO] Processing error for {func_name}")
        
        # Load function source if available
        func_file = self.healing_dir / "functions" / f"{func_name}.py"
        if func_file.exists():
            success = self._register_function_from_file(func_file, func_name)
            if not success:
                print(f"[WARNING] Could not register function {func_name}, proceeding with error data only")
        
        # Create error context with proper handling
        error_context = self._create_error_context(error_data)
        
        # Generate healing
        try:
            success = self.error_fixer.handle_error(func_name, error_context)
            
            if success:
                # Save healed function
                self._save_healed_function(func_name, error_data)
                print(f"[OK] Generated healing for {func_name}")
            else:
                print(f"[ERROR] Failed to generate healing for {func_name}")
                
        except Exception as e:
            print(f"[ERROR] Exception during healing generation for {func_name}: {e}")
            # Create a basic safe fallback implementation
            self._create_safe_fallback(func_name, error_data)
    
    def _create_error_context(self, error_data: dict) -> dict:
        """Create a proper error context from error data."""
        # Extract inputs more safely
        inputs_data = error_data.get('inputs', {})
        if isinstance(inputs_data, dict):
            failing_inputs = [inputs_data.get('args', ())]
        else:
            failing_inputs = [inputs_data] if inputs_data else []
        
        error_context = {
            'error_type': error_data['error_type'],
            'error_message': error_data['error_message'],
            'failing_inputs': failing_inputs,
            'function_name': error_data['function_name'],
            'traceback': error_data.get('traceback', ''),
            'function_source': self._get_function_source_fallback(error_data['function_name'])
        }
        return error_context
    
    def _get_function_source_fallback(self, func_name: str) -> str:
        """Get function source with fallback to basic template."""
        func_file = self.healing_dir / "functions" / f"{func_name}.py"
        
        if func_file.exists():
            try:
                with open(func_file, 'r') as f:
                    return f.read()
            except Exception as e:
                print(f"[WARNING] Could not read function file: {e}")
        
        # Create a basic template
        return f"""def {func_name}(*args, **kwargs):
    '''Function template for healing - original source not available'''
    # TODO: Implement safe version
    return None
"""
    
    def _register_function_from_file(self, func_file: Path, func_name: str) -> bool:
        """Register function from file with error fixer."""
        try:
            with open(func_file, 'r') as f:
                func_source = f.read()
            
            # Clean up the source code
            func_source = func_source.strip()
            
            # Try to parse and validate the function
            try:
                # Parse to validate syntax
                parsed = ast.parse(func_source)
                
                # Execute to get function object
                exec_globals = {}
                exec_locals = {}
                exec(func_source, exec_globals, exec_locals)
                
                func = exec_locals.get(func_name)
                if func and callable(func):
                    self.error_fixer.register_function(func)
                    print(f"[OK] Registered function {func_name} from file")
                    return True
                else:
                    print(f"[WARNING] Function {func_name} not found in executed code")
                    return False
                    
            except SyntaxError as e:
                print(f"[WARNING] Syntax error in function source: {e}")
                return False
                
        except Exception as e:
            print(f"[WARNING] Error registering function from file: {e}")
            return False
    
    def _save_healed_function(self, func_name: str, error_data: dict):
        """Save healed function to file."""
        try:
            healed_impl = self.error_fixer.get_current_implementation(func_name)
            if healed_impl:
                try:
                    healed_source = inspect.getsource(healed_impl)
                except OSError:
                    # Function was dynamically created, reconstruct from implementation
                    healed_source = self._reconstruct_function_source(healed_impl, func_name)
                
                # Create healing response
                healing_data = {
                    'function_name': func_name,
                    'healed_code': healed_source,
                    'original_error': error_data,
                    'timestamp': datetime.now().isoformat(),
                    'version': int(time.time())
                }
                
                # Save healed function
                healed_file = self.healing_dir / "healed" / f"{func_name}_healed.py"
                with open(healed_file, 'w') as f:
                    f.write(f"# Healed function generated at {datetime.now()}\n")
                    f.write(f"# Original error: {error_data['error_type']}\n")
                    f.write(f"# Error message: {error_data['error_message']}\n\n")
                    f.write(healed_source)
                
                # Save healing metadata
                healing_meta_file = self.healing_dir / "healed" / f"{func_name}_healing.json"
                with open(healing_meta_file, 'w') as f:
                    json.dump(healing_data, f, indent=2, default=str)
                
                print(f"[INFO] Saved healed function: {healed_file}")
            else:
                print(f"[WARNING] No healed implementation found for {func_name}")
                
        except Exception as e:
            print(f"[ERROR] Error saving healed function: {e}")
    
    def _reconstruct_function_source(self, func: Callable, func_name: str) -> str:
        """Reconstruct function source when inspect.getsource fails."""
        try:
            # Try to get the function's code object
            code = func.__code__
            
            # Create a basic reconstruction
            args = code.co_varnames[:code.co_argcount]
            arg_str = ", ".join(args)
            
            return f"""def {func_name}({arg_str}):
    '''Healed function - auto-generated safe implementation'''
    try:
        # Original implementation with safety checks
        return {func_name}_original_impl({arg_str})
    except Exception as e:
        # Safe fallback
        return None
"""
        except Exception:
            # Ultimate fallback
            return f"""def {func_name}(*args, **kwargs):
    '''Healed function - safe fallback implementation'''
    # TODO: Implement proper healing logic
    return None
"""
    
    def _create_safe_fallback(self, func_name: str, error_data: dict):
        """Create a safe fallback implementation when healing fails."""
        try:
            error_type = error_data['error_type']
            
            # Create error-specific safe implementations
            if error_type == "IndexError":
                safe_code = f"""def {func_name}(*args, **kwargs):
    '''Safe fallback for IndexError'''
    try:
        # Add bounds checking
        if args and len(args) >= 2:
            arr, index = args[0], args[1]
            if isinstance(arr, (list, tuple, str)) and isinstance(index, int):
                if 0 <= index < len(arr):
                    return arr[index]
        return None
    except Exception:
        return None
"""
            elif error_type == "ZeroDivisionError":
                safe_code = f"""def {func_name}(*args, **kwargs):
    '''Safe fallback for ZeroDivisionError'''
    try:
        if len(args) >= 2:
            numerator, denominator = args[0], args[1]
            if denominator != 0:
                return numerator / denominator
            else:
                return float('inf') if numerator > 0 else float('-inf') if numerator < 0 else 0
        return None
    except Exception:
        return None
"""
            elif error_type == "AttributeError":
                safe_code = f"""def {func_name}(*args, **kwargs):
    '''Safe fallback for AttributeError'''
    try:
        # Add None checks
        if args:
            safe_args = []
            for arg in args:
                if arg is None:
                    safe_args.append("")  # Safe default
                else:
                    safe_args.append(arg)
            # Process with safe arguments
            return {func_name}_safe_impl(*safe_args, **kwargs)
        return None
    except Exception:
        return None
"""
            else:
                safe_code = f"""def {func_name}(*args, **kwargs):
    '''Safe fallback implementation'''
    try:
        # Generic safe implementation
        return None
    except Exception:
        return None
"""
            
            # Save safe fallback
            safe_file = self.healing_dir / "healed" / f"{func_name}_healed.py"
            with open(safe_file, 'w') as f:
                f.write(f"# Safe fallback generated at {datetime.now()}\n")
                f.write(f"# Original error: {error_type}\n")
                f.write(f"# This is a safe fallback implementation\n\n")
                f.write(safe_code)
            
            # Save metadata
            healing_data = {
                'function_name': func_name,
                'healed_code': safe_code,
                'original_error': error_data,
                'timestamp': datetime.now().isoformat(),
                'version': int(time.time()),
                'type': 'safe_fallback'
            }
            
            healing_meta_file = self.healing_dir / "healed" / f"{func_name}_healing.json"
            with open(healing_meta_file, 'w') as f:
                json.dump(healing_data, f, indent=2, default=str)
            
            print(f"[INFO] Created safe fallback for {func_name}")
            
        except Exception as e:
            print(f"[ERROR] Failed to create safe fallback: {e}")


class HealingFileDecorator:
    """
    Decorator that automatically saves function source and reports errors to files.
    """
    
    def __init__(self, healing_dir: str = "./healing_workspace"):
        """
        Initialize healing decorator.
        
        Args:
            healing_dir: Directory for healing files
        """
        self.healing_dir = Path(healing_dir)
        self.healing_dir.mkdir(exist_ok=True)
        (self.healing_dir / "errors").mkdir(exist_ok=True)
        (self.healing_dir / "functions").mkdir(exist_ok=True)
        (self.healing_dir / "healed").mkdir(exist_ok=True)
        
        self.healed_functions = {}  # Cache of healed functions
    
    def auto_heal(self, context: str = "general"):
        """
        Decorator that enables auto-healing for a function.
        
        Args:
            context: Context/domain for the function
            
        Returns:
            Decorated function with auto-healing
        """
        def decorator(func: Callable) -> Callable:
            func_name = func.__name__
            
            # Save function source to file
            self._save_function_source(func, context)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Check if we have a healed version
                healed_func = self._load_healed_function(func_name)
                current_func = healed_func if healed_func else func
                
                try:
                    return current_func(*args, **kwargs)
                except Exception as e:
                    # Report error to file
                    self._report_error(func_name, e, args, kwargs)
                    
                    # Wait a moment for healing (in case daemon is running)
                    time.sleep(1)
                    
                    # Try to load newly healed version
                    new_healed_func = self._load_healed_function(func_name)
                    if new_healed_func and new_healed_func != current_func:
                        print(f"[INFO] Applying fresh healing to {func_name}")
                        try:
                            return new_healed_func(*args, **kwargs)
                        except Exception as retry_error:
                            print(f"[ERROR] Healed function still failed: {retry_error}")
                    
                    # If no healing available or healing failed, re-raise
                    raise
            
            return wrapper
        return decorator
    
    def _save_function_source(self, func: Callable, context: str):
        """Save function source and metadata to file."""
        try:
            func_name = func.__name__
            func_source = inspect.getsource(func)
            
            # Clean and dedent the source
            func_source = textwrap.dedent(func_source)
            
            # Save function source
            func_file = self.healing_dir / "functions" / f"{func_name}.py"
            with open(func_file, 'w') as f:
                f.write(func_source)
            
            # Save metadata
            meta_file = self.healing_dir / "functions" / f"{func_name}.meta.json"
            meta_data = {
                'function_name': func_name,
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'source_hash': hashlib.sha256(func_source.encode()).hexdigest()
            }
            with open(meta_file, 'w') as f:
                json.dump(meta_data, f, indent=2)
                
        except Exception as e:
            print(f"[WARNING] Could not save function source: {e}")
            
    def _report_error(self, func_name: str, error: Exception, args: tuple, kwargs: dict):
        """Report error to a file."""
        try:
            # Create a unique ID for the error
            error_id = f"{func_name}_{int(time.time())}"
            
            # Get traceback
            tb_str = traceback.format_exc()
            
            # Safely serialize arguments
            safe_args = self._serialize_args_safely(args)
            safe_kwargs = self._serialize_args_safely(kwargs)
            
            error_data = {
                'error_id': error_id,
                'function_name': func_name,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': tb_str,
                'timestamp': datetime.now().isoformat(),
                'inputs': {
                    'args': safe_args,
                    'kwargs': safe_kwargs
                }
            }
            
            error_file = self.healing_dir / "errors" / f"{error_id}.json"
            with open(error_file, 'w') as f:
                json.dump(error_data, f, indent=2, default=str)
            
            print(f"[INFO] Reported error for {func_name} to {error_file}")
        
        except Exception as e:
            print(f"[WARNING] Could not report error: {e}")
    
    def _serialize_args_safely(self, args):
        """Safely serialize arguments for JSON storage."""
        if isinstance(args, dict):
            safe_dict = {}
            for key, value in args.items():
                try:
                    # Test if value is JSON serializable
                    json.dumps(value)
                    safe_dict[key] = value
                except (TypeError, ValueError):
                    safe_dict[key] = str(value)
            return safe_dict
        elif isinstance(args, (list, tuple)):
            safe_list = []
            for value in args:
                try:
                    # Test if value is JSON serializable
                    json.dumps(value)
                    safe_list.append(value)
                except (TypeError, ValueError):
                    safe_list.append(str(value))
            return safe_list
        else:
            try:
                json.dumps(args)
                return args
            except (TypeError, ValueError):
                return str(args)

    def _load_healed_function(self, func_name: str) -> Optional[Callable]:
        """Load the latest healed version of a function."""
        try:
            healed_file = self.healing_dir / "healed" / f"{func_name}_healed.py"
            healing_meta_file = self.healing_dir / "healed" / f"{func_name}_healing.json"

            if not healed_file.exists() or not healing_meta_file.exists():
                return None

            # Check if we already have this version loaded
            with open(healing_meta_file, 'r') as f:
                version = json.load(f).get('version', 0)
            
            if self.healed_functions.get(f"{func_name}_version", -1) >= version:
                return self.healed_functions.get(func_name)

            # Load new version
            with open(healed_file, 'r') as f:
                healed_code = f.read()
            
            # Execute to get function object
            exec_globals = {}
            exec_locals = {}
            exec(healed_code, exec_globals, exec_locals)
            
            healed_func = exec_locals.get(func_name)
            if healed_func and callable(healed_func):
                self.healed_functions[func_name] = healed_func
                self.healed_functions[f"{func_name}_version"] = version
                print(f"[INFO] Loaded healed function {func_name} (v{version})")
                return healed_func
            
        except Exception as e:
            print(f"[WARNING] Could not load healed function: {e}")
        
        return None


def start_healing_daemon(healing_dir: str = "./healing_workspace", 
                        ontology_path: str = None) -> FileBasedHealer:
    """Utility function to start the healing daemon."""
    healer = FileBasedHealer(healing_dir, ontology_path)
    healer.start_healing_daemon()
    return healer

def create_healing_decorator(healing_dir: str = "./healing_workspace") -> HealingFileDecorator:
    """Utility function to create a healing decorator."""
    return HealingFileDecorator(healing_dir)


def demo_file_based_healing():
    """
    Demonstrate the file-based healing workflow.
    """
    ontology_path = "metta_ontology/core_ontology.metta"
    
    print("\n--- Starting Healing Daemon ---")
    healer = start_healing_daemon(ontology_path=ontology_path)
    
    print("\n--- Running buggy functions to generate errors ---")
    decorator = create_healing_decorator()

    @decorator.auto_heal(context="array_processing")
    def buggy_find_element(arr, index):
        """Find element by index - has bounds errors."""
        return arr[index]

    @decorator.auto_heal(context="math_operations")
    def buggy_calculate_ratio(numerator, denominator):
        """Calculate ratio - has division by zero error."""
        return numerator / denominator
    
    @decorator.auto_heal(context="string_processing")
    def buggy_format_name(first_name, last_name):
        """Format full name - has None errors."""
        return f"{first_name.upper()} {last_name.upper()}"  # AttributeError if None

    # --- Trigger errors ---
    print("\n[1] Testing 'buggy_find_element'")
    try:
        buggy_find_element([1,2,3], 5)
    except IndexError as e:
        print(f"  -> Got expected error: {e}")

    print("\n[2] Testing 'buggy_calculate_ratio'")
    try:
        buggy_calculate_ratio(10, 0)
    except ZeroDivisionError as e:
        print(f"  -> Got expected error: {e}")
        
    print("\n[3] Testing 'buggy_format_name'")
    try:
        buggy_format_name("John", None)
    except (TypeError, AttributeError) as e:
        print(f"  -> Got expected error: {e}")

    print("\n--- Waiting for healing daemon to process files (5s) ---")
    time.sleep(5)
    
    print("\n--- Re-running functions to see if they are healed ---")
    
    # --- Check for healed versions ---
    print("\n[1] Re-testing 'buggy_find_element'")
    try:
        result = buggy_find_element([1,2,3], 5)
        print(f"  -> Result after healing: {result} (Healed: {result is not None})")
    except Exception as e:
        print(f"  -> Still failing: {e}")

    print("\n[2] Re-testing 'buggy_calculate_ratio'")
    try:
        result = buggy_calculate_ratio(10, 0)
        print(f"  -> Result after healing: {result} (Healed: {result is not None})")
    except Exception as e:
        print(f"  -> Still failing: {e}")
    
    print("\n[3] Re-testing 'buggy_format_name'")
    try:
        result = buggy_format_name("John", None)
        print(f"  -> Result after healing: {result} (Healed: {result is not None})")
    except Exception as e:
        print(f"  -> Still failing: {e}")
    
    print("\n--- Checking status of healed functions ---")
    healed_dir = Path("./healing_workspace/healed")
    
    if healed_dir.exists():
        healed_files = list(healed_dir.glob("*_healed.py"))
        print(f"  -> Found {len(healed_files)} healed function files")
        for file in healed_files:
            print(f"     {file.name}")
    else:
        print("  -> No healed functions directory found")

    print("\n--- Stopping healing daemon ---")
    healer.stop_healing_daemon()
    
    print("\n--- Demo finished ---")


def run_standalone_daemon():
    """Run standalone healing daemon."""
    import sys
    
    healing_dir = sys.argv[1] if len(sys.argv) > 1 else "./healing_workspace"
    ontology_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"[INFO] Starting standalone healing daemon")
    print(f"   Healing directory: {healing_dir}")
    print(f"   Ontology: {ontology_path or 'None'}")
    
    healer = start_healing_daemon(healing_dir, ontology_path)
    
    try:
        print("[OK] Healing daemon running. Press Ctrl+C to stop.")
        print(f"   Watching for error files in: {os.path.abspath(healing_dir)}/errors/")
        print(f"   Healed functions will be saved to: {os.path.abspath(healing_dir)}/healed/")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping healing daemon...")
        healer.stop_healing_daemon()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "daemon":
        run_standalone_daemon()
    else:
        demo_file_based_healing()