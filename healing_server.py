"""
Self-Healing Function Server

A lightweight server that provides automatic error healing for Python functions.
Functions can register with the server and automatically receive fixes when they fail.
The server runs in the background and provides healing services via HTTP API.

Location: healing_server.py (root directory)
"""

import os
import sys
import json
import time
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

# HTTP server imports
import http.server
import socketserver
from urllib.parse import urlparse
import urllib.request
import urllib.parse

# Import the autonomous evolution components
from reflectors.autonomous_evolution import AutonomousErrorFixer
from hyperon import MeTTa


@dataclass
class FunctionRegistration:
    """Registration information for a function."""
    name: str
    source_code: str
    module_path: str
    context: str
    registration_time: datetime
    last_error_time: Optional[datetime] = None
    error_count: int = 0
    fix_count: int = 0
    current_version: int = 1
    is_healed: bool = False


@dataclass
class ErrorReport:
    """Error report from a client function."""
    function_name: str
    error_type: str
    error_message: str
    inputs: List[Any]
    traceback: str
    timestamp: datetime
    client_id: str


@dataclass
class HealingResponse:
    """Response containing healed function code."""
    function_name: str
    healed_code: str
    version: int
    confidence: float
    healing_strategy: str
    test_results: Dict[str, Any]
    timestamp: datetime


class SelfHealingServer:
    """
    Server that provides automatic function healing services.
    Functions can register and receive fixes when they encounter errors.
    """
    
    def __init__(self, port: int = 8765, ontology_path: str = None):
        """
        Initialize the healing server.
        
        Args:
            port: Port to run the server on
            ontology_path: Path to MeTTa ontology file
        """
        self.port = port
        self.ontology_path = ontology_path
        
        # Initialize MeTTa and healing components
        self.metta = MeTTa()
        self.metta_space = self.metta.space()
        self.error_fixer = AutonomousErrorFixer(self.metta_space)
        
        # Load ontology if provided
        if ontology_path and os.path.exists(ontology_path):
            self._load_ontology(ontology_path)
        
        # Server state
        self.registered_functions: Dict[str, FunctionRegistration] = {}
        self.error_history: List[ErrorReport] = []
        self.healing_history: List[HealingResponse] = []
        self.active_clients: Dict[str, datetime] = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.server_thread = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            'total_registrations': 0,
            'total_errors': 0,
            'total_healings': 0,
            'successful_healings': 0,
            'active_functions': 0,
            'start_time': time.time()
        }
        
        print(f"[INFO] Self-Healing Function Server initialized on port {port}")
    
    def _load_ontology(self, ontology_path: str):
        """Load MeTTa ontology for improved healing."""
        try:
            with open(ontology_path, 'r') as f:
                ontology_content = f.read()
            
            parsed_atoms = self.metta.parse_all(ontology_content)
            for atom in parsed_atoms:
                self.metta_space.add_atom(atom)
            
            print(f"[OK] Loaded ontology: {ontology_path} ({len(parsed_atoms)} atoms)")
        except Exception as e:
            print(f"[WARNING] Failed to load ontology: {e}")
    
    def start_server(self):
        """Start the healing server in a background thread."""
        if self.is_running:
            print("Server is already running")
            return
        
        self.is_running = True
        self.server_thread = threading.Thread(target=self._run_http_server, daemon=True)
        self.server_thread.start()
        
        print(f"[INFO] Self-Healing Server started on http://localhost:{self.port}")
        print(f"[INFO] API endpoints:")
        print(f"  POST /register - Register a function for healing")
        print(f"  POST /report_error - Report an error and request healing")
        print(f"  GET /status - Get server status and statistics")
        print(f"  GET /functions - List registered functions")
        print(f"  GET /health - Health check")
    
    def stop_server(self):
        """Stop the healing server."""
        self.is_running = False
        if self.server_thread:
            self.server_thread.join(timeout=5)
        print("[INFO] Self-Healing Server stopped")
    
    def _run_http_server(self):
        """Run the HTTP server."""
        handler = self._create_request_handler()
        
        try:
            with socketserver.TCPServer(("", self.port), handler) as httpd:
                httpd.serve_forever()
        except Exception as e:
            print(f"[ERROR] Server error: {e}")
            self.is_running = False
    
    def _create_request_handler(self):
        """Create HTTP request handler class."""
        server_instance = self
        
        class HealingRequestHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                """Suppress default logging."""
                pass
            
            def do_POST(self):
                """Handle POST requests."""
                try:
                    # Parse URL and content
                    url_parts = urlparse(self.path)
                    content_length = int(self.headers.get('Content-Length', 0))
                    post_data = self.rfile.read(content_length).decode('utf-8')
                    
                    if url_parts.path == '/register':
                        response = server_instance._handle_register(post_data)
                    elif url_parts.path == '/report_error':
                        response = server_instance._handle_error_report(post_data)
                    else:
                        response = {'error': 'Unknown endpoint'}
                        self._send_json_response(404, response)
                        return
                    
                    self._send_json_response(200, response)
                    
                except Exception as e:
                    error_response = {'error': str(e), 'traceback': traceback.format_exc()}
                    self._send_json_response(500, error_response)
            
            def do_GET(self):
                """Handle GET requests."""
                try:
                    url_parts = urlparse(self.path)
                    
                    if url_parts.path == '/status':
                        response = server_instance._get_status()
                    elif url_parts.path == '/functions':
                        response = server_instance._get_functions()
                    elif url_parts.path == '/health':
                        response = {'status': 'healthy', 'timestamp': datetime.now().isoformat()}
                    else:
                        response = {'error': 'Unknown endpoint'}
                        self._send_json_response(404, response)
                        return
                    
                    self._send_json_response(200, response)
                    
                except Exception as e:
                    error_response = {'error': str(e)}
                    self._send_json_response(500, error_response)
            
            def _send_json_response(self, status_code: int, data: dict):
                """Send JSON response."""
                json_data = json.dumps(data, indent=2, default=str)
                
                self.send_response(status_code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(json_data)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json_data.encode('utf-8'))
        
        return HealingRequestHandler
    
    def _handle_register(self, post_data: str) -> dict:
        """Handle function registration."""
        try:
            data = json.loads(post_data)
            
            registration = FunctionRegistration(
                name=data['name'],
                source_code=data['source_code'],
                module_path=data.get('module_path', ''),
                context=data.get('context', 'general'),
                registration_time=datetime.now()
            )
            
            # Use thread pool to register function source
            self.executor.submit(self._register_function_source, registration)
            
            self.registered_functions[data['name']] = registration
            self.stats['total_registrations'] += 1
            self.stats['active_functions'] = len(self.registered_functions)
            
            print(f"[INFO] Registered function: {data['name']}")
            return {'status': 'success', 'message': f"Function {data['name']} registered"}
            
        except Exception as e:
            print(f"[ERROR] Registration failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _register_function_source(self, registration: FunctionRegistration):
        """Register function source code with the error fixer."""
        try:
            exec_globals = {}
            exec_locals = {}
            exec(registration.source_code, exec_globals, exec_locals)
            
            func = exec_locals.get(registration.name)
            if func and callable(func):
                self.error_fixer.register_function(func)
                print(f"[INFO] Source code for {registration.name} registered with fixer.")
        except Exception as e:
            print(f"[ERROR] Could not register source for {registration.name}: {e}")
    
    def _handle_error_report(self, post_data: str) -> dict:
        """Handle error report and generate healing response."""
        try:
            data = json.loads(post_data)
            
            error_report = ErrorReport(
                function_name=data['function_name'],
                error_type=data['error_type'],
                error_message=data['error_message'],
                inputs=data.get('inputs', []),
                traceback=data.get('traceback', ''),
                timestamp=datetime.now(),
                client_id=data.get('client_id', 'unknown')
            )
            
            # Store error report
            self.error_history.append(error_report)
            self.stats['total_errors'] += 1
            
            # Update function registration
            if error_report.function_name in self.registered_functions:
                reg = self.registered_functions[error_report.function_name]
                reg.last_error_time = error_report.timestamp
                reg.error_count += 1
            
            print(f"[INFO] Received error report for {error_report.function_name} from {error_report.client_id}")
            
            # Asynchronously generate healing
            future = self.executor.submit(self._generate_healing, error_report)
            
            if future:
                healing_response = future.result()
                if healing_response:
                    self.healing_history.append(healing_response)
                    self.stats['total_healings'] += 1
                    if healing_response.confidence > 0.7:
                        self.stats['successful_healings'] += 1
                    
                    # Update function registration
                    if error_report.function_name in self.registered_functions:
                        reg = self.registered_functions[error_report.function_name]
                        reg.fix_count += 1
                        reg.current_version += 1
                        reg.is_healed = True
                    
                    return asdict(healing_response)
                else:
                    return {'status': 'received', 'message': 'Healing in progress...'}
            else:
                return {'status': 'error', 'message': 'Healing in progress...'}
                
        except Exception as e:
            print(f"[ERROR] Error handling failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _generate_healing(self, error_report: ErrorReport) -> Optional[HealingResponse]:
        """Generate healing response for an error."""
        try:
            # Create error context for the fixer
            error_context = {
                'error_type': error_report.error_type,
                'error_message': error_report.error_message,
                'failing_inputs': [error_report.inputs] if error_report.inputs else [],
                'function_name': error_report.function_name,
                'traceback': error_report.traceback
            }
            
            # Use autonomous error fixer to generate a fix
            success = self.error_fixer.handle_error(error_report.function_name, error_context)
            
            if success:
                healed_impl = self.error_fixer.get_current_implementation(error_report.function_name)
                
                if healed_impl:
                    try:
                        # Get healed source and create response
                        healed_code = inspect.getsource(healed_impl)
                        
                        # Test the healed function
                        test_results = self._test_healed_function(healed_impl, error_report)
                        
                        healing_response = HealingResponse(
                            function_name=error_report.function_name,
                            healed_code=healed_code,
                            version=self.registered_functions[error_report.function_name].current_version + 1,
                            confidence=test_results.get('confidence', 0.8),
                            healing_strategy=getattr(healed_impl, '_healing_strategy', 'unknown'),
                            test_results=test_results,
                            timestamp=datetime.now()
                        )
                        
                        # Update server state
                        self.healing_history.append(healing_response)
                        self.registered_functions[error_report.function_name].is_healed = True
                        self.registered_functions[error_report.function_name].fix_count += 1
                        self.registered_functions[error_report.function_name].current_version += 1
                        self.stats['total_healings'] += 1
                        if test_results.get('passed'):
                            self.stats['successful_healings'] += 1

                        print(f"[INFO] Generated healing for {error_report.function_name} with confidence {healing_response.confidence:.2f}")
                        return healing_response
                        
                    except Exception as e:
                        print(f"[ERROR] Error generating healing response for {error_report.function_name}: {e}")
            else:
                print(f"[INFO] Could not generate a fix for {error_report.function_name}")
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Healing generation error: {e}")
            return None
    
    def _test_healed_function(self, healed_func: Callable, error_report: ErrorReport) -> dict:
        """Test the healed function to estimate confidence."""
        test_results = {
            'confidence': 0.5,
            'strategy': 'autonomous_generation',
            'tests_passed': 0,
            'tests_total': 0,
            'error_fixed': False
        }
        
        try:
            # Test 1: Can it handle the original failing input?
            if error_report.inputs:
                try:
                    result = healed_func(*error_report.inputs)
                    test_results['error_fixed'] = True
                    test_results['tests_passed'] += 1
                except:
                    test_results['error_fixed'] = False
                test_results['tests_total'] += 1
            
            # Test 2: Basic robustness tests
            robustness_tests = [
                ([], "empty_list_test"),
                ([None], "none_input_test"),
                ([""], "empty_string_test")
            ]
            
            for test_input, test_name in robustness_tests:
                try:
                    if len(test_input) <= len(error_report.inputs):
                        healed_func(*test_input)
                        test_results['tests_passed'] += 1
                except:
                    pass  # Test failure is okay for robustness
                test_results['tests_total'] += 1
            
            # Calculate confidence
            if test_results['tests_total'] > 0:
                pass_rate = test_results['tests_passed'] / test_results['tests_total']
                error_fix_bonus = 0.3 if test_results['error_fixed'] else 0
                test_results['confidence'] = min(0.9, pass_rate + error_fix_bonus)
            
        except Exception as e:
            print(f"[WARNING] Testing error: {e}")
        
        return test_results
    
    def _get_status(self) -> dict:
        """Get server status and statistics."""
        status = {
            'server_status': 'running' if self.is_running else 'stopped',
            'active_clients': len(self.active_clients),
            'registered_functions_count': len(self.registered_functions),
            'uptime_seconds': int(time.time() - self.stats['start_time'])
        }
        status.update(self.stats)
        return status
    
    def _get_functions(self) -> dict:
        """Get list of registered functions."""
        return {
            name: asdict(reg) for name, reg in self.registered_functions.items()
        }


class HealingClient:
    """
    Client that can register functions with the healing server and automatically
    apply fixes when errors occur.
    """
    
    def __init__(self, server_url: str = "http://localhost:8765", client_id: str = None):
        """
        Initialize healing client.
        
        Args:
            server_url: URL of the healing server
            client_id: Unique client identifier
        """
        self.server_url = server_url.rstrip('/')
        self.client_id = client_id or f"client_{os.getpid()}_{int(time.time())}"
        self.healed_functions: Dict[str, Callable] = {}
        self.function_versions: Dict[str, int] = {}
        self.local_cache: Dict[str, dict] = {}

        print(f"[INFO] HealingClient initialized for server: {self.server_url}")
        print(f"     Client ID: {self.client_id}")
    
    def register_function(self, func: Callable, context: str = "general") -> bool:
        """
        Register a function with the healing server.
        
        Args:
            func: Function to register
            context: Context/domain for the function
            
        Returns:
            True if registration successful
        """
        try:
            import inspect
            source_code = inspect.getsource(func)
            module_path = inspect.getmodule(func).__file__ if inspect.getmodule(func) else ""
            
            payload = {
                'name': func.__name__,
                'source_code': source_code,
                'module_path': module_path,
                'context': context,
                'client_id': self.client_id
            }
            
            response = self._make_request('POST', '/register', data=payload)
            
            if response and response.get('status') == 'success':
                print(f"[OK] Function '{func.__name__}' registered successfully.")
                self.function_versions[func.__name__] = 1
                return True
            else:
                print(f"[ERROR] Failed to register '{func.__name__}': {response.get('message')}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error registering function: {e}")
            return False
    
    def healing_wrapper(self, context: str = "general"):
        """
        Decorator that registers a function and provides automatic healing.
        
        Args:
            context: Context/domain for the function
            
        Returns:
            Decorated function with auto-healing capabilities
        """
        def decorator(func: Callable) -> Callable:
            # Register the function
            self.register_function(func, context)
            
            def wrapped_function(*args, **kwargs):
                func_name = func.__name__
                
                # Try current implementation (original or healed)
                current_func = self.healed_functions.get(func_name, func)
                
                try:
                    return current_func(*args, **kwargs)
                    
                except Exception as e:
                    # Report error and request healing
                    print(f"[INFO] Error in {func_name}: {type(e).__name__}: {e}")
                    
                    healing_response = self._request_healing(func_name, e, args, kwargs)
                    
                    if healing_response:
                        # Apply the healing
                        healed_func = self._apply_healing(healing_response)
                        
                        if healed_func:
                            print(f"[INFO] Applying healing to {func_name}, retrying...")
                            try:
                                return healed_func(*args, **kwargs)
                            except Exception as retry_e:
                                print(f"[ERROR] Healed function still failed: {retry_e}")
                    
                    # If healing failed or wasn't available, re-raise original error
                    raise e
            
            return wrapped_function
        return decorator
    
    def _request_healing(self, func_name: str, error: Exception, args: tuple, kwargs: dict) -> Optional[dict]:
        """Request healing for a function error."""
        try:
            payload = {
                'function_name': func_name,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'inputs': { 'args': list(args), 'kwargs': kwargs },
                'traceback': traceback.format_exc(),
                'client_id': self.client_id
            }
            
            print(f"[INFO] Reporting error for '{func_name}' and requesting healing...")
            response = self._make_request('POST', '/report_error', data=payload)
            
            if response and 'healed_code' in response:
                print(f"[INFO] Received healing response for {func_name}")
                return response
            else:
                print(f"[ERROR] Healing request failed: {response}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Could not request healing: {e}")
            return None
    
    def _apply_healing(self, healing_response: dict) -> Optional[Callable]:
        """Apply healing response to create healed function."""
        try:
            func_name = healing_response['function_name']
            healed_code = healing_response['healed_code']
            confidence = healing_response.get('confidence', 0.0)
            
            print(f"[INFO] Applying healing to {func_name} (confidence: {confidence:.1%})")
            
            # Execute the healed code to get the function object
            exec(healed_code, exec_globals, exec_locals)
            healed_func = exec_locals.get(func_name)
            
            if healed_func and callable(healed_func):
                # Store the new healed function and its version
                self.healed_functions[func_name] = healed_func
                self.function_versions[func_name] = healing_response['version']
                print(f"[OK] Successfully applied healing for '{func_name}' (v{healing_response['version']})")
                return healed_func
            else:
                print(f"[ERROR] Could not find function '{func_name}' in healed code.")
                return None
                
        except Exception as e:
            print(f"[ERROR] Error applying healing: {e}")
            return None
    
    def _make_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make HTTP request to healing server."""
        url = f"{self.server_url}{endpoint}"
        
        try:
            if method == 'POST':
                json_data = json.dumps(data, default=str).encode('utf-8')
                req = urllib.request.Request(url, data=json_data, 
                                           headers={'Content-Type': 'application/json'})
            else:
                req = urllib.request.Request(url)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode('utf-8'))
                
        except Exception as e:
            return {'error': f'Request failed: {e}'}
    
    def get_server_status(self) -> dict:
        """Get healing server status."""
        return self._make_request('GET', '/status')
    
    def get_registered_functions(self) -> dict:
        """Get list of functions registered on server."""
        return self._make_request('GET', '/functions')


# Server management utilities
def start_healing_server(port: int = 8765, ontology_path: str = None, 
                        background: bool = True) -> SelfHealingServer:
    """
    Start a healing server.
    
    Args:
        port: Port to run server on
        ontology_path: Path to MeTTa ontology file
        background: Whether to run in background
        
    Returns:
        SelfHealingServer instance
    """
    server = SelfHealingServer(port, ontology_path)
    server.start_server()
    
    if not background:
        try:
            print("Press Ctrl+C to stop the server...")
            while server.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping server...")
            server.stop_server()
    
    return server


# Convenience functions for easy integration
def create_healing_client(server_url: str = "http://localhost:8765") -> HealingClient:
    """Create a healing client with default settings."""
    return HealingClient(server_url)


@contextmanager
def healing_server_context(port: int = 8765, ontology_path: str = None):
    """Context manager for running healing server temporarily."""
    server = start_healing_server(port, ontology_path, background=True)
    try:
        yield server
    finally:
        server.stop_server()


# Example usage and demonstration
def demo_healing_server():
    """Demonstrate the full client-server healing workflow."""
    
    print("[INFO] Self-Healing Server Demo")
    print("=" * 40)
    
    ontology_file = "metta_ontology/core_ontology.metta"
    
    # Use context manager for easy server start/stop
    with healing_server_context(ontology_path=ontology_file):
        
        print("[INFO] Server is running in the background.")
        client = create_healing_client()
        
        # Define buggy functions to be healed by the server
        @client.healing_wrapper(context="array_processing")
        def buggy_find_max(numbers, start, end):
            """Find max in a sub-array - has multiple errors."""
            sub_array = numbers[start:end]
            return max(sub_array) if sub_array else None
            
        @client.healing_wrapper(context="math_operations")
        def buggy_divide(a, b):
            """Divide two numbers - can have ZeroDivisionError."""
            return a / b
            
        @client.healing_wrapper(context="string_processing")
        def buggy_process_text(text, prefix):
            """Process text - can fail on None input."""
            return f"{prefix}: {text.upper()}"
        
        # Test cases to trigger errors and healing
        test_cases = [
            ("Find max with invalid slice", buggy_find_max, ([1, 2, 3], 5, 10)),
            ("Find max with empty slice", buggy_find_max, ([], 0, 0)),
            ("Divide by zero", buggy_divide, (10, 0)),
            ("Process None text", buggy_process_text, (None, "INFO")),
        ]
        
        print(f"[INFO] Testing {len(test_cases)} scenarios...")
        successful_healings = 0
        
        for i, (desc, func, args) in enumerate(test_cases, 1):
            print(f"[INFO] --- Test {i}: {desc} ---")
            
            try:
                result = func(*args)
                print(f"  [OK] Initial call succeeded. Result: {result}")
            except Exception as e:
                print(f"  [INFO] Initial call failed as expected: {type(e).__name__}")
                
                # After the wrapper handles it, the function might be healed
                # Let's try calling it again
                print("[INFO] Retrying function call to see if healing was applied...")
                try:
                    result = func(*args)
                    print(f"  [OK] Success after healing! Result: {result}")
                    successful_healings += 1
                except Exception as retry_e:
                    print(f"  [ERROR] Still failed after healing attempt: {retry_e}")

        print("[INFO] Demo finished.")
        print(f"  Successful healings: {successful_healings} / {len(test_cases)}")
        
        # Print server stats
        status = client.get_server_status()
        print("[INFO] Final Server Status:")
        print(json.dumps(status, indent=2, default=str))


# --- Standalone server execution ---
def run_standalone_server():
    """Run the server as a standalone process."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-Healing Function Server")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--ontology", type=str, default=None, help="Path to MeTTa ontology")
    args = parser.parse_args()
    
    server = start_healing_server(args.port, args.ontology, background=False)
    
    try:
        server._run_http_server()
    except KeyboardInterrupt:
        print("[INFO] Shutting down server...")
        server.stop_server()


if __name__ == "__main__":
    # Check for command line arguments to run as standalone server
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        print("[INFO] Starting in standalone server mode...")
        run_standalone_server()
    else:
        # Run demo by default
        demo_healing_server()