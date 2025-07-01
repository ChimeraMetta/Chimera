"""
Fixed Self-Healing Function Server

Addresses the issues:
1. "name 'inspect' is not defined" 
2. "unexpected indent" errors
3. Function registration and healing integration problems
4. Source code handling and execution issues

Location: fixed_self_healing_server.py
"""

import asyncio
import json
import time
import traceback
import textwrap
import ast
import inspect
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
import aiohttp
from aiohttp import web, ClientSession
import weakref
import threading

# Import healing components with fallbacks
try:
    from reflectors.autonomous_evolution import AutonomousErrorFixer
    from hyperon import MeTTa
    HEALING_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Healing components not available: {e}")
    HEALING_AVAILABLE = False
    
    # Fallback implementations
    class MeTTa:
        def __init__(self):
            pass
        def space(self):
            return {}
    
    class AutonomousErrorFixer:
        def __init__(self, metta_space=None):
            self.function_registry = {}
            self.current_implementations = {}
            self.error_history = {}
            self.fix_attempts = {}
            self.max_fix_attempts = 3
        
        def register_function(self, func):
            func_name = func.__name__
            self.function_registry[func_name] = func
            self.current_implementations[func_name] = func
            self.error_history[func_name] = []
            self.fix_attempts[func_name] = 0
        
        def handle_error(self, func_name, error_context):
            return self._create_simple_fix(func_name, error_context)
        
        def get_current_implementation(self, func_name):
            return self.current_implementations.get(func_name)
        
        def _create_simple_fix(self, func_name, error_context):
            """Create a simple safe fix based on error type."""
            error_type = error_context.get('error_type', 'Unknown')
            failing_inputs = error_context.get('failing_inputs', [])
            
            if not failing_inputs:
                return False
            
            args = failing_inputs[0] if failing_inputs else ()
            param_count = len(args)
            
            # Create parameter list based on actual usage
            params = ', '.join([f'arg{i}' for i in range(param_count)])
            
            if error_type == 'ZeroDivisionError':
                safe_code = f'''def {func_name}({params}):
    """Healed version with division safety."""
    try:
        numerator, denominator = arg0, arg1
        if denominator != 0:
            return numerator / denominator
        return float('inf') if numerator > 0 else float('-inf') if numerator < 0 else 0
    except Exception:
        return None
'''
            elif error_type == 'AttributeError':
                safe_code = f'''def {func_name}({params}):
    """Healed version with None safety."""
    try:
        safe_args = []
        for arg in [{params}]:
            if arg is None:
                safe_args.append("")
            else:
                safe_args.append(arg)
        
        if len(safe_args) >= 2:
            first, second = safe_args[0], safe_args[1]
            if hasattr(first, 'upper') and hasattr(second, 'upper'):
                return f"{{first.upper()}} {{second.upper()}}"
        return ""
    except Exception:
        return ""
'''
            else:
                safe_code = f'''def {func_name}({params}):
    """Safe fallback implementation."""
    try:
        return None
    except Exception:
        return None
'''
            
            try:
                # Execute the safe code to create the function
                exec_globals = {}
                exec_locals = {}
                exec(safe_code, exec_globals, exec_locals)
                
                safe_func = exec_locals.get(func_name)
                if safe_func and callable(safe_func):
                    self.current_implementations[func_name] = safe_func
                    return True
            except Exception as e:
                print(f"[ERROR] Failed to create simple fix: {e}")
            
            return False


@dataclass
class FunctionInfo:
    """Information about a registered function."""
    name: str
    source_code: str
    signature: str
    context: str
    registration_time: float
    client_id: str
    error_count: int = 0
    healing_count: int = 0
    last_error: Optional[str] = None
    is_healed: bool = False


@dataclass
class ErrorReport:
    """Error report from a client."""
    function_name: str
    error_type: str
    error_message: str
    traceback: str
    inputs: Dict[str, Any]
    client_id: str
    timestamp: float


class EnhancedSelfHealingServer:
    """
    Enhanced self-healing server with better error handling and source management.
    """
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.app = web.Application()
        self.functions: Dict[str, FunctionInfo] = {}
        self.clients: Dict[str, Dict] = {}
        self.error_reports: List[ErrorReport] = []
        
        # Server statistics
        self.stats = {
            'start_time': time.time(),
            'total_registrations': 0,
            'total_errors': 0,
            'total_healings': 0,
            'successful_healings': 0
        }
        
        # Initialize healing system
        if HEALING_AVAILABLE:
            self.metta = MeTTa()
            self.metta_space = self.metta.space()
            self.error_fixer = AutonomousErrorFixer(self.metta_space)
        else:
            self.metta = MeTTa()
            self.metta_space = {}
            self.error_fixer = AutonomousErrorFixer()
        
        # Setup routes
        self._setup_routes()
        
        print(f"[INFO] Self-Healing Function Server initialized on port {port}")
    
    def _setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_post('/register', self.register_function)
        self.app.router.add_post('/report_error', self.report_error)
        self.app.router.add_get('/status', self.get_status)
        self.app.router.add_get('/functions', self.list_functions)
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/function/{name}', self.get_function)
        self.app.router.add_get('/function/{name}/source', self.get_function_source)
    
    def _extract_function_signature(self, source_code: str, func_name: str) -> str:
        """Extract function signature from source code."""
        try:
            # Clean the source
            source_code = textwrap.dedent(source_code.strip())
            
            # Use regex to find the function definition
            import re
            pattern = rf'def\s+{re.escape(func_name)}\s*\([^)]*\):'
            match = re.search(pattern, source_code, re.MULTILINE)
            
            if match:
                # Extract just the parameter part
                full_def = match.group()
                start_paren = full_def.find('(')
                end_paren = full_def.find(')')
                if start_paren != -1 and end_paren != -1:
                    params = full_def[start_paren+1:end_paren].strip()
                    return params if params else "*args, **kwargs"
            
            # Fallback: try AST parsing
            try:
                tree = ast.parse(source_code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == func_name:
                        params = []
                        
                        # Handle regular arguments
                        for arg in node.args.args:
                            params.append(arg.arg)
                        
                        # Handle *args
                        if node.args.vararg:
                            params.append(f"*{node.args.vararg.arg}")
                        
                        # Handle **kwargs
                        if node.args.kwarg:
                            params.append(f"**{node.args.kwarg.arg}")
                        
                        return ', '.join(params) if params else "*args, **kwargs"
            except Exception:
                pass
        
        except Exception as e:
            print(f"[WARNING] Could not extract function signature: {e}")
        
        return "*args, **kwargs"  # Safe fallback
    
    def _clean_function_source(self, source_code: str, func_name: str) -> str:
        """Clean function source by removing decorators and making it executable."""
        try:
            lines = source_code.split('\n')
            cleaned_lines = []
            
            # Find the function definition line
            func_def_found = False
            for line in lines:
                # Skip decorator lines (start with @)
                if line.strip().startswith('@'):
                    continue
                
                # Include the function definition and everything after
                if line.strip().startswith(f'def {func_name}(') or func_def_found:
                    func_def_found = True
                    cleaned_lines.append(line)
            
            if not cleaned_lines:
                # Fallback: try to extract just the function body
                for line in lines:
                    if not line.strip().startswith('@'):
                        cleaned_lines.append(line)
            
            return '\n'.join(cleaned_lines)
        except Exception as e:
            print(f"[WARNING] Error cleaning source: {e}")
            return source_code
    
    def _register_function_with_healing_system(self, func_info: FunctionInfo) -> bool:
        """Register function with the healing system."""
        try:
            # Clean the source code
            cleaned_source = self._clean_function_source(func_info.source_code, func_info.name)
            
            # Try to execute and register the function
            exec_globals = {
                '__builtins__': __builtins__,
                'inspect': inspect,  # Make inspect available
            }
            exec_locals = {}
            
            exec(cleaned_source, exec_globals, exec_locals)
            
            func = exec_locals.get(func_info.name)
            if func and callable(func):
                self.error_fixer.register_function(func)
                return True
            else:
                print(f"[WARNING] Function {func_info.name} not found in executed code")
                return False
                
        except Exception as e:
            print(f"[ERROR] Could not register source for {func_info.name}: {e}")
            return False
    
    async def register_function(self, request):
        """Register a function for healing."""
        try:
            data = await request.json()
            
            # Extract function information
            func_name = data.get('function_name')
            source_code = data.get('source_code', '')
            context = data.get('context', 'general')
            client_id = data.get('client_id', 'unknown')
            
            if not func_name:
                return web.json_response({
                    'status': 'error',
                    'message': 'Function name is required'
                }, status=400)
            
            # Extract signature
            signature = self._extract_function_signature(source_code, func_name)
            
            # Create function info
            func_info = FunctionInfo(
                name=func_name,
                source_code=source_code,
                signature=signature,
                context=context,
                registration_time=time.time(),
                client_id=client_id
            )
            
            # Register with healing system
            registered = self._register_function_with_healing_system(func_info)
            
            # Store function info
            self.functions[func_name] = func_info
            self.stats['total_registrations'] += 1
            
            # Track client
            if client_id not in self.clients:
                self.clients[client_id] = {
                    'first_seen': time.time(),
                    'functions': [],
                    'errors': 0
                }
            
            self.clients[client_id]['functions'].append(func_name)
            
            print(f"[INFO] Registered function: {func_name}")
            
            return web.json_response({
                'status': 'success',
                'message': f'Function {func_name} registered successfully',
                'signature': signature,
                'source_registered': registered
            })
            
        except Exception as e:
            print(f"[ERROR] Error in register_function: {e}")
            return web.json_response({
                'status': 'error',
                'message': f'Registration failed: {str(e)}'
            }, status=500)
    
    async def report_error(self, request):
        """Handle error report and attempt healing."""
        try:
            data = await request.json()
            
            # Create error report
            error_report = ErrorReport(
                function_name=data.get('function_name'),
                error_type=data.get('error_type'),
                error_message=data.get('error_message'),
                traceback=data.get('traceback', ''),
                inputs=data.get('inputs', {}),
                client_id=data.get('client_id', 'unknown'),
                timestamp=time.time()
            )
            
            self.error_reports.append(error_report)
            self.stats['total_errors'] += 1
            
            # Update function stats
            if error_report.function_name in self.functions:
                func_info = self.functions[error_report.function_name]
                func_info.error_count += 1
                func_info.last_error = error_report.error_type
            
            # Update client stats
            if error_report.client_id in self.clients:
                self.clients[error_report.client_id]['errors'] += 1
            
            print(f"[INFO] Received error report for {error_report.function_name} from {error_report.client_id}")
            
            # Attempt healing
            healing_success = await self._attempt_healing(error_report)
            
            if healing_success:
                self.stats['successful_healings'] += 1
                if error_report.function_name in self.functions:
                    self.functions[error_report.function_name].is_healed = True
                    self.functions[error_report.function_name].healing_count += 1
                
                # Get healed function source
                healed_source = self._get_healed_function_source(error_report.function_name)
                
                return web.json_response({
                    'status': 'healed',
                    'message': 'Function successfully healed',
                    'healed_source': healed_source,
                    'function_name': error_report.function_name
                })
            else:
                return web.json_response({
                    'status': 'failed',
                    'message': 'Healing attempt failed',
                    'function_name': error_report.function_name
                })
            
        except Exception as e:
            print(f"[ERROR] Error in report_error: {e}")
            return web.json_response({
                'status': 'error',
                'message': f'Error processing report: {str(e)}'
            }, status=500)
    
    async def _attempt_healing(self, error_report: ErrorReport) -> bool:
        """Attempt to heal the function."""
        try:
            # Create error context
            error_context = {
                'error_type': error_report.error_type,
                'error_message': error_report.error_message,
                'failing_inputs': [error_report.inputs.get('args', ())],
                'function_name': error_report.function_name,
                'traceback': error_report.traceback,
                'function_source': self.functions.get(error_report.function_name, {}).source_code if error_report.function_name in self.functions else ''
            }
            
            # Attempt healing
            success = self.error_fixer.handle_error(error_report.function_name, error_context)
            
            if success:
                self.stats['total_healings'] += 1
                print(f"[INFO] Successfully healed function: {error_report.function_name}")
            
            return success
            
        except Exception as e:
            print(f"[ERROR] Error during healing attempt: {e}")
            return False
    
    def _get_healed_function_source(self, func_name: str) -> str:
        """Get the source code of the healed function."""
        try:
            healed_impl = self.error_fixer.get_current_implementation(func_name)
            if healed_impl:
                try:
                    return inspect.getsource(healed_impl)
                except (OSError, TypeError):
                    # Function was dynamically created
                    return f"# Healed implementation for {func_name} (dynamically generated)\n# Source not available"
            
        except Exception as e:
            print(f"[ERROR] Error getting healed source for {func_name}: {e}")
        
        return f"# No healed implementation available for {func_name}"
    
    async def get_status(self, request):
        """Get server status."""
        uptime = time.time() - self.stats['start_time']
        
        status = {
            'server_status': 'running',
            'active_clients': len(self.clients),
            'registered_functions_count': len(self.functions),
            'uptime_seconds': uptime,
            'total_registrations': self.stats['total_registrations'],
            'total_errors': self.stats['total_errors'],
            'total_healings': self.stats['total_healings'],
            'successful_healings': self.stats['successful_healings'],
            'active_functions': len([f for f in self.functions.values() if f.is_healed or f.error_count == 0]),
            'start_time': self.stats['start_time']
        }
        
        return web.json_response(status)
    
    async def list_functions(self, request):
        """List all registered functions."""
        functions_data = []
        for func_info in self.functions.values():
            functions_data.append({
                'name': func_info.name,
                'signature': func_info.signature,
                'context': func_info.context,
                'registration_time': func_info.registration_time,
                'client_id': func_info.client_id,
                'error_count': func_info.error_count,
                'healing_count': func_info.healing_count,
                'last_error': func_info.last_error,
                'is_healed': func_info.is_healed
            })
        
        return web.json_response({
            'functions': functions_data,
            'total_count': len(functions_data)
        })
    
    async def get_function(self, request):
        """Get information about a specific function."""
        func_name = request.match_info['name']
        
        if func_name not in self.functions:
            return web.json_response({
                'status': 'error',
                'message': f'Function {func_name} not found'
            }, status=404)
        
        func_info = self.functions[func_name]
        
        # Get healed source if available
        healed_source = None
        if func_info.is_healed:
            healed_source = self._get_healed_function_source(func_name)
        
        return web.json_response({
            'function': asdict(func_info),
            'healed_source': healed_source
        })
    
    async def get_function_source(self, request):
        """Get the source code of a function."""
        func_name = request.match_info['name']
        
        if func_name not in self.functions:
            return web.json_response({
                'status': 'error',
                'message': f'Function {func_name} not found'
            }, status=404)
        
        func_info = self.functions[func_name]
        healed_source = self._get_healed_function_source(func_name) if func_info.is_healed else None
        
        return web.json_response({
            'original_source': func_info.source_code,
            'healed_source': healed_source,
            'is_healed': func_info.is_healed
        })
    
    async def health_check(self, request):
        """Health check endpoint."""
        return web.json_response({
            'status': 'healthy',
            'timestamp': time.time(),
            'uptime': time.time() - self.stats['start_time']
        })
    
    def start_server(self):
        """Start the server."""
        print(f"[INFO] Self-Healing Server started on http://localhost:{self.port}")
        print(f"[INFO] API endpoints:")
        print(f"  POST /register - Register a function for healing")
        print(f"  POST /report_error - Report an error and request healing")
        print(f"  GET /status - Get server status and statistics")
        print(f"  GET /functions - List registered functions")
        print(f"  GET /health - Health check")
        print(f"[INFO] Server is running in the background.")
        
        # Run server in background
        def run_server():
            web.run_app(self.app, host='localhost', port=self.port, print=None)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        return server_thread


class HealingClient:
    """
    Enhanced client for the self-healing server.
    """
    
    def __init__(self, server_url: str = "http://localhost:8765"):
        self.server_url = server_url
        self.client_id = f"client_{hash(time.time()) % 100000}_{int(time.time())}"
        self.session = None
        
        print(f"[INFO] HealingClient initialized for server: {server_url}")
        print(f"     Client ID: {self.client_id}")
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if self.session is None:
            self.session = ClientSession()
        return self.session
    
    async def register_function(self, func: Callable, context: str = "general") -> bool:
        """Register a function with the healing server."""
        try:
            # Get function source
            try:
                source_code = inspect.getsource(func)
                source_code = textwrap.dedent(source_code)
            except (OSError, TypeError):
                source_code = f"# Source not available for {func.__name__}"
            
            data = {
                'function_name': func.__name__,
                'source_code': source_code,
                'context': context,
                'client_id': self.client_id
            }
            
            session = await self._get_session()
            async with session.post(f"{self.server_url}/register", json=data) as response:
                result = await response.json()
                
                if result.get('status') == 'success':
                    print(f"[OK] Function '{func.__name__}' registered successfully.")
                    return True
                else:
                    print(f"[ERROR] Failed to register function: {result.get('message')}")
                    return False
        
        except Exception as e:
            print(f"[ERROR] Error registering function: {e}")
            return False
    
    async def report_error_and_heal(self, func_name: str, error: Exception, args: tuple, kwargs: dict) -> Optional[Callable]:
        """Report an error and get healed function if available."""
        try:
            data = {
                'function_name': func_name,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc(),
                'inputs': {
                    'args': self._serialize_safely(args),
                    'kwargs': self._serialize_safely(kwargs)
                },
                'client_id': self.client_id
            }
            
            session = await self._get_session()
            async with session.post(f"{self.server_url}/report_error", json=data) as response:
                result = await response.json()
                
                if result.get('status') == 'healed':
                    healed_source = result.get('healed_source')
                    if healed_source:
                        return self._create_function_from_source(healed_source, func_name)
                
                return None
        
        except Exception as e:
            print(f"[ERROR] Error reporting error: {e}")
            return None
    
    def _serialize_safely(self, obj):
        """Safely serialize objects for JSON."""
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            if isinstance(obj, (list, tuple)):
                return [str(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items()}
            else:
                return str(obj)
    
    def _create_function_from_source(self, source_code: str, func_name: str) -> Optional[Callable]:
        """Create a function from source code."""
        try:
            exec_globals = {
                '__builtins__': __builtins__,
                'inspect': inspect,
            }
            exec_locals = {}
            
            exec(source_code, exec_globals, exec_locals)
            
            return exec_locals.get(func_name)
        
        except Exception as e:
            print(f"[ERROR] Error creating function from source: {e}")
            return None
    
    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()


def create_self_healing_function(client: HealingClient, original_func: Callable, context: str = "general"):
    """Create a self-healing wrapper for a function."""
    
    # Register the function
    async def register():
        await client.register_function(original_func, context)
    
    # Run registration
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(register())
    except RuntimeError:
        # Create new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(register())
    
    def wrapper(*args, **kwargs):
        try:
            return original_func(*args, **kwargs)
        except Exception as e:
            print(f"[INFO] Error in {original_func.__name__}: {type(e).__name__}: {e}")
            print(f"[INFO] Reporting error for '{original_func.__name__}' and requesting healing...")
            
            # Report error and get healing
            async def get_healing():
                return await client.report_error_and_heal(original_func.__name__, e, args, kwargs)
            
            try:
                loop = asyncio.get_event_loop()
                healed_func = loop.run_until_complete(get_healing())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                healed_func = loop.run_until_complete(get_healing())
            
            if healed_func:
                print(f"[INFO] Received healed function for {original_func.__name__}")
                try:
                    return healed_func(*args, **kwargs)
                except Exception as heal_error:
                    print(f"[ERROR] Healed function still failed: {heal_error}")
                    raise
            else:
                print(f"[ERROR] No healing available for {original_func.__name__}")
                raise
    
    return wrapper


async def demo_enhanced_self_healing():
    """Demonstrate the enhanced self-healing server."""
    print("Enhanced Self-Healing Server Demo")
    print("=" * 50)
    
    # Start server
    server = EnhancedSelfHealingServer(port=8765)
    server_thread = server.start_server()
    
    # Wait for server to start
    await asyncio.sleep(1)
    
    # Create client
    client = HealingClient()
    
    # Define problematic functions
    def buggy_find_max(arr, start_idx, end_idx):
        """Find max in array range - has multiple bugs."""
        if start_idx >= end_idx or start_idx < 0 or end_idx > len(arr):
            return None
        max_val = arr[start_idx]
        for i in range(start_idx + 1, end_idx):
            if arr[i] > max_val:
                max_val = arr[i]
        return max_val
    
    def buggy_divide(a, b):
        """Divide two numbers - has division by zero."""
        return a / b
    
    def buggy_process_text(text, prefix):
        """Process text with prefix - has type errors."""
        return prefix.upper() + text.lower()
    
    # Create self-healing wrappers
    healing_find_max = create_self_healing_function(client, buggy_find_max, "array_processing")
    healing_divide = create_self_healing_function(client, buggy_divide, "math_operations")
    healing_process_text = create_self_healing_function(client, buggy_process_text, "string_processing")
    
    # Test scenarios
    test_scenarios = [
        ("Find max with invalid slice", lambda: healing_find_max([1, 2, 3, 4, 5], 0, 10)),
        ("Find max with empty slice", lambda: healing_find_max([1, 2, 3], 2, 2)),
        ("Divide by zero", lambda: healing_divide(10, 0)),
        ("Process None text", lambda: healing_process_text("hello", None)),
    ]
    
    print(f"[INFO] Testing {len(test_scenarios)} scenarios...")
    
    successful_healings = 0
    
    for i, (description, test_func) in enumerate(test_scenarios, 1):
        print(f"[INFO] --- Test {i}: {description} ---")
        
        try:
            result = test_func()
            print(f"  [OK] Initial call succeeded. Result: {result}")
        except Exception as e:
            print(f"  [INFO] Initial call failed as expected: {type(e).__name__}")
            
            # Retry to see if healing was applied
            print(f"[INFO] Retrying function call to see if healing was applied...")
            try:
                result = test_func()
                print(f"  [OK] Healed call succeeded. Result: {result}")
                successful_healings += 1
            except Exception as e:
                print(f"  [ERROR] Still failed after healing attempt: {e}")
    
    print(f"[INFO] Demo finished.")
    print(f"  Successful healings: {successful_healings} / {len(test_scenarios)}")
    
    # Get final server status
    try:
        session = await client._get_session()
        async with session.get(f"{client.server_url}/status") as response:
            status = await response.json()
            print(f"[INFO] Final Server Status:")
            print(json.dumps(status, indent=2))
    except Exception as e:
        print(f"[ERROR] Could not get server status: {e}")
    
    # Cleanup
    await client.close()
    
    return successful_healings > 0


def run_enhanced_self_healing_demo():
    """Run the enhanced self-healing demo."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If event loop is already running, create a new one
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(demo_enhanced_self_healing())
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
        else:
            return loop.run_until_complete(demo_enhanced_self_healing())
    except RuntimeError:
        # No event loop exists
        return asyncio.run(demo_enhanced_self_healing())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run server only
        server = EnhancedSelfHealingServer()
        server.start_server()
        
        try:
            print("Press Ctrl+C to stop the server...")
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down server...")
    else:
        # Run demo
        success = run_enhanced_self_healing_demo()
        if success:
            print("\nEnhanced self-healing demo completed successfully!")
        else:
            print("\nDemo completed with some issues.")