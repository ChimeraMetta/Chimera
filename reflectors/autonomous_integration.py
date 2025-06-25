"""
Integration layer for autonomous error fixing with existing evolution system.
This bridges the gap between the new autonomous system and your existing components.

Location: reflectors/autonomous_integration.py
"""

import os
import time
import functools
from typing import Dict, List, Callable, Any, Optional
from contextlib import contextmanager

# Imports for integration with existing system
from reflectors.dynamic_monitor import DynamicMonitor
from reflectors.autonomous_evolution import AutonomousMonitor


class AutonomousEvolutionIntegrator:
    """
    Integrates autonomous error fixing with your existing evolution system.
    Provides a bridge between new autonomous capabilities and existing CLI/components.
    """
    
    def __init__(self, base_monitor: DynamicMonitor = None, ontology_path: str = None):
        """Initialize with optional existing monitor and ontology."""
        self.base_monitor = base_monitor or DynamicMonitor()
        self.ontology_path = ontology_path
        
        # Initialize autonomous components
        self.autonomous_monitor = AutonomousMonitor(self.base_monitor.metta_space)
        self.error_fixer = self.autonomous_monitor.error_fixer
        
        # Load ontology if provided
        if ontology_path and os.path.exists(ontology_path):
            self.autonomous_monitor.load_metta_rules(ontology_path)
            print(f"[OK] Loaded ontology for autonomous evolution: {ontology_path}")
        
        # Track function replacements for rollback
        self.replacement_history = {}  # func_name -> list of (timestamp, implementation)
        
    def enhance_existing_function(self, func: Callable, context: str = None) -> Callable:
        """
        Enhance an existing function with autonomous error fixing.
        This can be used to upgrade functions that are already in your system.
        """
        func_name = func.__name__
        
        # Store original for rollback capability
        self.replacement_history[func_name] = [(time.time(), func)]
        
        # Apply autonomous transformation
        enhanced_func = self.autonomous_monitor.autonomous_transform(
            context=context, 
            enable_auto_fix=True, 
            max_fix_attempts=3
        )(func)
        
        print(f"[OK] Enhanced '{func_name}' with autonomous error fixing")
        return enhanced_func
    
    def monitor_module_functions(self, module, context_prefix: str = None):
        """
        Automatically apply autonomous monitoring to all functions in a module.
        Useful for monitoring entire codebases.
        """
        enhanced_functions = {}
        
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and not name.startswith('_'):
                try:
                    # Create context from module and function name
                    if context_prefix:
                        context = f"{context_prefix}.{name}"
                    else:
                        context = f"{module.__name__}.{name}"
                    
                    # Enhance the function
                    enhanced_func = self.enhance_existing_function(obj, context)
                    enhanced_functions[name] = enhanced_func
                    
                    # Replace in module
                    setattr(module, name, enhanced_func)
                    
                except Exception as e:
                    print(f"[WARNING] Could not enhance {name}: {e}")
        
        print(f"[OK] Enhanced {len(enhanced_functions)} functions in {module.__name__}")
        return enhanced_functions
    
    def create_fixing_context_manager(self, func: Callable):
        """
        Create a context manager that provides autonomous fixing for a specific function.
        Useful for temporary enhancement of critical code sections.
        """
        @contextmanager
        def autonomous_context(*args, **kwargs):
            """Context manager that applies autonomous fixing temporarily."""
            func_name = func.__name__
            
            # Store original implementation
            original_impl = self.error_fixer.function_registry.get(func_name, func)
            
            try:
                # Register for autonomous fixing if not already registered
                if func_name not in self.error_fixer.function_registry:
                    self.error_fixer.register_function(func)
                
                # Create enhanced version
                enhanced_func = self.autonomous_monitor.autonomous_transform()(func)
                
                yield enhanced_func
                
            except Exception as e:
                # Handle any autonomous fixing errors
                print(f"[ERROR] Error in autonomous context for '{func_name}': {e}")
                
                # Try to apply emergency fix
                error_context = self.autonomous_monitor._create_error_context(func, e, args)
                fix_applied = self.error_fixer.handle_error(func_name, error_context)
                
                if fix_applied:
                    print(f"[INFO] Emergency fix applied for '{func_name}'")
                    fixed_impl = self.error_fixer.get_current_implementation(func_name)
                    yield fixed_impl
                else:
                    raise
            
            finally:
                # Restore original if needed
                if func_name in self.error_fixer.current_implementations:
                    print(f"[INFO] Autonomous context ended for '{func_name}'")
        
        return autonomous_context
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about autonomous evolution."""
        stats = {
            'registered_functions': len(self.error_fixer.function_registry),
            'functions_with_fixes': len([
                name for name, impl in self.error_fixer.current_implementations.items()
                if impl != self.error_fixer.function_registry.get(name)
            ]),
            'total_fix_attempts': sum(self.error_fixer.fix_attempts.values()),
            'error_history_count': sum(len(history) for history in self.error_fixer.error_history.values()),
            'replacement_history': {
                name: len(history) for name, history in self.replacement_history.items()
            }
        }
        
        # Calculate success rate
        total_errors = stats['error_history_count']
        successful_fixes = stats['functions_with_fixes']
        stats['fix_success_rate'] = successful_fixes / total_errors if total_errors > 0 else 0.0
        
        return stats
    
    def rollback_function(self, func_name: str, steps_back: int = 1) -> bool:
        """Rollback a function to a previous implementation."""
        if func_name not in self.replacement_history:
            print(f"[ERROR] No rollback history for '{func_name}'")
            return False
        
        history = self.replacement_history[func_name]
        if len(history) <= steps_back:
            print(f"[ERROR] Not enough history to rollback {steps_back} steps for '{func_name}'")
            return False
        
        # Get the implementation from steps_back ago
        target_impl = history[-(steps_back + 1)][1]
        
        # Update current implementation
        self.error_fixer.current_implementations[func_name] = target_impl
        
        print(f"[INFO] Rolled back '{func_name}' {steps_back} step(s)")
        return True
    
    def export_fixed_functions(self, output_dir: str) -> Dict[str, str]:
        """Export all autonomously fixed functions to files."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        for func_name, current_impl in self.error_fixer.current_implementations.items():
            original_impl = self.error_fixer.function_registry.get(func_name)
            
            # Only export if it's been fixed (implementation changed)
            if current_impl != original_impl:
                try:
                    # Get source code of fixed implementation
                    import inspect
                    fixed_source = inspect.getsource(current_impl)
                    
                    # Create output file
                    output_file = os.path.join(output_dir, f"{func_name}_autonomous_fix.py")
                    
                    with open(output_file, 'w') as f:
                        f.write(f"# Autonomously fixed implementation of {func_name}\n")
                        f.write(f"# Generated on: {time.ctime()}\n")
                        f.write(f"# Fix attempts: {self.error_fixer.fix_attempts.get(func_name, 0)}\n")
                        f.write(f"# Error history: {len(self.error_fixer.error_history.get(func_name, []))} errors\n")
                        f.write(f"\n{fixed_source}\n")
                    
                    exported_files[func_name] = output_file
                    print(f"[INFO] Exported fixed '{func_name}' to {output_file}")
                    
                except Exception as e:
                    print(f"[ERROR] Could not export '{func_name}': {e}")
        
        return exported_files


# Integration with existing dynamic monitor
class EnhancedDynamicMonitor(DynamicMonitor):
    """
    Enhanced version of your existing DynamicMonitor with autonomous capabilities.
    Drop-in replacement that adds autonomous evolution.
    """
    
    def __init__(self, metta_space=None):
        super().__init__(metta_space)
        self.autonomous_integrator = AutonomousEvolutionIntegrator(self)
        self._autonomous_mode = False
    
    def enable_autonomous_mode(self, ontology_path: str = None):
        """Enable autonomous error fixing mode."""
        if ontology_path:
            self.autonomous_integrator.ontology_path = ontology_path
            if os.path.exists(ontology_path):
                self.load_metta_rules(ontology_path)
        
        self._autonomous_mode = True
        print("[INFO] Autonomous evolution mode enabled")
    
    def autonomous_transform(self, context: Optional[str] = None, **kwargs):
        """
        New decorator that provides autonomous error fixing.
        Can be used alongside or instead of hybrid_transform.
        """
        if not self._autonomous_mode:
            print("[WARNING] Autonomous mode not enabled, falling back to hybrid_transform")
            return self.hybrid_transform(context, **kwargs)
        
        return self.autonomous_integrator.autonomous_monitor.autonomous_transform(context, **kwargs)
    
    def get_autonomous_stats(self) -> Dict[str, Any]:
        """Get autonomous evolution statistics."""
        if not self._autonomous_mode:
            return {"autonomous_mode": False, "message": "Autonomous mode not enabled"}
        
        return self.autonomous_integrator.get_evolution_statistics()
    
    def export_autonomous_fixes(self, output_dir: str) -> Dict[str, str]:
        """Export all autonomously generated fixes."""
        if not self._autonomous_mode:
            print("[ERROR] Autonomous mode not enabled")
            return {}
        
        return self.autonomous_integrator.export_fixed_functions(output_dir)


class AutonomousEvolutionWrapper:
    """
    Wrapper class that provides backwards compatibility with your existing system
    while adding autonomous evolution capabilities.
    """
    
    def __init__(self, existing_monitor=None, ontology_path: str = None):
        """
        Initialize wrapper with optional existing monitor.
        
        Args:
            existing_monitor: Your existing DynamicMonitor instance
            ontology_path: Path to MeTTa ontology file
        """
        
        if existing_monitor:
            # Enhance existing monitor
            self.monitor = EnhancedDynamicMonitor(existing_monitor.metta_space)
            # Copy over any existing configuration
            if hasattr(existing_monitor, 'evolution_callback'):
                self.monitor.evolution_callback = existing_monitor.evolution_callback
        else:
            # Create new enhanced monitor
            self.monitor = EnhancedDynamicMonitor()
        
        # Enable autonomous mode
        self.monitor.enable_autonomous_mode(ontology_path)
        
        self.integrator = self.monitor.autonomous_integrator
    
    def enhance_function(self, func: callable, context: str = None) -> callable:
        """Enhanced function with autonomous error fixing."""
        return self.integrator.enhance_existing_function(func, context)
    
    def get_autonomous_decorator(self, context: str = None, **kwargs):
        """Get the autonomous transformation decorator."""
        return self.monitor.autonomous_transform(context, **kwargs)
    
    def get_stats(self) -> dict:
        """Get autonomous evolution statistics."""
        return self.monitor.get_autonomous_stats()
    
    def export_fixes(self, output_dir: str) -> dict:
        """Export all autonomous fixes."""
        return self.monitor.export_autonomous_fixes(output_dir)


# Testing and validation utilities
class AutonomousEvolutionTester:
    """
    Testing utilities for validating autonomous evolution behavior.
    """
    
    def __init__(self, integrator: AutonomousEvolutionIntegrator):
        self.integrator = integrator
        self.test_results = []
    
    def test_error_recovery(self, func: Callable, error_cases: List[tuple]) -> Dict[str, Any]:
        """
        Test autonomous error recovery for a function with known error cases.
        
        Args:
            func: Function to test
            error_cases: List of (args, kwargs, expected_error_type) tuples
        """
        func_name = func.__name__
        enhanced_func = self.integrator.enhance_existing_function(func)
        
        recovery_results = {
            "function_name": func_name,
            "total_cases": len(error_cases),
            "recoveries": 0,
            "failures": 0,
            "details": []
        }
        
        for i, (args, kwargs, expected_error) in enumerate(error_cases):
            case_result = {
                "case_number": i + 1,
                "args": args,
                "kwargs": kwargs,
                "expected_error": expected_error,
                "original_failed": False,
                "autonomous_recovered": False,
                "final_result": None
            }
            
            # Test original function (should fail)
            try:
                func(*args, **kwargs)
                case_result["original_failed"] = False
            except Exception as e:
                case_result["original_failed"] = type(e).__name__ == expected_error
            
            # Test enhanced function (should recover)
            try:
                result = enhanced_func(*args, **kwargs)
                case_result["autonomous_recovered"] = True
                case_result["final_result"] = result
                recovery_results["recoveries"] += 1
            except Exception as e:
                case_result["autonomous_recovered"] = False
                case_result["final_result"] = f"Still failed: {e}"
                recovery_results["failures"] += 1
            
            recovery_results["details"].append(case_result)
        
        recovery_results["success_rate"] = recovery_results["recoveries"] / recovery_results["total_cases"]
        self.test_results.append(recovery_results)
        
        return recovery_results
    
    def generate_test_report(self, output_file: str = None) -> str:
        """Generate a comprehensive test report."""
        if not self.test_results:
            return "No test results available"
        
        report_lines = [
            "Autonomous Evolution Test Report",
            "=" * 40,
            f"Generated: {time.ctime()}",
            f"Tests completed: {len(self.test_results)}",
            ""
        ]
        
        total_cases = sum(r["total_cases"] for r in self.test_results)
        total_recoveries = sum(r["recoveries"] for r in self.test_results)
        overall_success_rate = total_recoveries / total_cases if total_cases > 0 else 0
        
        report_lines.extend([
            f"Overall Statistics:",
            f"  Total test cases: {total_cases}",
            f"  Successful recoveries: {total_recoveries}",
            f"  Overall success rate: {overall_success_rate:.1%}",
            ""
        ])
        
        for result in self.test_results:
            report_lines.extend([
                f"Function: {result['function_name']}",
                f"  Cases tested: {result['total_cases']}",
                f"  Recoveries: {result['recoveries']}",
                f"  Failures: {result['failures']}",
                f"  Success rate: {result['success_rate']:.1%}",
                ""
            ])
            
            for detail in result["details"]:
                status = "[OK] RECOVERED" if detail["autonomous_recovered"] else "[ERROR] FAILED"
                report_lines.append(f"    Case {detail['case_number']}: {status}")
                report_lines.append(f"      Args: {detail['args']}")
                report_lines.append(f"      Result: {detail['final_result']}")
            
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"[INFO] Test report saved to: {output_file}")
        
        return report


# Example usage patterns
def example_usage_patterns():
    """
    Examples of how to use the autonomous evolution system in different scenarios.
    """
    
    # Pattern 1: Quick autonomous enhancement of a single function
    def pattern_1_single_function():
        integrator = AutonomousEvolutionIntegrator()
        
        def my_buggy_function(data, index):
            return data[index]  # IndexError if index invalid
        
        # Enhance with autonomous fixing
        fixed_function = integrator.enhance_existing_function(my_buggy_function, "data_access")
        
        # Use the enhanced function
        try:
            result = fixed_function([1, 2, 3], 5)  # Will trigger auto-fix
            print(f"Result: {result}")
        except:
            print("Even the fix couldn't handle this case")
    
    # Pattern 2: Temporary autonomous protection
    def pattern_2_context_manager():
        integrator = AutonomousEvolutionIntegrator()
        
        def risky_operation(data):
            return data[0] / data[1]  # Multiple potential errors
        
        # Use context manager for temporary protection
        context = integrator.create_fixing_context_manager(risky_operation)
        
        with context() as protected_func:
            result = protected_func([])  # Will try to auto-fix
            print(f"Protected result: {result}")
    
    # Pattern 3: Module-wide autonomous monitoring
    def pattern_3_module_monitoring():
        integrator = AutonomousEvolutionIntegrator()
        
        # Monitor an entire module
        import some_module  # Your existing module
        enhanced_funcs = integrator.monitor_module_functions(some_module, "my_project")
        
        # All functions in the module now have autonomous fixing
        # They will self-repair when errors occur
        
        # Get statistics later
        stats = integrator.get_evolution_statistics()
        print(f"Autonomous fixes applied: {stats['functions_with_fixes']}")
    
    # Pattern 4: Integration with existing monitoring
    def pattern_4_enhanced_monitor():
        # Replace your existing monitor with enhanced version
        monitor = EnhancedDynamicMonitor()
        monitor.enable_autonomous_mode("path/to/ontology.metta")
        
        @monitor.autonomous_transform(context="critical_operations")
        def critical_function(data):
            # This function will auto-fix itself if errors occur
            return process_data(data)
        
        # Use normally - auto-fixing happens transparently
        result = critical_function(some_data)
    
    return {
        "single_function": pattern_1_single_function,
        "context_manager": pattern_2_context_manager, 
        "module_monitoring": pattern_3_module_monitoring,
        "enhanced_monitor": pattern_4_enhanced_monitor
    }


if __name__ == "__main__":
    # Demo the integration capabilities
    print("[INFO] Autonomous Evolution Integration Demo")
    print("=" * 50)
    
    # Create integrator
    integrator = AutonomousEvolutionIntegrator()
    
    # Example problematic function
    def demo_function(numbers, operation, default_value):
        """Demo function with multiple potential error points."""
        if operation == "sum":
            return sum(numbers) + default_value
        elif operation == "max":
            return max(numbers) + default_value  
        elif operation == "avg":
            return sum(numbers) / len(numbers) + default_value
        else:
            return numbers[0] + default_value
    
    # Test autonomous recovery
    tester = AutonomousEvolutionTester(integrator)
    
    error_cases = [
        (([],), {"operation": "sum", "default_value": 0}, "TypeError"),  # sum([]) is fine, but max([]) fails
        (([],), {"operation": "max", "default_value": 0}, "ValueError"),  # max([]) fails
        (([],), {"operation": "avg", "default_value": 0}, "ZeroDivisionError"),  # division by zero
        (([],), {"operation": "first", "default_value": 0}, "IndexError"),  # numbers[0] fails
        ((None,), {"operation": "sum", "default_value": 0}, "TypeError"),  # sum(None) fails
    ]
    
    results = tester.test_error_recovery(demo_function, error_cases)
    
    print(f"\n[INFO] Test Results for {results['function_name']}:")
    print(f"  Success rate: {results['success_rate']:.1%}")
    print(f"  Recoveries: {results['recoveries']}/{results['total_cases']}")
    
    # Generate report
    report = tester.generate_test_report()
    print(f"\n[INFO] Full Test Report:")
    print(report)
    
    print(f"\n[OK] Integration demo completed!")
    print(f"Ready to integrate autonomous evolution into your existing system.")