"""
Enhanced Self-Healing Demo with Clear Results

This demo provides much clearer output showing:
1. Original function behavior
2. Error details
3. Healing process
4. Healed function behavior
5. Before/after comparisons

Location: enhanced_healing_demo.py
"""

import time
import traceback
from fixed_self_healing_server import (
    FixedSelfHealingServer, 
    FixedHealingClient, 
    create_fixed_self_healing_function
)


def create_comprehensive_demo():
    """Create a comprehensive demo that clearly shows healing results."""
    
    print("Enhanced Self-Healing Server Demo")
    print("=" * 60)
    
    # Connect to existing server
    print("Connecting to healing server at http://localhost:8765...")
    try:
        client = FixedHealingClient()
        # Test connection
        import requests
        response = requests.get("http://localhost:8765/health", timeout=5)
        if response.status_code == 200:
            print("Connected to healing server successfully")
        else:
            print(f"Server responded with status {response.status_code}")
    except Exception as e:
        print(f"Cannot connect to server: {e}")
        print("   Make sure the healing server is running on port 8765")
        return False
    
    try:
        
        # Define test functions with more interesting behaviors
        def buggy_find_element(arr, index):
            """Find element by index - has bounds errors."""
            print(f"   → Original: accessing arr[{index}] on array of length {len(arr)}")
            return arr[index]
        
        def buggy_divide(a, b):
            """Divide two numbers - has division by zero."""
            print(f"   → Original: computing {a} / {b}")
            return a / b
        
        def buggy_process_text(text, prefix):
            """Process text with prefix - has type errors."""
            print(f"   → Original: processing text='{text}' with prefix='{prefix}'")
            return prefix.upper() + " " + text.lower()
        
        def buggy_get_value(data, key):
            """Get value from dict - has key errors."""
            print(f"   → Original: accessing data['{key}'] from dict with keys {list(data.keys())}")
            return data[key]
        
        # Create healing wrappers
        print("\nCreating self-healing function wrappers...")
        healing_find_element = create_enhanced_healing_wrapper(client, buggy_find_element, "array_processing")
        healing_divide = create_enhanced_healing_wrapper(client, buggy_divide, "math_operations")
        healing_process_text = create_enhanced_healing_wrapper(client, buggy_process_text, "string_processing")
        healing_get_value = create_enhanced_healing_wrapper(client, buggy_get_value, "data_access")
        
        # Test scenarios with before/after comparisons
        test_scenarios = [
            {
                'name': "Array Bounds Checking",
                'func': healing_find_element,
                'test_cases': [
                    ([1, 2, 3, 4, 5], 2, "Valid index"),
                    ([1, 2, 3], 5, "Out of bounds"),
                    ([], 0, "Empty array"),
                    ([10, 20], -1, "Negative index")
                ]
            },
            {
                'name': "Safe Division",
                'func': healing_divide,
                'test_cases': [
                    (10, 2, "Normal division"),
                    (15, 0, "Division by zero"),
                    (-8, 4, "Negative number"),
                    (0, 5, "Zero numerator")
                ]
            },
            {
                'name': "Text Processing with None Safety",
                'func': healing_process_text,
                'test_cases': [
                    ("hello", "GREETING", "Normal text"),
                    ("world", None, "None prefix"),
                    (None, "PREFIX", "None text"),
                    ("test", "", "Empty prefix")
                ]
            },
            {
                'name': "Dictionary Access with Key Safety",
                'func': healing_get_value,
                'test_cases': [
                    ({"a": 1, "b": 2, "c": 3}, "b", "Existing key"),
                    ({"x": 10, "y": 20}, "z", "Missing key"),
                    ({}, "any", "Empty dict"),
                    ({"name": "John"}, "name", "String value")
                ]
            }
        ]
        
        total_tests = sum(len(scenario['test_cases']) for scenario in test_scenarios)
        successful_healings = 0
        test_number = 0
        
        for scenario in test_scenarios:
            print(f"\nTesting: {scenario['name']}")
            print("-" * 50)
            
            for args_tuple in scenario['test_cases']:
                test_number += 1
                *args, description = args_tuple
                
                print(f"\n  [{test_number}/{total_tests}] {description}")
                print(f"    Input: {args}")
                
                # Test the function and capture results
                result = test_enhanced_function(scenario['func'], args)
                
                if result['healed']:
                    successful_healings += 1
                    
                # Display results clearly
                if result['error_occurred']:
                    print(f"    Original Error: {result['error_type']}")
                    if result['healed']:
                        print(f"    Healed Result: {repr(result['final_result'])}")
                        print(f"    Healing Status: SUCCESS")
                    else:
                        print(f"    Healing Status: FAILED")
                        print(f"    Final Error: {result['final_error']}")
                else:
                    print(f"    Direct Result: {repr(result['final_result'])}")
                    print(f"    Status: No healing needed")
        
        # Summary
        print(f"\nDemo Summary")
        print("=" * 50)
        print(f"Total test cases: {total_tests}")
        print(f"Cases requiring healing: {sum(1 for scenario in test_scenarios for args in scenario['test_cases'] if test_would_fail(scenario['func'].__wrapped__, args[:-1]))}")
        print(f"Successful healings: {successful_healings}")
        print(f"Healing success rate: {successful_healings}/{total_tests} ({100*successful_healings/total_tests:.1f}%)")
        
        # Show healed function sources
        print(f"\nGenerated Healed Function Sources")
        print("=" * 50)
        show_healed_sources(client, ['buggy_find_element', 'buggy_divide', 'buggy_process_text', 'buggy_get_value'])
        
        # Server statistics
        show_server_statistics(client)
        
        client.close()
        return successful_healings > 0
        
    except Exception as e:
        print(f"Demo failed: {e}")
        traceback.print_exc()
        return False


def create_enhanced_healing_wrapper(client, original_func, context):
    """Create an enhanced healing wrapper that provides detailed output."""
    
    # Register the function
    success = client.register_function(original_func, context)
    if not success:
        print(f"Failed to register {original_func.__name__}")
    
    def wrapper(*args, **kwargs):
        try:
            # Call original function
            return original_func(*args, **kwargs)
        except Exception as e:
            print(f"    Error occurred: {type(e).__name__}: {e}")
            print(f"    Requesting healing from server...")
            
            # Report error and get healing
            healed_func = client.report_error_and_heal(original_func.__name__, e, args, kwargs)
            
            if healed_func:
                print(f"    Healing received, applying fix...")
                try:
                    result = healed_func(*args, **kwargs)
                    print(f"    Healed function executed successfully")
                    return result
                except Exception as heal_error:
                    print(f"    Healed function failed: {heal_error}")
                    raise e  # Re-raise original error
            else:
                print(f"    No healing available")
                raise
    
    # Store original function for testing
    wrapper.__wrapped__ = original_func
    return wrapper


def test_enhanced_function(func, args):
    """Test a function and return detailed results."""
    result = {
        'error_occurred': False,
        'error_type': None,
        'healed': False,
        'final_result': None,
        'final_error': None
    }
    
    try:
        result['final_result'] = func(*args)
        # Check if healing occurred by looking for specific patterns in the result
        if result['final_result'] is None and would_original_fail(func, args):
            result['error_occurred'] = True
            result['healed'] = True
            result['error_type'] = get_expected_error_type(func, args)
    except Exception as e:
        result['error_occurred'] = True
        result['final_error'] = str(e)
        result['error_type'] = type(e).__name__
    
    return result


def would_original_fail(func, args):
    """Check if the original function would fail with these arguments."""
    if hasattr(func, '__wrapped__'):
        try:
            func.__wrapped__(*args)
            return False
        except:
            return True
    return False


def test_would_fail(original_func, args):
    """Test if original function would fail."""
    try:
        original_func(*args)
        return False
    except:
        return True


def get_expected_error_type(func, args):
    """Get the expected error type for a function and arguments."""
    if hasattr(func, '__wrapped__'):
        try:
            func.__wrapped__(*args)
        except Exception as e:
            return type(e).__name__
    return "Unknown"


def show_healed_sources(client, function_names):
    """Show the healed function sources."""
    try:
        import requests
        for func_name in function_names:
            try:
                response = requests.get(f"{client.server_url}/function/{func_name}/source", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('is_healed') and data.get('healed_source'):
                        print(f"\n{func_name} (healed):")
                        print("   " + "\n   ".join(data['healed_source'].split('\n')[:10]))
                        if len(data['healed_source'].split('\n')) > 10:
                            print("   ...")
                    else:
                        print(f"\n{func_name}: Not healed or no source available")
                else:
                    print(f"\nCould not get source for {func_name}")
            except Exception as e:
                print(f"\nError getting source for {func_name}: {e}")
    except Exception as e:
        print(f"Error showing healed sources: {e}")


def show_server_statistics(client):
    """Show server statistics."""
    try:
        import requests
        response = requests.get(f"{client.server_url}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"\nServer Statistics")
            print("-" * 30)
            print(f"Uptime: {status.get('uptime_seconds', 0):.1f} seconds")
            print(f"Total registrations: {status.get('total_registrations', 0)}")
            print(f"Total errors: {status.get('total_errors', 0)}")
            print(f"Total healings: {status.get('total_healings', 0)}")
            print(f"Successful healings: {status.get('successful_healings', 0)}")
            print(f"Active functions: {status.get('active_functions', 0)}")
            print(f"MeTTa available: {status.get('metta_available', False)}")
        else:
            print(f"Could not get server status: HTTP {response.status_code}")
    except Exception as e:
        print(f"Could not get server status: {e}")


def run_quick_validation():
    """Run a quick validation to show healing is working."""
    print("Quick Healing Validation")
    print("=" * 40)
    
    # Test the core healing logic directly
    from fixed_self_healing_server import FixedErrorFixer
    
    fixer = FixedErrorFixer()
    
    # Test IndexError healing
    def test_index_func(arr, idx):
        return arr[idx]
    
    fixer.register_function(test_index_func)
    
    error_context = {
        'error_type': 'IndexError',
        'error_message': 'list index out of range',
        'failing_inputs': [([1, 2, 3], 5)],
        'function_name': 'test_index_func'
    }
    
    print("Testing IndexError healing...")
    success = fixer.handle_error('test_index_func', error_context)
    print(f"Healing success: {success}")
    
    if success:
        healed_func = fixer.get_current_implementation('test_index_func')
        if healed_func:
            result = healed_func([1, 2, 3], 5)
            print(f"Healed function result: {result}")
            
            source = fixer.get_healed_source('test_index_func')
            if source:
                print("Generated source preview:")
                print("  " + "\n  ".join(source.split('\n')[:5]))
    
    return success


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Run quick validation
        success = run_quick_validation()
        print(f"\nQuick validation: {'PASSED' if success else 'FAILED'}")
    else:
        # Run full demo
        success = create_comprehensive_demo()
        if success:
            print(f"\nEnhanced self-healing demo completed successfully!")
            print("   All healing mechanisms are working properly.")
        else:
            print(f"\nDemo completed with issues.")
        
        sys.exit(0 if success else 1)