from dynamic_monitor import hybrid_transform
import time

# Load the MeTTa reasoning rules (optional)
try:
    from dynamic_monitor import monitor
    monitor.load_metta_rules("ontology.metta")
except ImportError:
    print("Couldn't import monitor directly, continuing with decorators only")

# A simple function that always succeeds
@hybrid_transform(context="demo.math")
def add_numbers(a, b):
    """Add two numbers together."""
    return a + b

# A function that might raise division by zero
@hybrid_transform(context="demo.math", auto_fix=True)
def divide_numbers(a, b):
    """Divide a by b."""
    return a / b

# A function with potential index error
@hybrid_transform(context="demo.data")
def get_item(items, index):
    """Get an item from a list at the specified index."""
    return items[index]

# A function with type mismatches
@hybrid_transform(context="demo.strings")
def concatenate(text, value):
    """Concatenate text with a value."""
    return text + value

# A more complex function with multiple operations
@hybrid_transform(context="demo.processing")
def process_data(data, factor=1):
    """Process a list of numbers by multiplying by factor and calculating sum."""
    result = []
    for item in data:
        result.append(item * factor)
    return {"processed": result, "sum": sum(result)}

# Main test function
def run_tests():
    print("Running MeTTa Monitor tests...")
    
    # Test successful operations
    print("\n1. Testing add_numbers (should succeed):")
    try:
        result = add_numbers(5, 7)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed with: {type(e).__name__}: {e}")
    
    # Test division by zero
    print("\n2. Testing divide_numbers with zero divisor (should fail):")
    try:
        result = divide_numbers(10, 0)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed with: {type(e).__name__}: {e}")
    
    # Test successful division
    print("\n3. Testing divide_numbers with non-zero divisor (should succeed):")
    try:
        result = divide_numbers(10, 2)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed with: {type(e).__name__}: {e}")
    
    # Test index error
    print("\n4. Testing get_item with invalid index (should fail):")
    try:
        result = get_item([1, 2, 3], 5)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed with: {type(e).__name__}: {e}")
    
    # Test type error
    print("\n5. Testing concatenate with incompatible types (should fail):")
    try:
        result = concatenate("Hello, ", 42)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed with: {type(e).__name__}: {e}")
    
    # Test complex function
    print("\n6. Testing process_data (should succeed):")
    try:
        result = process_data([1, 2, 3, 4, 5], 2)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed with: {type(e).__name__}: {e}")
    
    # Wait a bit to let MeTTa process everything
    time.sleep(1)
    
    # Query MeTTa for insights (if monitor is available)
    try:
        print("\nQuerying MeTTa for insights:")
        
        # Get function recommendations
        print("\nFunction recommendations:")
        for func_name in ["divide_numbers", "get_item", "concatenate"]:
            recommendations = monitor.get_function_recommendations(func_name)
            print(f"\n  For {func_name}:")
            if recommendations:
                for rec in recommendations:
                    print(f"    - {rec['type']}: {rec['description']} (confidence: {rec['confidence']})")
            else:
                print("    No recommendations found")
        
        # Get error patterns
        print("\nError patterns:")
        for func_name in ["divide_numbers", "get_item", "concatenate"]:
            patterns = monitor.get_error_patterns(func_name)
            print(f"\n  For {func_name}:")
            if patterns:
                for pattern in patterns:
                    print(f"    - {pattern['error_type']}: {pattern['description']} (frequency: {pattern['frequency']})")
            else:
                print("    No error patterns found")
        
        # Direct query of MeTTa space
        print("\nDirect query of MeTTa space:")
        execution_count = monitor.query("(match &self (execution-start $id $func $time) $id)")
        print(f"  Number of execution records: {len(execution_count)}")
        
        error_count = monitor.query("(match &self (execution-error $exec $error $time) $error)")
        print(f"  Number of error records: {len(error_count)}")
        
    except (NameError, ImportError) as e:
        print(f"\nCouldn't query MeTTa: {e}")
    
    print("\nTests completed!")

if __name__ == "__main__":
    run_tests()