#!/usr/bin/env python3
"""
Simple test script for the static analyzer component.
"""

import os
import sys
from static_analyzer import decompose_file

def test_static_analysis():
    # Test file path - change this to your test file
    test_file = "example.py"
    
    if not os.path.exists(test_file):
        print(f"File not found: {test_file}")
        print("Creating a sample test file...")
        
        # Create a sample file
        with open(test_file, "w") as f:
            f.write("""
def divide_numbers(a, b):
    \"\"\"Divide a by b without any checks.\"\"\"
    return a / b  # Potential zero division

def format_user(name, age):
    \"\"\"Format user information.\"\"\"
    return name + ": " + age  # Potential type error

class DataProcessor:
    def __init__(self, name):
        self.name = name
        self.data = []
    
    def process_data(self, items):
        \"\"\"Process a list of items.\"\"\"
        result = ""
        for item in items:
            result += str(item) + ", "  # String concatenation in loop
        return result.rstrip(", ")
""")
        print(f"Created sample file: {test_file}")
    
    # Run static analysis
    print(f"Running static analysis on: {test_file}")
    result = decompose_file(test_file)
    
    if "error" in result and result["error"]:
        print(f"Analysis error: {result['error']}")
        return
    
    # Display results
    print(f"\nGenerated {len(result['metta_atoms'])} MeTTa atoms")
    
    # Display some sample atoms
    print("\nSample MeTTa atoms:")
    for atom in result['metta_atoms'][:10]:  # Show first 10 atoms
        print(f"  {atom}")
    
    # Display function info
    functions = [item for item in result.get('structure', []) if item.get('type') == 'function_def']
    print(f"\nDetected {len(functions)} functions:")
    for func in functions:
        print(f"  {func['name']} (lines {func['line_start']}-{func['line_end']})")
    
    # Display operations
    bin_ops = [item for item in result.get('structure', []) if item.get('type') == 'bin_op']
    print(f"\nDetected {len(bin_ops)} binary operations:")
    op_types = {}
    for op in bin_ops:
        op_type = f"{op['op']} ({op['left_type']}, {op['right_type']})"
        op_types[op_type] = op_types.get(op_type, 0) + 1
    
    for op_type, count in op_types.items():
        print(f"  {op_type}: {count}")
    
    # Save results to a file if needed
    if "--save" in sys.argv:
        import json
        output_file = "analysis_results.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    test_static_analysis()