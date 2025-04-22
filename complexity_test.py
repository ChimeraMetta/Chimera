import sys
import os
from static_analyzer import decompose_file, CodeDecomposer

# Add the fixes from analyzer-fix.py to CodeDecomposer
# [Copy the methods from the first fix here before running]

def analyze_function_complexity(file_path):
    """Analyze complexity of functions in a file with our enhanced metrics."""
    print(f"Analyzing complexity in {file_path}...")
    
    # Decompose the file
    result = decompose_file(file_path)
    
    if "error" in result and result["error"]:
        print(f"Error: {result['error']}")
        return
    
    atoms = result["metta_atoms"]
    
    # Extract function definitions
    function_defs = {}
    for atom in atoms:
        if atom.startswith("(function-def "):
            parts = atom.strip("()").split()
            if len(parts) >= 5:
                func_name = parts[1]
                scope = " ".join(parts[2:-2])
                line_start = int(parts[-2])
                line_end = int(parts[-1])
                function_defs[func_name] = {
                    "scope": scope,
                    "line_start": line_start,
                    "line_end": line_end,
                    "operations": 0,
                    "loops": 0,
                    "calls": 0
                }
    
    # Count operations (bin_op)
    for atom in atoms:
        if atom.startswith("(bin-op "):
            parts = atom.strip("()").split()
            if len(parts) >= 6:
                op = parts[1]
                scope = " ".join(parts[4:-1])
                line = int(parts[-1])
                
                # Find which function this operation belongs to
                for func_name, func_info in function_defs.items():
                    if (scope == func_info["scope"] or 
                        scope.startswith(func_info["scope"]) or 
                        func_info["scope"] == "global") and \
                       line >= func_info["line_start"] and \
                       line <= func_info["line_end"]:
                        func_info["operations"] += 1
    
    # Count loops (both explicit and implicit)
    for atom in atoms:
        if atom.startswith("(loop-pattern ") or atom.startswith("(implicit-loop "):
            parts = atom.strip("()").split()
            if len(parts) >= 5:
                loop_id = parts[1]
                loop_type = parts[2]
                scope = " ".join(parts[3:-1])
                line = int(parts[-1])
                
                # Find which function this loop belongs to
                for func_name, func_info in function_defs.items():
                    if (scope == func_info["scope"] or 
                        scope.startswith(func_info["scope"]) or 
                        func_info["scope"] == "global") and \
                       line >= func_info["line_start"] and \
                       line <= func_info["line_end"]:
                        func_info["loops"] += 1
    
    # Count function calls
    for atom in atoms:
        if atom.startswith("(function-call "):
            parts = atom.strip("()").split()
            if len(parts) >= 5:
                callee = parts[1]
                args = parts[2]
                scope = " ".join(parts[3:-1])
                line = int(parts[-1])
                
                # Find which function this call belongs to
                for func_name, func_info in function_defs.items():
                    if (scope == func_info["scope"] or 
                        scope.startswith(func_info["scope"]) or 
                        func_info["scope"] == "global") and \
                       line >= func_info["line_start"] and \
                       line <= func_info["line_end"]:
                        func_info["calls"] += 1
    
    # Calculate overall complexity scores
    for func_name, func_info in function_defs.items():
        # Weighted score: operations + 3*loops + 0.5*calls
        func_info["score"] = (
            func_info["operations"] + 
            3 * func_info["loops"] + 
            0.5 * func_info["calls"]
        )
    
    # Sort functions by complexity score
    sorted_functions = sorted(
        function_defs.items(), 
        key=lambda x: x[1]["score"], 
        reverse=True
    )
    
    # Print complexity ranking
    print("\n=== Function Complexity Analysis ===")
    print("Function complexity ranking:")
    for i, (func_name, func_info) in enumerate(sorted_functions):
        print(f"{i+1}. {func_name}: score {func_info['score']:.1f} ({func_info['operations']} operations, {func_info['loops']} loops, {func_info['calls']} calls)")
        if i > 20:  # Only show top 20 functions
            print("(... and more functions)")
            break
    
    # Identify complex functions based on criteria
    complex_funcs = []
    for func_name, func_info in function_defs.items():
        if (func_info["operations"] > 10 or 
            func_info["loops"] > 2 or 
            func_info["score"] > 15):
            complex_funcs.append((func_name, func_info))
    
    # Sort complex functions by score
    complex_funcs.sort(key=lambda x: x[1]["score"], reverse=True)
    
    print("\n=== Complex Functions Detected ===")
    if complex_funcs:
        for i, (func_name, func_info) in enumerate(complex_funcs):
            print(f"{i+1}. {func_name}: score {func_info['score']:.1f} ({func_info['operations']} operations, {func_info['loops']} loops, {func_info['calls']} calls)")
    else:
        print("No complex functions detected")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_complexity.py <python_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    analyze_function_complexity(file_path)