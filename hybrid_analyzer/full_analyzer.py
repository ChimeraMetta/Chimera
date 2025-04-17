from static_analyzer import decompose_file, decompose_source
from dynamic_monitor import DynamicMonitor, hybrid_transform
import os
import sys

def analyze_codebase(path):
    """Analyze a Python file or directory of Python files."""
    if os.path.isfile(path) and path.endswith('.py'):
        # Analyze a single file
        analyze_file(path)
    elif os.path.isdir(path):
        # Analyze all Python files in directory
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.py'):
                    analyze_file(os.path.join(root, file))
    else:
        print(f"Invalid path or not a Python file: {path}")

def analyze_file(file_path):
    """Analyze a single Python file."""
    print(f"Analyzing {file_path}...")
    
    # Run static analysis
    analysis_result = decompose_file(file_path)
    
    # Add analysis results to MeTTa
    if "metta_atoms" in analysis_result and analysis_result["metta_atoms"]:
        for atom in analysis_result["metta_atoms"]:
            monitor.metta_space.add_atom(atom)
        print(f"Added {len(analysis_result['metta_atoms'])} atoms to MeTTa for {file_path}")
    else:
        print(f"Warning: No MeTTa atoms generated for {file_path}")

def get_codebase_insights():
    """Query MeTTa for insights about the analyzed codebase."""
    
    # Get architectural insights
    print("\nArchitectural Insights:")
    insights = monitor.query_metta("(match &self (architectural-insight $description $confidence) ($description $confidence))")
    for insight in insights:
        # Parse the result (format depends on your MeTTa implementation)
        parts = insight.strip("()").split(" ", 1)
        if len(parts) >= 2:
            description, confidence = parts[0], parts[1]
            print(f"- {description} (confidence: {confidence})")
    
    # Find error-prone functions
    print("\nError-Prone Functions:")
    error_prone = monitor.query_metta("(find-error-prone-functions 0.2)")
    if error_prone:
        for func in error_prone:
            print(f"- {func}")
    else:
        print("No error-prone functions detected.")
    
    # Find functions with high complexity
    print("\nComplex Functions:")
    complex_funcs = monitor.query_metta("(match &self (code-quality-issue $func complexity $description $confidence) ($func $description $confidence))")
    for func_info in complex_funcs:
        # Parse the result
        parts = func_info.strip("()").split(" ", 2)
        if len(parts) >= 3:
            func, description, confidence = parts[0], parts[1], parts[2]
            print(f"- {func}: {description} (confidence: {confidence})")
    
    # Get type safety issues
    print("\nType Safety Issues:")
    type_issues = monitor.query_metta("(match &self (type-safety-issue $func $description $confidence) ($func $description $confidence))")
    for issue in type_issues:
        # Parse the result
        parts = issue.strip("()").split(" ", 2)
        if len(parts) >= 3:
            func, description, confidence = parts[0], parts[1], parts[2]
            print(f"- {func}: {description} (confidence: {confidence})")

def get_function_recommendations(function_name):
    """Get recommendations for a specific function."""
    print(f"\nRecommendations for {function_name}:")
    
    # Get recommendations from MeTTa
    recs = monitor.get_function_recommendations(function_name)
    
    if recs:
        for rec in recs:
            print(f"- {rec['type']}: {rec['description']} (confidence: {rec['confidence']})")
    else:
        print("No recommendations available.")
        
    # Get related functions
    related = monitor.query_metta(f"(find-related-functions {function_name} (list))")
    if related:
        print("\nRelated functions:")
        for func in related:
            print(f"- {func}")

def visualize_module_dependencies():
    """Create a visualization of module dependencies."""
    # Query MeTTa for module dependencies
    modules = monitor.query_metta("(match &self (module-depends $module $dependency) ($module $dependency))")
    
    # This would typically use a visualization library like networkx and matplotlib
    # For example:
    # import networkx as nx
    # import matplotlib.pyplot as plt
    # G = nx.DiGraph()
    # for edge in modules:
    #     parts = edge.strip("()").split()
    #     if len(parts) >= 2:
    #         source, target = parts[0], parts[1]
    #         G.add_edge(source, target)
    # nx.draw(G, with_labels=True)
    # plt.savefig("module_dependencies.png")
    
    print(f"Found {len(modules)} module dependencies")

def visualize_class_hierarchy():
    """Create a visualization of class hierarchies."""
    # Query MeTTa for class hierarchies
    hierarchies = monitor.query_metta("(match &self (class-hierarchy $base $derived) ($base $derived))")
    
    # Similar to module dependencies, you would use a visualization library here
    
    print(f"Found {len(hierarchies)} class inheritance relationships")

def analyze_function_dependencies():
    """Analyze function dependencies to find potential bottlenecks."""
    # Find functions with high fan-in (many callers)
    high_fan_in = monitor.query_metta("(match &self (high-fan-in-functions 5 $funcs) $funcs)")
    
    if high_fan_in:
        print("\nFunctions with high fan-in (potential bottlenecks):")
        for func in high_fan_in:
            print(f"- {func}")
            
            # Get callers for each high fan-in function
            callers = monitor.query_metta(f"(find-calling-functions {func} (list))")
            if callers:
                print(f"  Called by: {', '.join(callers)}")
    else:
        print("\nNo high fan-in functions detected.")

def find_error_patterns():
    """Find common error patterns across the codebase."""
    error_patterns = monitor.query_metta("(match &self (function-error-pattern $func $error $freq $desc) ($func $error $freq $desc))")
    
    if error_patterns:
        print("\nCommon Error Patterns:")
        for pattern in error_patterns:
            # Parse the result
            parts = pattern.strip("()").split(" ", 3)
            if len(parts) >= 4:
                func, error, freq, desc = parts[0], parts[1], parts[2], parts[3]
                print(f"- {func}: {error} (frequency: {freq}) - {desc}")
    else:
        print("\nNo common error patterns detected.")

def suggest_refactorings():
    """Suggest potential refactorings based on code analysis."""
    # Find functions with high complexity
    complex_funcs = monitor.query_metta("(match &self (code-quality-issue $func complexity $desc $conf) ($func $desc $conf))")
    
    if complex_funcs:
        print("\nRefactoring Suggestions:")
        for func_info in complex_funcs:
            # Parse the result
            parts = func_info.strip("()").split(" ", 2)
            if len(parts) >= 3:
                func = parts[0]
                
                # Check if function has many loops
                loop_query = monitor.query_metta(f"(match &self (loop-pattern $id $type $scope $line) (contains-scope $scope {func}))")
                if len(loop_query) > 2:
                    print(f"- {func}: Consider extracting loop logic into separate functions")
                
                # Check if function has many parameters
                param_query = monitor.query_metta(f"(match &self (function-param {func} $idx $name $type) $name)")
                if len(param_query) > 4:
                    print(f"- {func}: Has many parameters ({len(param_query)}). Consider grouping related parameters into a class")
    else:
        print("\nNo refactoring suggestions available.")

def analyze_type_usage():
    """Analyze type usage across the codebase."""
    # Find functions with type issues
    type_issues = monitor.query_metta("(match &self (type-safety-issue $func $desc $conf) ($func $desc $conf))")
    
    if type_issues:
        print("\nType Usage Analysis:")
        for issue in type_issues:
            # Parse the result
            parts = issue.strip("()").split(" ", 2)
            if len(parts) >= 3:
                func = parts[0]
                
                # Check for type conversion patterns
                conversions = monitor.query_metta(f"(match &self (execution-error $exec $error-id $time) (execution-start $exec {func} $start) (error-type $error-id TypeError))")
                if conversions:
                    # Find parameters with potential type issues
                    params = monitor.query_metta(f"(match &self (input-param $exec $idx $name $type) (execution-error $exec $error-id $time) $name)")
                    for param in params:
                        print(f"- {func}: Parameter '{param}' may need explicit type conversion")
    else:
        print("\nNo type usage issues detected.")

if __name__ == "__main__":
    # Initialize MeTTa monitor
    monitor = DynamicMonitor()

    # Load the MeTTa reasoning rules
    monitor.load_metta_rules("ontology.metta")
    
    if len(sys.argv) < 2:
        print("Usage: python code_analyzer.py <path_to_file_or_directory> [function_name]")
        sys.exit(1)
    
    # Path to analyze
    path = sys.argv[1]
    
    # Run analysis
    analyze_codebase(path)
    
    # Get codebase insights
    get_codebase_insights()
    
    # Analyze function dependencies
    analyze_function_dependencies()
    
    # # Find error patterns
    # find_error_patterns()
    
    # # Suggest refactorings
    # suggest_refactorings()
    
    # # Analyze type usage
    # analyze_type_usage()
    
    # # Visualize dependencies
    # visualize_module_dependencies()
    # visualize_class_hierarchy()
    
    # Optionally get recommendations for specific function
    if len(sys.argv) > 2:
        function_name = sys.argv[2]
        get_function_recommendations(function_name)