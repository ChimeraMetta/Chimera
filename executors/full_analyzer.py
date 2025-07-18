from reflectors.static_analyzer import decompose_file
from reflectors.dynamic_monitor import DynamicMonitor
from reflectors.temporal_analyzer import TemporalCodeAnalyzer
import os
import re
import sys
from collections import defaultdict
from common.logging_utils import get_logger

# Configure logging
logger = get_logger("full_analyzer")

ONTOLOGY_PATH = "metta/code_ontology.metta"

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
        logger.error(f"Invalid path or not a Python file: {path}")

def analyze_file(file_path):
    """Analyze a single Python file and add to the ontology."""
    logger.info(f"Analyzing {file_path}...")
    
    # Run static analysis
    analysis_result = decompose_file(file_path)
    
    # Add analysis results to MeTTa
    if "metta_atoms" in analysis_result and analysis_result["metta_atoms"]:
        atoms_added = 0
        for atom_str in analysis_result["metta_atoms"]:
            try:
                monitor.add_atom(atom_str)
                atoms_added += 1
            except Exception as e:
                logger.error(f"Error adding atom {atom_str}: {e}")
                pass
                
        logger.info(f"Added {atoms_added}/{len(analysis_result['metta_atoms'])} atoms from {file_path}")
    else:
        logger.warning(f"No MeTTa atoms generated for {file_path}")

def find_function_relationships():
    """Find and analyze function call relationships."""
    logger.info("\n=== Function Call Relationships ===")
    
    # Get function calls with a more robust query pattern
    # The old query may have had scope matching issues
    results = monitor.query("""
        (match &self 
               (function-call $callee $args $scope $line)
               (function-def $caller $caller-scope $start $end)
               (and (= $scope $caller-scope)
                    (>= $line $start)
                    (<= $line $end))
               ($caller $callee))
    """)
    
    # If the first approach doesn't work, try an alternative query
    if not results:
        results = monitor.query("""
            (match &self 
                   (function-def $caller $caller-scope $start $end)
                   $caller_atom)
            (match &self 
                   (function-call $callee $args $scope $line)
                   $callee_atom)
            (and (>= $line $start)
                 (<= $line $end)
                 (= $scope $caller-scope))
            ($caller $callee)
        """)
    
    # If still no results, try with function-depends which might have been populated
    if not results:
        results = monitor.query("""
            (match &self (function-depends $caller $callee)
                  ($caller $callee))
        """)
    
    if results:
        logger.info(f"Found {len(results)} function call relationships")
        
        # Build caller -> callee map
        call_graph = {}
        reverse_graph = {}
        
        for result in results:
            try:
                # Handle different result formats by converting to string and parsing
                result_str = str(result)
                parts = result_str.strip('()').split()
                if len(parts) >= 2:
                    caller, callee = parts[0], parts[1]
                    
                    # Add to call graph
                    if caller not in call_graph:
                        call_graph[caller] = set()
                    call_graph[caller].add(callee)
                    
                    # Add to reverse graph
                    if callee not in reverse_graph:
                        reverse_graph[callee] = set()
                    reverse_graph[callee].add(caller)
            except Exception as e:
                logger.error(f"Error processing result {result}: {e}")
        
        # Display function call graph
        logger.info(f"\nFunction call relationships:")
        for caller, callees in call_graph.items():
            logger.info(f"- {caller} calls: {', '.join(callees)}")
        
        # Find high fan-in functions (called by many)
        high_fan_in = [(func, len(callers)) for func, callers in reverse_graph.items() if len(callers) > 1]
        high_fan_in.sort(key=lambda x: x[1], reverse=True)
        
        if high_fan_in:
            logger.info(f"\nMost called functions (high fan-in):")
            for func, count in high_fan_in[:10]:  # Show top 10
                logger.info(f"- {func}: called by {count} functions")
        
        # Find high fan-out functions (call many others)
        high_fan_out = [(caller, len(callees)) for caller, callees in call_graph.items() if len(callees) > 1]
        high_fan_out.sort(key=lambda x: x[1], reverse=True)
        
        if high_fan_out:
            logger.info(f"\nFunctions calling many others (high fan-out):")
            for func, count in high_fan_out[:10]:  # Show top 10
                logger.info(f"- {func}: calls {count} functions")
        
        # Additional debugging to show all function definitions and calls
        logger.warning(f"\nDebugging information:")
        
        func_defs = monitor.query("(match &self (function-def $name $scope $start $end) ($name $scope $start $end))")
        logger.info(f"Total function definitions: {len(func_defs)}")
        if len(func_defs) > 0 and len(func_defs) < 10:
            for f in func_defs:
                logger.info(f"  {f}")
        
        func_calls = monitor.query("(match &self (function-call $name $args $scope $line) ($name $scope $line))")
        logger.info(f"Total function calls: {len(func_calls)}")
        if len(func_calls) > 0 and len(func_calls) < 10:
            for c in func_calls:
                logger.info(f"  {c}")
    else:
        logger.warning(f"No function call relationships found. Adding diagnostic information:")
        
        # Diagnostic information
        func_defs = monitor.query("(match &self (function-def $name $scope $start $end) $name)")
        func_calls = monitor.query("(match &self (function-call $name $args $scope $line) $name)")
        deps = monitor.query("(match &self (function-depends $caller $callee) ($caller $callee))")
        
        logger.info(f"Function definitions found: {len(func_defs)}")
        logger.info(f"Function calls found: {len(func_calls)}")
        logger.info(f"Function dependencies found: {len(deps)}")
        
        # Show sample function definitions and calls for debugging
        if func_defs:
            logger.warning(f"\nSample function definitions:")
            for i, func in enumerate(func_defs[:5]):
                logger.info(f"  {func}")
        
        if func_calls:
            logger.warning(f"\nSample function calls:")
            for i, call in enumerate(func_calls[:5]):
                logger.info(f"  {call}")

def find_type_relationships():
    """Find and analyze type relationships between functions."""
    logger.info("\n=== Type Flow Relationships ===")
    
    # Get function return types
    return_types = monitor.query("(match &self (: $func (-> $params $return)) ($func $return))")
    
    # Get function parameter types
    param_types = monitor.query("(match &self (function-param $func $idx $name $type) ($func $name $type))")
    
    if return_types and param_types:
        # Build maps
        returns = {}
        params = {}
        
        for result in return_types:
            parts = str(result).split()
            if len(parts) >= 2:
                func, ret_type = parts[0], parts[1]
                returns[func] = ret_type
        
        for result in param_types:
            parts = str(result).split()
            if len(parts) >= 3:
                func, name, param_type = parts[0], parts[1], parts[2]
                if func not in params:
                    params[func] = []
                params[func].append((name, param_type))
        
        # Find type flows
        type_flows = []
        for source_func, ret_type in returns.items():
            for target_func, target_params in params.items():
                if source_func != target_func:  # Skip self-references
                    for param_name, param_type in target_params:
                        if ret_type == param_type:
                            type_flows.append((source_func, target_func, ret_type))
        
        if type_flows:
            logger.info(f"Found {len(type_flows)} potential type flows between functions")
            logger.info("\nPotential data flow paths:")
            for source, target, type_name in type_flows:
                logger.info(f"- {source} -> {target} (type: {type_name})")
        else:
            logger.info("No type flow relationships found")
        
        # Analyze type usage
        type_usage = {}
        for func, params_list in params.items():
            for name, type_name in params_list:
                if type_name not in type_usage:
                    type_usage[type_name] = 0
                type_usage[type_name] += 1
        
        for func, ret_type in returns.items():
            if ret_type not in type_usage:
                type_usage[ret_type] = 0
            type_usage[ret_type] += 1
        
        logger.info("\nType usage frequency:")
        for type_name, count in sorted(type_usage.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"- {type_name}: used {count} times")
    else:
        logger.info("Insufficient type information found")

def find_class_relationships():
    """Find and analyze class inheritance relationships."""
    logger.info("\n=== Class Relationships ===")
    
    # Get class inheritance relationships
    inheritance = monitor.query("(match &self (class-inherits $derived $base) ($derived $base))")
    
    if inheritance:
        logger.info(f"Found {len(inheritance)} class inheritance relationships")
        logger.info("\nClass inheritance:")
        for rel in inheritance:
            logger.info(f"- {rel}")
        
        # Build inheritance graph
        inheritance_graph = {}
        for rel in inheritance:
            parts = str(rel).split()
            if len(parts) >= 2:
                derived, base = parts[0], parts[1]
                if base not in inheritance_graph:
                    inheritance_graph[base] = set()
                inheritance_graph[base].add(derived)
        
        # Show hierarchy
        logger.info("\nClass hierarchy:")
        for base, derived_classes in inheritance_graph.items():
            logger.info(f"- {base} is extended by: {', '.join(derived_classes)}")
    else:
        logger.info("No class inheritance relationships found")

def find_module_relationships():
    """Find and analyze module import relationships."""
    logger.info("\n=== Module Relationships ===")
    
    # Get import relationships
    imports = monitor.query("(match &self (import $module $scope $line) ($scope $module))")
    
    if imports:
        logger.info(f"Found {len(imports)} direct module imports")
        
        # Build scope -> imports map
        scope_imports = {}
        for imp in imports:
            parts = str(imp).split()
            if len(parts) >= 2:
                scope, module = parts[0], parts[1]
                if scope not in scope_imports:
                    scope_imports[scope] = set()
                scope_imports[scope].add(module)
        
        logger.info("\nModule dependencies by scope:")
        for scope, modules in scope_imports.items():
            logger.info(f"- {scope} imports: {', '.join(modules)}")
    else:
        logger.info("No direct module imports found")
    
    # Get from-import relationships
    from_imports = monitor.query("(match &self (import-from $module $name $scope $line) ($scope $module $name))")
    
    if from_imports:
        logger.info(f"\nFound {len(from_imports)} from-type imports")
        
        # Build scope -> (module, name) map
        scope_from_imports = {}
        for imp in from_imports:
            parts = str(imp).split()
            if len(parts) >= 3:
                scope, module, name = parts[0], parts[1], parts[2]
                if scope not in scope_from_imports:
                    scope_from_imports[scope] = []
                scope_from_imports[scope].append((module, name))
        
        logger.info("\nModule component imports by scope:")
        for scope, imports in scope_from_imports.items():
            logger.info(f"- {scope} imports:")
            for module, name in imports:
                logger.info(f"  - {name} from {module}")
    else:
        logger.info("No from-type imports found")

def find_operation_patterns():
    """Find and analyze operation patterns in the code."""
    logger.info("\n=== Operation Patterns ===")
    
    # Get binary operations
    bin_ops = monitor.query("(match &self (bin-op $op $left $right $scope $line) ($op $left $right))")
    
    if bin_ops:
        logger.info(f"Found {len(bin_ops)} binary operations")
        
        # Count operations by type
        op_counts = {}
        for op in bin_ops:
            parts = str(op).split()
            if len(parts) >= 1:
                op_type = parts[0]
                if op_type not in op_counts:
                    op_counts[op_type] = 0
                op_counts[op_type] += 1
        
        logger.info("\nOperation frequency:")
        for op_type, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"- {op_type}: {count} times")
        
        # Find common type patterns in operations
        type_patterns = {}
        for op in bin_ops:
            parts = str(op).split()
            if len(parts) >= 3:
                op_type, left_type, right_type = parts[0], parts[1], parts[2]
                key = f"{op_type}({left_type}, {right_type})"
                if key not in type_patterns:
                    type_patterns[key] = 0
                type_patterns[key] += 1
        
        logger.info("\nCommon operation patterns:")
        for pattern, count in sorted(type_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:  # Top 10
            logger.info(f"- {pattern}: {count} times")
    else:
        logger.info("No binary operations found")

def analyze_type_safety():
    """Analyze code for potential type safety issues."""
    logger.info("\n=== Type Safety Analysis ===")
    
    # Check for binary operation type mismatches
    bin_op_mismatches = monitor.query("""
        (match &self 
               (binary-op-type-mismatch $op $left-type $right-type $scope $line)
               ($op $left-type $right-type $scope $line))
    """)
    
    if bin_op_mismatches:
        logger.warning(f"Found {len(bin_op_mismatches)} potential binary operation type mismatches:")
        for mismatch in bin_op_mismatches:
            parts = str(mismatch).strip('()').split()
            if len(parts) >= 5:
                op, left, right, scope, line = parts[:5]
                logger.warning(f"- Line {line}: {op} operation between {left} and {right} in {scope}")
    
    # Check for function parameter type mismatches
    param_mismatches = monitor.query("""
        (match &self 
               (function-param-type-mismatch $func $idx $expected $actual $scope $line)
               ($func $idx $expected $actual $scope $line))
    """)
    
    if param_mismatches:
        logger.warning(f"\nFound {len(param_mismatches)} potential function parameter type mismatches:")
        for mismatch in param_mismatches:
            parts = str(mismatch).strip('()').split()
            if len(parts) >= 6:
                func, idx, expected, actual, scope, line = parts[:6]
                logger.warning(f"- Line {line}: Function {func} parameter {idx} expects {expected} but got {actual}")
    
    # Check for potential division by zero
    div_zero = monitor.query("""
        (match &self 
               (potential-division-by-zero $scope $line)
               ($scope $line))
    """)
    
    if div_zero:
        logger.warning(f"\nFound {len(div_zero)} potential division by zero operations:")
        for div in div_zero:
            parts = str(div).strip('()').split()
            if len(parts) >= 2:
                scope, line = parts[:2]
                logger.warning(f"- Line {line}: Division operation with potential zero divisor in {scope}")
    
    # Check for return type mismatches
    return_mismatches = monitor.query("""
        (match &self 
               (return-type-mismatch $func $expected $actual $line)
               ($func $expected $actual $line))
    """)
    
    if return_mismatches:
        logger.warning(f"\nFound {len(return_mismatches)} potential return type mismatches:")
        for mismatch in return_mismatches:
            parts = str(mismatch).strip('()').split()
            if len(parts) >= 4:
                func, expected, actual, line = parts[:4]
                logger.warning(f"- Line {line}: Function {func} declares return type {expected} but returns {actual}")
    
    # Check for potential null dereferences
    null_derefs = monitor.query("""
        (match &self 
               (potential-null-dereference $scope $line)
               ($scope $line))
    """)
    
    if null_derefs:
        logger.warning(f"\nFound {len(null_derefs)} potential null/None dereferences:")
        for deref in null_derefs:
            parts = str(deref).strip('()').split()
            if len(parts) >= 2:
                scope, line = parts[:2]
                logger.warning(f"- Line {line}: Potential None/null value used in function call in {scope}")
    
    if not (bin_op_mismatches or param_mismatches or div_zero or return_mismatches or null_derefs):
        logger.info("No type safety issues detected.")


def analyze_structural_patterns():
    """Analyze structural patterns in the codebase."""
    logger.info("\n=== Structural Patterns ===")
    
    # Get counts for key elements
    functions = monitor.query("(match &self (function-def $name $scope $start $end) $name)")
    classes = monitor.query("(match &self (class-def $name $scope $start $end) $name)")
    loops = monitor.query("(match &self (loop-pattern $id $type $scope $line) $type)")
    variables = monitor.query("(match &self (variable-assign $name $scope $line) $name)")
    
    # Count by scope
    scopes = {}
    for func in monitor.query("(match &self (function-def $name $scope $start $end) $scope)"):
        scope = str(func)
        if scope not in scopes:
            scopes[scope] = {"functions": 0, "classes": 0, "variables": 0}
        scopes[scope]["functions"] += 1
    
    for cls in monitor.query("(match &self (class-def $name $scope $start $end) $scope)"):
        scope = str(cls)
        if scope not in scopes:
            scopes[scope] = {"functions": 0, "classes": 0, "variables": 0}
        scopes[scope]["classes"] += 1
    
    for var in monitor.query("(match &self (variable-assign $name $scope $line) $scope)"):
        scope = str(var)
        if scope not in scopes:
            scopes[scope] = {"functions": 0, "classes": 0, "variables": 0}
        scopes[scope]["variables"] += 1
    
    logger.info(f"\nCodebase structure summary:")
    logger.info(f"- Functions: {len(functions)}")
    logger.info(f"- Classes: {len(classes)}")
    logger.info(f"- Variables: {len(variables)}")
    logger.info(f"- Loops: {len(loops)}")
    logger.info(f"- Scopes: {len(scopes)}")
    
    # Determine overall architecture
    if len(classes) == 0 and len(functions) > 0:
        logger.info("\nArchitectural pattern: Primarily Functional")
    elif len(classes) > 0 and len(functions) / max(1, len(classes)) < 2:
        logger.info("\nArchitectural pattern: Primarily Object-Oriented")
    else:
        logger.info("\nArchitectural pattern: Mixed (OO and Functional)")
    
    # Analyze module structure
    if len(scopes) > 1:
        logger.info("\nModule structure:")
        for scope, counts in scopes.items():
            total = counts["functions"] + counts["classes"] + counts["variables"]
            if total > 0:
                logger.info(f"- {scope}: {counts['functions']} functions, {counts['classes']} classes, {counts['variables']} variables")

# Function to analyze temporal code evolution
# This is a direct replacement for the analyze_temporal_evolution function in full_analyzer.py

def analyze_temporal_evolution(repo_path, monitor=None):
    """Analyze temporal code evolution using Git history."""
    logger.info("\n=== Temporal Code Evolution Analysis ===")
    
    if not monitor:
        # Create monitor if not provided
        monitor = DynamicMonitor()
        monitor.load_metta_rules("ontology.metta")
    
    # Create temporal analyzer
    temporal_analyzer = TemporalCodeAnalyzer(repo_path, monitor)
    
    # Analyze Git history
    if not temporal_analyzer.analyze_history(max_commits=20):  # Limit to 20 commits for faster processing
        logger.warning("Could not analyze Git history. Skipping temporal analysis.")
        return
        
    # DIRECT FIX: Rewrite the approach completely to use more direct queries and careful result handling
    
    try:
        # === Find functions with frequent changes ===
        logger.info("\nFunctions with frequent changes:")
        
        # Use a more direct query to find function names first
        func_names = monitor.query("""(match &self (function-signature-at $func $commit $_) $func)""")
        
        # Convert to a set of unique function names
        unique_funcs = set()
        for func_atom in func_names:
            try:
                func_name = str(func_atom)
                if not func_name.startswith("("):  # Skip complex expressions
                    unique_funcs.add(func_name)
            except:
                continue
        
        # For each function, count its changes directly
        frequent_changes = []
        for func_name in unique_funcs:
            try:
                # Count commits where this function appears - directly in Python
                count_query = monitor.query(f"""(match &self (function-signature-at "{func_name}" $commit $_) $commit)""")
                change_count = len(count_query)
                
                if change_count > 3:  # Threshold for "frequent"
                    frequent_changes.append((func_name, change_count))
            except Exception as e:
                logger.error(f"Error analyzing changes for {func_name}: {e}")
        
        # Sort by frequency
        frequent_changes.sort(key=lambda x: x[1], reverse=True)
        
        # Display results
        if frequent_changes:
            logger.info(f"Found {len(frequent_changes)} functions with frequent changes:")
            for func, count in frequent_changes[:10]:  # Show top 10
                logger.info(f"- {func}: changed {count} times")
        else:
            logger.info("No functions with frequent changes found.")
            
        # === Find functions with complexity growth ===
        logger.info("\nFunctions with complexity growth:")
        
        complexity_growth = []
        for func_name in unique_funcs:
            try:
                # Get all complexity measurements for this function
                complexity_query = monitor.query(f"""(match &self (function-complexity-at "{func_name}" $commit $complexity) ($commit $complexity))""")
                
                if len(complexity_query) < 2:
                    continue  # Need at least 2 measurements to calculate growth
                
                # Extract and parse commit IDs and complexity values
                measurements = []
                for result in complexity_query:
                    try:
                        # Handle different result formats
                        result_str = str(result)
                        if result_str.startswith("(") and result_str.endswith(")"):
                            # Parse (commit_id complexity) format
                            content = result_str[1:-1].strip()
                            parts = content.split(None, 1)
                            if len(parts) == 2:
                                commit_id = parts[0]
                                try:
                                    complexity = int(parts[1])
                                    measurements.append((commit_id, complexity))
                                except ValueError:
                                    pass
                    except:
                        continue
                
                if len(measurements) < 2:
                    continue
                    
                # Simplistic approach: just check if last complexity > first complexity
                # For real timestamp-based sorting, we would need to use the commit timestamps
                measurements.sort()  # Alphabetic sort by commit ID as a rough proxy
                first_complexity = measurements[0][1]
                last_complexity = measurements[-1][1]
                
                growth = last_complexity - first_complexity
                if growth > 0:
                    complexity_growth.append((func_name, growth))
            except Exception as e:
                logger.error(f"Error analyzing complexity for {func_name}: {e}")
        
        # Sort by growth
        complexity_growth.sort(key=lambda x: x[1], reverse=True)
        
        # Display results
        if complexity_growth:
            logger.info(f"Found {len(complexity_growth)} functions that grew in complexity:")
            for func, growth in complexity_growth[:10]:  # Show top 10
                logger.info(f"- {func}: complexity increased by {growth}")
        else:
            logger.info("No functions with complexity growth found.")
            
        # === Find co-evolving functions ===
        logger.info("\nCo-evolving function pairs:")
        
        # This is computationally expensive, so limit to most frequently changing functions
        top_funcs = [f for f, _ in frequent_changes[:20]]
        
        co_evolving = []
        processed_pairs = set()
        
        for i, func1 in enumerate(top_funcs):
            for func2 in top_funcs[i+1:]:
                pair_key = tuple(sorted([func1, func2]))
                if pair_key in processed_pairs:
                    continue
                    
                processed_pairs.add(pair_key)
                
                try:
                    # Get commits where func1 appears
                    func1_commits = set(str(c) for c in monitor.query(f"""(match &self (function-signature-at "{func1}" $commit $_) $commit)"""))
                    
                    # Get commits where func2 appears
                    func2_commits = set(str(c) for c in monitor.query(f"""(match &self (function-signature-at "{func2}" $commit $_) $commit)"""))
                    
                    # Find common commits
                    common_commits = func1_commits.intersection(func2_commits)
                    
                    # Calculate co-change ratio
                    if len(func1_commits) > 0 and len(func2_commits) > 0:
                        co_change_ratio = len(common_commits) / min(len(func1_commits), len(func2_commits))
                        
                        if co_change_ratio > 0.7:  # Threshold for co-evolution
                            co_evolving.append((func1, func2, co_change_ratio))
                except Exception as e:
                    logger.error(f"Error analyzing co-evolution for {func1}/{func2}: {e}")
        
        # Sort by co-change ratio
        co_evolving.sort(key=lambda x: x[2], reverse=True)
        
        # Display results
        if co_evolving:
            logger.info(f"Found {len(co_evolving)} co-evolving function pairs:")
            for func1, func2, ratio in co_evolving[:10]:  # Show top 10
                logger.info(f"- {func1} and {func2}: co-change ratio {ratio:.2f}")
        else:
            logger.info("No co-evolving function pairs found.")
            
        # === Identify potential hotspots ===
        logger.info("\nPotential code hotspots:")
        
        # Combine frequency and complexity to find hotspots
        hotspots = []
        
        func_to_freq = dict(frequent_changes)
        func_to_growth = dict(complexity_growth)
        
        # Score functions based on frequency and complexity growth
        for func in set(func_to_freq.keys()).union(func_to_growth.keys()):
            freq = func_to_freq.get(func, 0)
            growth = func_to_growth.get(func, 0)
            
            # Simple scoring algorithm
            score = freq * 0.6 + growth * 0.4
            
            if score > 3:  # Threshold for hotspot
                confidence = "high" if score > 8 else "medium"
                hotspots.append((func, score, confidence))
        
        # Sort by score
        hotspots.sort(key=lambda x: x[1], reverse=True)
        
        # Display results
        if hotspots:
            logger.info(f"Found {len(hotspots)} potential code hotspots:")
            for func, score, confidence in hotspots[:10]:  # Show top 10
                logger.info(f"- {func}: score {score:.2f} ({confidence} confidence)")
        else:
            logger.info("No potential code hotspots detected.")
        
    except Exception as e:
        logger.error(f"Error in temporal analysis: {e}")
        import traceback
        traceback.print_exc()

def analyze_function_complexity(file_path):
    """Analyze complexity of functions in a file with our enhanced metrics."""
    logger.info(f"Analyzing complexity in {file_path}...")
    
    # Decompose the file
    result = decompose_file(file_path)
    
    if "error" in result and result["error"]:
        logger.error(f"Error: {result['error']}")
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
    logger.info(f"\n=== Function Complexity Analysis ===")
    logger.info("Function complexity ranking:")
    for i, (func_name, func_info) in enumerate(sorted_functions):
        logger.info(f"{i+1}. {func_name}: score {func_info['score']:.1f} ({func_info['operations']} operations, {func_info['loops']} loops, {func_info['calls']} calls)")
        if i > 20:  # Only show top 20 functions
            logger.info("(... and more functions)")
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
    
    logger.info(f"\n=== Complex Functions Detected ===")
    if complex_funcs:
        for i, (func_name, func_info) in enumerate(complex_funcs):
            logger.info(f"{i+1}. {func_name}: score {func_info['score']:.1f} ({func_info['operations']} operations, {func_info['loops']} loops, {func_info['calls']} calls)")
    else:
        logger.info("No complex functions detected")

def analyze_domain_concepts():
    """Analyze potential domain concepts in the codebase."""
    logger.info("\n=== Domain Concept Analysis ===")
    
    # Find non-standard types (potential domain types)
    standard_types = {"String", "Number", "Bool", "List", "Dict", "Tuple", "Set", "Any", "None"}
    types = monitor.query("(match &self (function-param $func $idx $name $type) $type)")
    types += monitor.query("(match &self (: $func (-> $params $return)) $return)")
    
    # Filter to unique types
    unique_types = set()
    for t in types:
        type_str = str(t)
        unique_types.add(type_str)
    
    # Filter to domain types
    domain_types = [t for t in unique_types if t not in standard_types and t and t[0] not in "$" and "(" not in t]
    
    if domain_types:
        logger.info(f"Found {len(domain_types)} potential domain types:")
        for dtype in sorted(domain_types):
            logger.info(f"- {dtype}")
        
        # Try to find functions operating on these types
        logger.info("\nFunctions working with domain types:")
        for dtype in domain_types:
            try:
                query = f"(match &self (function-param $func $idx $name {dtype}) $func)"
                funcs = monitor.query(query)
                if funcs:
                    logger.info(f"- {dtype} is used by: {', '.join(str(f) for f in funcs)}")
            except Exception as e:
                logger.info(f"Error querying functions for type {dtype}: {e}")
    else:
        logger.info("No domain-specific types found")
    
    # Look for potential domain concepts in naming
    domain_concepts = set()
    # Look for domain terms in function names
    for func in monitor.query("(match &self (function-def $name $scope $start $end) $name)"):
        name = str(func)
        if "_" in name:
            parts = name.split("_")
            for part in parts:
                if len(part) > 3 and part not in standard_types:
                    domain_concepts.add(part)
    
    # Look for domain terms in variable names
    for var in monitor.query("(match &self (variable-assign $name $scope $line) $name)"):
        name = str(var)
        if "_" in name:
            parts = name.split("_")
            for part in parts:
                if len(part) > 3 and part not in standard_types:
                    domain_concepts.add(part)
    
    if domain_concepts:
        logger.info(f"\nPotential domain concepts from naming patterns:")
        for concept in sorted(domain_concepts):
            logger.info(f"- {concept}")
    else:
        logger.info("\nNo clear domain concepts found in naming patterns")

def analyze_function_call_relationships(file_path):
    """Analyze function call relationships with proper handling of class methods."""
    logger.info(f"Analyzing function call relationships in {file_path}...")
    
    # Decompose the file
    result = decompose_file(file_path)
    
    if "error" in result and result["error"]:
        logger.error(f"Error: {result['error']}")
        return
    
    atoms = result["metta_atoms"]
    
    # Extract function definitions with full scope information
    function_defs = {}
    for atom in atoms:
        if atom.startswith("(function-def "):
            # Parse the function definition atom
            match = re.match(r'\(function-def\s+(\S+)\s+(.*?)\s+(\d+)\s+(\d+)\)', atom)
            if match:
                func_name = match.group(1)
                scope = match.group(2).strip()
                line_start = int(match.group(3))
                line_end = int(match.group(4))
                
                # Create a qualified name that includes the class for methods
                qualified_name = func_name
                if scope and scope != "global":
                    # If it's a class method, include the class name
                    if scope.startswith("class-"):
                        class_name = scope.replace("class-", "", 1)
                        qualified_name = f"{class_name}.{func_name}"
                
                function_defs[qualified_name] = {
                    "name": func_name,
                    "scope": scope,
                    "line_start": line_start,
                    "line_end": line_end,
                    "calls": []
                }
    
    # Extract direct function calls
    direct_calls = []
    for atom in atoms:
        if atom.startswith("(direct-call "):
            # Parse the direct call atom
            match = re.match(r'\(direct-call\s+(\S+)\s+(\S+)\s+(\d+)\)', atom)
            if match:
                caller = match.group(1)
                callee = match.group(2)
                line = int(match.group(3))
                direct_calls.append((caller, callee, line))
    
    # Extract function calls with their scopes
    function_calls = []
    for atom in atoms:
        if atom.startswith("(function-call "):
            # Parse the function call atom
            match = re.match(r'\(function-call\s+(\S+)\s+(\d+)\s+(.*?)\s+(\d+)\)', atom)
            if match:
                callee = match.group(1)
                args = int(match.group(2))
                scope = match.group(3).strip()
                line = int(match.group(4))
                function_calls.append((callee, scope, line))
    
    # Associate function calls with their callers
    call_relationships = defaultdict(list)
    
    # First, process direct calls which are already associated with callers
    for caller, callee, line in direct_calls:
        # Find the qualified name for the caller
        caller_qualified = None
        for qname, func_info in function_defs.items():
            if func_info["name"] == caller:
                caller_qualified = qname
                break
        
        # Find the qualified name for the callee
        callee_qualified = None
        for qname, func_info in function_defs.items():
            if func_info["name"] == callee or qname == callee:
                callee_qualified = qname
                break
        
        if caller_qualified and callee_qualified:
            call_relationships[caller_qualified].append(callee_qualified)
    
    # Then, process general function calls by determining their callers based on scope
    for callee, scope, line in function_calls:
        # Find which function this call belongs to
        for qname, func_info in function_defs.items():
            call_scope = scope
            func_scope = func_info["scope"]
            
            # Check if the call is within this function's scope and lines
            is_in_scope = (call_scope == func_scope or 
                          call_scope.startswith(func_scope) or 
                          (func_scope == "global" and not call_scope.startswith("class-")))
            
            is_in_lines = (line >= func_info["line_start"] and line <= func_info["line_end"])
            
            if is_in_scope and is_in_lines:
                # Find the qualified name for the callee
                callee_qualified = None
                for callee_qname, callee_info in function_defs.items():
                    if callee_info["name"] == callee or callee_qname == callee:
                        callee_qualified = callee_qname
                        break
                
                if callee_qualified:
                    call_relationships[qname].append(callee_qualified)
    
    # Print function call relationships
    logger.info(f"\n=== Function Call Relationships ===")
    logger.info(f"Found {len(call_relationships)} function call relationships")
    logger.info(f"Function call relationships:")
    
    for caller, callees in sorted(call_relationships.items()):
        # Remove duplicates while preserving order
        unique_callees = []
        for callee in callees:
            if callee not in unique_callees:
                unique_callees.append(callee)
        
        logger.info(f"- {caller} calls: {', '.join(unique_callees)}")
    
    # Print debugging information
    logger.warning(f"Debugging information:")
    logger.info(f"Total function definitions: {len(function_defs)}")
    for qname, info in sorted(function_defs.items()):
        logger.info(f"  ({info['name']} {info['scope']} {info['line_start']} {info['line_end']})")
    
    logger.info(f"Total function calls: {len(function_calls)}")
    
    # Identify potential call chains
    logger.info(f"\n=== Function Call Chains ===")
    call_chains = find_call_chains(call_relationships)
    if call_chains:
        logger.info(f"Found {len(call_chains)} significant call chains:")
        for i, chain in enumerate(call_chains[:10], 1):  # Show top 10
            logger.info(f"{i}. {' : '.join(chain)}")
    else:
        logger.info("No significant call chains found")

def find_call_chains(call_relationships, min_length=3):
    """Find significant call chains in the codebase."""
    chains = []
    
    def dfs(current, path, visited):
        """Depth-first search to find call chains."""
        if current in visited:
            return
        
        new_path = path + [current]
        visited.add(current)
        
        if len(new_path) >= min_length:
            chains.append(new_path)
        
        for callee in call_relationships.get(current, []):
            dfs(callee, new_path, visited.copy())
    
    # Start DFS from each function
    for caller in call_relationships:
        dfs(caller, [], set())
    
    # Sort chains by length (longest first)
    return sorted(chains, key=len, reverse=True)

if __name__ == "__main__":
    # Initialize MeTTa monitor
    monitor = DynamicMonitor()
    
    # Create basic relationship rules
    # Load the MeTTa reasoning rules
    monitor.load_metta_rules(ONTOLOGY_PATH)
    
    if len(sys.argv) < 2:
        logger.info("Usage: python full_analyzer.py <path_to_file_or_directory>")
        sys.exit(1)
    
    # Path to analyze
    path = sys.argv[1]
    
    # Run analysis
    analyze_codebase(path)
    analyze_type_safety()

    analyze_temporal_evolution(path, monitor)
    analyze_function_complexity(path)
    
    # Perform relationship analysis
    # find_function_relationships()
    analyze_function_call_relationships(path)
    find_type_relationships()
    find_class_relationships()
    find_module_relationships()
    find_operation_patterns()
    analyze_structural_patterns()
    analyze_domain_concepts()