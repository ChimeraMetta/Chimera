"""
File Processor module for the Hybrid Code Analyzer.
Handles file operations, function extraction, and code modifications.
"""

import ast
import os
from typing import List, Dict, Any, Optional


def extract_functions(file_path: str) -> List[Dict[str, Any]]:
    """Extract all function definitions from a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        functions = []
        
        # Helper function to extract all functions
        def extract_functions_recursive(node, class_name=None):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    full_name = f"{class_name}.{item.name}" if class_name else item.name
                    line_start = item.lineno
                    line_end = get_last_line(item)
                    functions.append({
                        'name': item.name,
                        'full_name': full_name,
                        'line_start': line_start,
                        'line_end': line_end,
                        'is_method': class_name is not None,
                        'class_name': class_name
                    })
                elif isinstance(item, ast.ClassDef):
                    # Get methods within the class
                    extract_functions_recursive(item, item.name)
        
        def get_last_line(node):
            """Get the last line of a node."""
            if not hasattr(node, 'body') or not node.body:
                return node.lineno
            
            last_line = node.lineno
            for item in node.body:
                if hasattr(item, 'lineno'):
                    last_line = max(last_line, item.lineno)
                    
                    # Recursively get the last line for compound statements
                    if hasattr(item, 'body'):
                        last_line = max(last_line, get_last_line(item))
            
            return last_line
        
        # Extract all functions and methods
        extract_functions_recursive(tree)
        return functions
            
    except Exception as e:
        print(f"Failed to extract functions from {file_path}: {e}")
        return []


def add_monitoring_to_file(file_path: str, functions_to_monitor: List[Dict[str, Any]], 
                          context: Optional[str] = None, auto_fix: bool = False) -> bool:
    """
    Add dynamic monitoring to selected functions in a file.
    
    Args:
        file_path: Path to the Python file
        functions_to_monitor: List of function details to monitor
        context: Optional context string for monitoring
        auto_fix: Whether to enable auto-fixing
        
    Returns:
        Boolean indicating success
    """
    if not functions_to_monitor:
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Split the content into lines for easier manipulation
        lines = content.split('\n')
        
        # Sort functions by line number in reverse order to avoid position shifts
        # when modifying the file
        functions_to_monitor.sort(key=lambda f: f['line_start'], reverse=True)
        
        # Track insertions
        line_adjustments = 0
        modified = False
        
        for func in functions_to_monitor:
            # Find the line with the function definition
            line_idx = func['line_start'] - 1
            
            # Check if the function already has @hybrid_transform
            if line_idx > 0 and '@hybrid_transform' in lines[line_idx - 1]:
                print(f"Function {func['name']} already has @hybrid_transform")
                continue
            
            # Create the decorator string
            if context:
                decorator = f'@hybrid_transform(context="{context}", auto_fix={str(auto_fix).lower()})'
            else:
                decorator = f'@hybrid_transform(auto_fix={str(auto_fix).lower()})'
            
            # Insert the decorator before the function definition
            lines.insert(line_idx, decorator)
            line_adjustments += 1
            modified = True
        
        # If we made any changes, add the import if it doesn't exist
        if modified:
            # Look for existing imports and insert hybrid_transform
            import_line = "from dynamic_monitor import hybrid_transform"
            
            # Check if import is already present
            if import_line not in content:
                # Try to insert after existing imports
                inserted = False
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        # Keep searching until no more imports
                        if i + 1 < len(lines) and (lines[i + 1].startswith('import ') or lines[i + 1].startswith('from ')):
                            continue
                        lines.insert(i + 1, "\n" + import_line)
                        inserted = True
                        break
                
                # If no imports found, insert at the beginning of the file
                if not inserted:
                    lines.insert(0, import_line + "\n")
            
            # Save the modified file
            modified_content = '\n'.join(lines)
            
            with open(file_path, 'w') as f:
                f.write(modified_content)
            
            return True
        
        return False
        
    except Exception as e:
        print(f"Failed to add monitoring to {file_path}: {e}")
        return False


def get_file_summary(file_path: str) -> Dict[str, Any]:
    """
    Get a summary of a Python file's contents.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Dictionary with file summary information
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        stats = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'size_bytes': os.path.getsize(file_path),
            'line_count': len(content.split('\n')),
            'functions': [],
            'classes': []
        }
        
        # Parse the file
        tree = ast.parse(content)
        
        # Count top-level definitions
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                stats['functions'].append({
                    'name': node.name,
                    'line': node.lineno
                })
            elif isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'line': node.lineno,
                    'methods': []
                }
                
                # Count methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_info['methods'].append({
                            'name': item.name,
                            'line': item.lineno
                        })
                
                stats['classes'].append(class_info)
        
        return stats
        
    except Exception as e:
        print(f"Failed to get file summary for {file_path}: {e}")
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'error': str(e)
        }


def check_monitoring_status(file_path: str) -> Dict[str, Any]:
    """
    Check which functions in a file already have monitoring.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Dictionary with monitoring status for each function
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Get all functions
        functions = extract_functions(file_path)
        lines = content.split('\n')
        
        # Check each function for monitoring
        for func in functions:
            line_idx = func['line_start'] - 1
            
            # Check if the function has @hybrid_transform
            if line_idx > 0 and '@hybrid_transform' in lines[line_idx - 1]:
                func['has_monitoring'] = True
                
                # Extract monitoring settings
                decorator_line = lines[line_idx - 1]
                if 'context=' in decorator_line:
                    import re
                    context_match = re.search(r'context="([^"]+)"', decorator_line)
                    if context_match:
                        func['context'] = context_match.group(1)
                
                if 'auto_fix=' in decorator_line:
                    func['auto_fix'] = 'auto_fix=True' in decorator_line
            else:
                func['has_monitoring'] = False
        
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'functions': functions,
            'has_import': 'from dynamic_monitor import hybrid_transform' in content
        }
            
    except Exception as e:
        print(f"Failed to check monitoring status for {file_path}: {e}")
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'error': str(e),
            'functions': []
        }


def remove_monitoring_from_file(file_path: str, functions: Optional[List[Dict[str, Any]]] = None) -> bool:
    """
    Remove dynamic monitoring from functions in a file.
    
    Args:
        file_path: Path to the Python file
        functions: Optional list of specific functions to remove monitoring from
                   If None, removes monitoring from all functions
        
    Returns:
        Boolean indicating success
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        lines = content.split('\n')
        modified = False
        
        # Get monitoring status
        status = check_monitoring_status(file_path)
        
        # If specific functions are provided, filter the list
        funcs_to_process = status['functions']
        if functions:
            func_names = {f['name'] for f in functions}
            funcs_to_process = [f for f in funcs_to_process if f['name'] in func_names]
        
        # Sort by line number in reverse to avoid position shifts
        funcs_to_process.sort(key=lambda f: f['line_start'], reverse=True)
        
        # Remove decorators
        for func in funcs_to_process:
            if func.get('has_monitoring', False):
                line_idx = func['line_start'] - 1
                if line_idx > 0 and '@hybrid_transform' in lines[line_idx - 1]:
                    # Remove the decorator line
                    lines.pop(line_idx - 1)
                    modified = True
        
        # Check if we should remove the import
        if modified and status.get('has_import', False):
            # Check if any functions still have monitoring
            if not any(f.get('has_monitoring', False) for f in status['functions'] 
                    if functions is None or f['name'] not in {func['name'] for func in functions}):
                # Remove the import line
                import_line = "from dynamic_monitor import hybrid_transform"
                if import_line in content:
                    for i, line in enumerate(lines):
                        if import_line in line:
                            lines.pop(i)
                            # Remove any extra blank line
                            if i < len(lines) and lines[i].strip() == '':
                                lines.pop(i)
                            break
        
        # Save changes if modified
        if modified:
            modified_content = '\n'.join(lines)
            with open(file_path, 'w') as f:
                f.write(modified_content)
                
        return modified
        
    except Exception as e:
        print(f"Failed to remove monitoring from {file_path}: {e}")
        return False