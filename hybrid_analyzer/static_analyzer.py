import ast
import os
import sys
import inspect
from typing import Dict, List, Any, Set, Tuple, Optional

class CodeDecomposer(ast.NodeVisitor):
    """
    Extract type-theoretic structure from Python code for MeTTa reasoning.
    Enhanced to include layered donor system information and ontological relationships.
    """
    
    def __init__(self):
        # Structural atoms for MeTTa
        self.atoms = []
        
        # Track scopes and structure
        self.current_function = None
        self.current_class = None
        self.scope_stack = []
        self.function_calls = {}
        self.variables = {}
        
        # Track code patterns for donor system
        self.string_operations = []
        self.arithmetic_operations = []
        self.loop_patterns = []
        self.function_patterns = []
        
        # Track ontological relationships
        self.module_relationships = {}
        self.class_hierarchies = {}
        self.function_dependencies = {}
        
        # Line number tracking for code location
        self.line_mapping = {}
    
    def visit_Module(self, node):
        """Process an entire module."""
        self.atoms.append({
            "type": "module",
            "name": getattr(node, 'name', 'unnamed_module'),
            "docstring": ast.get_docstring(node)
        })
        
        # Visit all statements in the module
        for stmt in node.body:
            self.visit(stmt)
    
    def visit_FunctionDef(self, node):
        """Process function definitions."""
        prev_function = self.current_function
        self.current_function = node.name
        self.scope_stack.append(f"function:{node.name}")
        
        # Extract parameters and return type
        params = []
        return_type = None
        
        # Get return annotation if present
        if node.returns:
            return_type = self._get_type_from_annotation(node.returns)
        
        # Get parameter annotations
        for arg in node.args.args:
            param_type = None
            if arg.annotation:
                param_type = self._get_type_from_annotation(arg.annotation)
            params.append((arg.arg, param_type))
        
        # Create type signature atom
        self.atoms.append({
            "type": "function_def",
            "name": node.name,
            "params": params,
            "return_type": return_type,
            "docstring": ast.get_docstring(node),
            "scope": ".".join(self.scope_stack[:-1]) if len(self.scope_stack) > 1 else "global",
            "line_start": node.lineno,
            "line_end": self._get_last_line(node)
        })
        
        # Add to line mapping
        self.line_mapping[node.lineno] = {
            "type": "function_def",
            "name": node.name
        }
        
        # Record function pattern
        self.function_patterns.append({
            "name": node.name,
            "params": len(params),
            "has_return": return_type is not None,
            "scope": ".".join(self.scope_stack[:-1]) if len(self.scope_stack) > 1 else "global"
        })
        
        # Visit function body
        for stmt in node.body:
            self.visit(stmt)
        
        # Restore previous state
        self.current_function = prev_function
        self.scope_stack.pop()
    
    def visit_ClassDef(self, node):
        """Process class definitions."""
        prev_class = self.current_class
        self.current_class = node.name
        self.scope_stack.append(f"class:{node.name}")
        
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._get_name(base.value)}.{base.attr}")
        
        # Create class atom
        self.atoms.append({
            "type": "class_def",
            "name": node.name,
            "bases": bases,
            "docstring": ast.get_docstring(node),
            "scope": ".".join(self.scope_stack[:-1]) if len(self.scope_stack) > 1 else "global",
            "line_start": node.lineno,
            "line_end": self._get_last_line(node)
        })
        
        # Add to line mapping
        self.line_mapping[node.lineno] = {
            "type": "class_def",
            "name": node.name
        }
        
        # Add to class hierarchy
        for base in bases:
            if base not in self.class_hierarchies:
                self.class_hierarchies[base] = []
            self.class_hierarchies[base].append(node.name)
        
        # Visit class body
        for stmt in node.body:
            self.visit(stmt)
        
        # Restore previous state
        self.current_class = prev_class
        self.scope_stack.pop()
    
    def visit_Assign(self, node):
        """Process variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                var_type = self._infer_type_from_value(node.value)
                
                # Add variable atom
                current_scope = ".".join(self.scope_stack)
                self.atoms.append({
                    "type": "variable_assign",
                    "name": var_name,
                    "inferred_type": var_type,
                    "scope": current_scope,
                    "line": node.lineno
                })
                
                # Track in variables dictionary
                if current_scope not in self.variables:
                    self.variables[current_scope] = {}
                self.variables[current_scope][var_name] = var_type
                
                # Add to line mapping
                self.line_mapping[node.lineno] = {
                    "type": "variable_assign",
                    "name": var_name
                }
        
        # Continue visiting children
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node):
        """Process annotated assignments."""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            var_type = self._get_type_from_annotation(node.annotation)
            
            # Add variable atom
            current_scope = ".".join(self.scope_stack)
            self.atoms.append({
                "type": "variable_ann_assign",
                "name": var_name,
                "declared_type": var_type,
                "scope": current_scope,
                "line": node.lineno
            })
            
            # Track in variables dictionary
            if current_scope not in self.variables:
                self.variables[current_scope] = {}
            self.variables[current_scope][var_name] = var_type
            
            # Add to line mapping
            self.line_mapping[node.lineno] = {
                "type": "variable_ann_assign",
                "name": var_name
            }
        
        # Continue visiting children
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Process function calls."""
        func_name = self._get_name(node.func)
        
        # Track dependencies
        if self.current_function:
            if self.current_function not in self.function_dependencies:
                self.function_dependencies[self.current_function] = set()
            self.function_dependencies[self.current_function].add(func_name)
        
        # Create function call atom
        current_scope = ".".join(self.scope_stack)
        self.atoms.append({
            "type": "function_call",
            "name": func_name,
            "args": len(node.args),
            "scope": current_scope,
            "line": node.lineno
        })
        
        # Track arg types for donor system
        arg_types = []
        for arg in node.args:
            arg_type = self._infer_type_from_value(arg)
            arg_types.append(arg_type or "Any")
        
        if func_name not in self.function_calls:
            self.function_calls[func_name] = []
        self.function_calls[func_name].append(arg_types)
        
        # Add to line mapping
        self.line_mapping[node.lineno] = {
            "type": "function_call",
            "name": func_name
        }
        
        # Continue visiting children
        self.generic_visit(node)
    
    def visit_BinOp(self, node):
        """Process binary operations."""
        left_type = self._infer_type_from_value(node.left)
        right_type = self._infer_type_from_value(node.right)
        op_type = type(node.op).__name__
        
        # Create binary operation atom
        current_scope = ".".join(self.scope_stack)
        self.atoms.append({
            "type": "bin_op",
            "op": op_type,
            "left_type": left_type or "Any",
            "right_type": right_type or "Any", 
            "scope": current_scope,
            "line": node.lineno
        })
        
        # Track patterns for donor system
        if left_type == "String" or right_type == "String":
            self.string_operations.append({
                "op": op_type,
                "left_type": left_type,
                "right_type": right_type,
                "line": node.lineno
            })
        elif left_type == "Number" and right_type == "Number":
            self.arithmetic_operations.append({
                "op": op_type,
                "line": node.lineno
            })
        
        # Add to line mapping
        self.line_mapping[node.lineno] = {
            "type": "bin_op",
            "op": op_type
        }
        
        # Continue visiting children
        self.generic_visit(node)
    
    def visit_For(self, node):
        """Process for loops."""
        # Create for loop atom
        current_scope = ".".join(self.scope_stack)
        self.atoms.append({
            "type": "for_loop",
            "target": self._get_name(node.target),
            "scope": current_scope,
            "line_start": node.lineno,
            "line_end": self._get_last_line(node)
        })
        
        # Track pattern for donor system
        self.loop_patterns.append({
            "type": "for",
            "scope": current_scope,
            "line": node.lineno
        })
        
        # Add to line mapping
        self.line_mapping[node.lineno] = {
            "type": "for_loop",
            "target": self._get_name(node.target)
        }
        
        # Continue visiting children
        self.generic_visit(node)
    
    def visit_While(self, node):
        """Process while loops."""
        # Create while loop atom
        current_scope = ".".join(self.scope_stack)
        self.atoms.append({
            "type": "while_loop",
            "scope": current_scope,
            "line_start": node.lineno,
            "line_end": self._get_last_line(node)
        })
        
        # Track pattern for donor system
        self.loop_patterns.append({
            "type": "while",
            "scope": current_scope,
            "line": node.lineno
        })
        
        # Add to line mapping
        self.line_mapping[node.lineno] = {
            "type": "while_loop"
        }
        
        # Continue visiting children
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Process import statements."""
        current_scope = ".".join(self.scope_stack)
        
        for name in node.names:
            # Create import atom
            self.atoms.append({
                "type": "import",
                "name": name.name,
                "alias": name.asname,
                "scope": current_scope,
                "line": node.lineno
            })
            
            # Track module relationships
            if current_scope not in self.module_relationships:
                self.module_relationships[current_scope] = []
            self.module_relationships[current_scope].append(name.name)
            
            # Add to line mapping
            self.line_mapping[node.lineno] = {
                "type": "import",
                "name": name.name
            }
        
        # Continue visiting children
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Process from...import statements."""
        current_scope = ".".join(self.scope_stack)
        
        for name in node.names:
            # Create import atom
            self.atoms.append({
                "type": "import_from",
                "module": node.module,
                "name": name.name,
                "alias": name.asname,
                "scope": current_scope,
                "line": node.lineno
            })
            
            # Track module relationships
            if current_scope not in self.module_relationships:
                self.module_relationships[current_scope] = []
            self.module_relationships[current_scope].append(f"{node.module}.{name.name}")
            
            # Add to line mapping
            self.line_mapping[node.lineno] = {
                "type": "import_from",
                "module": node.module,
                "name": name.name
            }
        
        # Continue visiting children
        self.generic_visit(node)
    
    def _get_name(self, node):
        """Extract name from a node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Tuple):
            return tuple(self._get_name(elt) for elt in node.elts)
        return "Unknown"
    
    def _get_type_from_annotation(self, node):
        """Extract type from a type annotation."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            base_type = self._get_name(node.value)
            if isinstance(node.slice, ast.Index):  # Python 3.8 and below
                if isinstance(node.slice.value, ast.Name):
                    return f"{base_type}[{node.slice.value.id}]"
                elif isinstance(node.slice.value, ast.Tuple):
                    params = []
                    for elt in node.slice.value.elts:
                        if isinstance(elt, ast.Name):
                            params.append(elt.id)
                    return f"{base_type}[{', '.join(params)}]"
            elif isinstance(node.slice, ast.Name):  # Python 3.9+
                return f"{base_type}[{node.slice.id}]"
            elif isinstance(node.slice, ast.Tuple):
                params = []
                for elt in node.slice.elts:
                    if isinstance(elt, ast.Name):
                        params.append(elt.id)
                return f"{base_type}[{', '.join(params)}]"
        return "Any"
    
    def _infer_type_from_value(self, node):
        """Infer type from a value expression."""
        if node is None:
            return "None"
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return "String"
            elif isinstance(node.value, int) or isinstance(node.value, float):
                return "Number"
            elif isinstance(node.value, bool):
                return "Bool"
            elif node.value is None:
                return "None"
        elif isinstance(node, ast.Str):  # For older Python versions
            return "String"
        elif isinstance(node, ast.Num):  # For older Python versions
            return "Number"
        elif isinstance(node, ast.List):
            return "List"
        elif isinstance(node, ast.Dict):
            return "Dict"
        elif isinstance(node, ast.Set):
            return "Set"
        elif isinstance(node, ast.Tuple):
            return "Tuple"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                # Map built-in functions to return types
                builtin_types = {
                    'int': 'Number',
                    'float': 'Number',
                    'str': 'String',
                    'list': 'List',
                    'dict': 'Dict',
                    'set': 'Set',
                    'tuple': 'Tuple',
                    'bool': 'Bool',
                    'len': 'Number',
                    'sum': 'Number',
                }
                if node.func.id in builtin_types:
                    return builtin_types[node.func.id]
                return node.func.id  # Use function name as type
        elif isinstance(node, ast.Name):
            if node.id == 'True' or node.id == 'False':
                return "Bool"
            elif node.id == 'None':
                return "None"
            
            # Try to find variable in current scope
            current_scope = ".".join(self.scope_stack)
            if current_scope in self.variables and node.id in self.variables[current_scope]:
                return self.variables[current_scope][node.id]
        elif isinstance(node, ast.BinOp):
            left_type = self._infer_type_from_value(node.left)
            right_type = self._infer_type_from_value(node.right)
            
            # Simple type inference for binary operations
            if isinstance(node.op, ast.Add):
                if left_type == "String" or right_type == "String":
                    return "String"
                elif left_type == "Number" and right_type == "Number":
                    return "Number"
            elif isinstance(node.op, (ast.Sub, ast.Mult, ast.Div)):
                if left_type == "Number" and right_type == "Number":
                    return "Number"
                elif isinstance(node.op, ast.Mult) and (
                    (left_type == "String" and right_type == "Number") or
                    (left_type == "Number" and right_type == "String")
                ):
                    return "String"
        
        return "Any"
    
    def _get_last_line(self, node):
        """Get the last line of a node's body."""
        if not hasattr(node, 'body') or not node.body:
            return node.lineno
        
        # Find the last line in the body
        last_line = node.lineno
        for stmt in node.body:
            if hasattr(stmt, 'body'):
                # For compound statements, get the last line recursively
                stmt_last_line = self._get_last_line(stmt)
                last_line = max(last_line, stmt_last_line)
            else:
                # For simple statements, use the lineno
                last_line = max(last_line, getattr(stmt, 'lineno', node.lineno))
        
        return last_line


def convert_to_metta_atoms(decomposer: CodeDecomposer) -> List[str]:
    """Convert decomposed code structure to MeTTa atoms with proper atomic representation."""
    metta_atoms = []
    
    # Convert basic structure atoms
    for atom in decomposer.atoms:
        if atom["type"] == "function_def":
            # Function definition atoms
            params_list = []
            for _, param_type in atom["params"]:
                params_list.append(param_type or "Any")
            
            params_str = " ".join(params_list)
            return_str = atom["return_type"] or "Any"
            
            # Create proper type signature - names as atoms, not strings
            metta_atoms.append(f"(: {atom['name']} (-> {params_str} {return_str}))")
            
            # Create function definition with scope as a proper atom, not a string
            # Replace colons with hyphens for better MeTTa compatibility
            scope_atoms = [s.replace(':', '-') for s in atom['scope'].split('.')]
            scope_expr = " ".join(scope_atoms)
            metta_atoms.append(f"(function-def {atom['name']} {scope_expr} {atom['line_start']} {atom['line_end']})")
            
            # Add parameters with proper atomic representation
            for i, (param_name, param_type) in enumerate(atom["params"]):
                param_type_atom = param_type or "Any"
                metta_atoms.append(f"(function-param {atom['name']} {i} {param_name} {param_type_atom})")
        
        elif atom["type"] == "class_def":
            # Class definition atoms
            metta_atoms.append(f"(: {atom['name']} Type)")
            
            # Create class definition with scope as proper atoms - replace colons with hyphens
            scope_atoms = [s.replace(':', '-') for s in atom['scope'].split('.')]
            scope_expr = " ".join(scope_atoms)
            metta_atoms.append(f"(class-def {atom['name']} {scope_expr} {atom['line_start']} {atom['line_end']})")
            
            # Add base classes as atoms
            for base in atom["bases"]:
                metta_atoms.append(f"(class-inherits {atom['name']} {base})")
        
        elif atom["type"] == "variable_assign" or atom["type"] == "variable_ann_assign":
            # Variable assignment atoms
            var_type = atom.get("declared_type") or atom.get("inferred_type") or "Any"
            
            # Create scope path as atoms - replace colons with hyphens
            scope_atoms = [s.replace(':', '-') for s in atom['scope'].split('.')]
            var_path = " ".join(scope_atoms + [atom['name']])
            
            metta_atoms.append(f"(: {var_path} {var_type})")
            
            # Variable assignment with scope as atoms
            scope_expr = " ".join(scope_atoms)
            metta_atoms.append(f"(variable-assign {atom['name']} {scope_expr} {atom['line']})")
        
        elif atom["type"] == "function_call":
            # Function call atoms with scope as atoms - replace colons with hyphens
            scope_atoms = [s.replace(':', '-') for s in atom['scope'].split('.')]
            scope_expr = " ".join(scope_atoms)
            metta_atoms.append(f"(function-call {atom['name']} {atom['args']} {scope_expr} {atom['line']})")
        
        elif atom["type"] == "bin_op":
            # Binary operation atoms with scope as atoms - replace colons with hyphens
            scope_atoms = [s.replace(':', '-') for s in atom['scope'].split('.')]
            scope_expr = " ".join(scope_atoms)
            metta_atoms.append(f"(bin-op {atom['op']} {atom['left_type']} {atom['right_type']} {scope_expr} {atom['line']})")
        
        elif atom["type"] == "import" or atom["type"] == "import_from":
            # Import atoms with scope as atoms - replace colons with hyphens
            scope_atoms = [s.replace(':', '-') for s in atom['scope'].split('.')]
            scope_expr = " ".join(scope_atoms)
            
            if atom["type"] == "import":
                metta_atoms.append(f"(import {atom['name']} {scope_expr} {atom['line']})")
            else:
                metta_atoms.append(f"(import-from {atom['module']} {atom['name']} {scope_expr} {atom['line']})")
    
    # Add ontological relationships
    
    # Module relationships
    for scope, modules in decomposer.module_relationships.items():
        scope_atoms = [s.replace(':', '-') for s in scope.split('.')]
        scope_expr = " ".join(scope_atoms)
        
        for module in modules:
            metta_atoms.append(f"(module-depends {scope_expr} {module})")
    
    # Class hierarchies
    for base, derived in decomposer.class_hierarchies.items():
        for cls in derived:
            metta_atoms.append(f"(class-hierarchy {base} {cls})")
    
    # Function dependencies
    for func, calls in decomposer.function_dependencies.items():
        for called_func in calls:
            metta_atoms.append(f"(function-depends {func} {called_func})")
    
    # Add pattern information for donor system
    
    # String operations patterns
    for i, op in enumerate(decomposer.string_operations):
        metta_atoms.append(f"(string-op-pattern {i} {op['op']} {op.get('left_type', 'Any')} {op.get('right_type', 'Any')} {op['line']})")
    
    # Arithmetic operations patterns
    for i, op in enumerate(decomposer.arithmetic_operations):
        metta_atoms.append(f"(arithmetic-op-pattern {i} {op['op']} {op['line']})")
    
    # Loop patterns
    for i, loop in enumerate(decomposer.loop_patterns):
        # Convert scope to atoms - replace colons with hyphens
        scope_atoms = [s.replace(':', '-') for s in loop['scope'].split('.')]
        scope_expr = " ".join(scope_atoms)
        
        metta_atoms.append(f"(loop-pattern {i} {loop['type']} {scope_expr} {loop['line']})")
    
    # Function patterns
    for i, func in enumerate(decomposer.function_patterns):
        has_return_str = "True" if func["has_return"] else "False"
        
        # Convert scope to atoms - replace colons with hyphens
        scope_atoms = [s.replace(':', '-') for s in func['scope'].split('.')]
        scope_expr = " ".join(scope_atoms)
        
        metta_atoms.append(f"(function-pattern {i} {func['name']} {func['params']} {has_return_str} {scope_expr})")
    
    return metta_atoms


def decompose_file(file_path: str) -> Dict:
    """Decompose a Python file into a type-theoretic structure."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        return decompose_source(source)
    except Exception as e:
        print(f"Error decomposing file {file_path}: {e}")
        return {"metta_atoms": [], "error": str(e)}


def decompose_source(source: str) -> Dict:
    """Decompose Python source code into a type-theoretic structure."""
    try:
        tree = ast.parse(source)
        decomposer = CodeDecomposer()
        decomposer.visit(tree)
        
        metta_atoms = convert_to_metta_atoms(decomposer)
        
        return {
            "metta_atoms": metta_atoms,
            "structure": decomposer.atoms,
            "function_calls": decomposer.function_calls,
            "variables": decomposer.variables,
            "module_relationships": decomposer.module_relationships,
            "class_hierarchies": decomposer.class_hierarchies,
            "function_dependencies": decomposer.function_dependencies,
            "line_mapping": decomposer.line_mapping
        }
    except SyntaxError as e:
        return {
            "metta_atoms": [],
            "error": f"Syntax error at line {e.lineno}, column {e.offset}: {e.msg}"
        }
    except Exception as e:
        return {
            "metta_atoms": [],
            "error": str(e)
        }


def decompose_function(func) -> Dict:
    """Decompose a Python function into a type-theoretic structure."""
    try:
        source = inspect.getsource(func)
        return decompose_source(source)
    except Exception as e:
        return {
            "metta_atoms": [],
            "error": f"Failed to decompose function {func.__name__}: {e}"
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python static_analyzer.py <python_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    result = decompose_file(file_path)
    
    if "error" in result and result["error"]:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    print(f"Successfully decomposed {file_path}")
    print(f"Generated {len(result['metta_atoms'])} MeTTa atoms")
    
    # Output atoms to a file if requested
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        with open(output_file, "w") as f:
            for atom in result["metta_atoms"]:
                f.write(atom + "\n")
        print(f"Atoms written to {output_file}")