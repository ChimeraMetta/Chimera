import ast

class TypeExtractor(ast.NodeVisitor):
    """Extract type information from Python AST."""
    
    def __init__(self):
        self.type_atoms = []  # Will store MeTTa-compatible atoms
        self.current_function = None
        self.current_class = None
        self.current_scope = []
        self.function_calls = {}  # Track function calls and their argument types
        
    def visit_FunctionDef(self, node):
        prev_function = self.current_function
        self.current_function = node.name
        self.current_scope.append(f"function:{node.name}")
        
        # Extract function signature
        args = []
        return_type = None
        
        # Handle return annotation
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return_type = node.returns.id
            elif isinstance(node.returns, ast.Constant):
                return_type = str(node.returns.value)
            elif isinstance(node.returns, ast.Attribute):
                return_type = f"{node.returns.value.id}.{node.returns.attr}"
            elif isinstance(node.returns, ast.Subscript):
                # Handle generic types like List[int]
                return_type = self._extract_subscript_type(node.returns)
                
        # Handle arguments and their annotations
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = None
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_type = arg.annotation.id
                elif isinstance(arg.annotation, ast.Attribute):
                    arg_type = f"{arg.annotation.value.id}.{arg.annotation.attr}"
                elif isinstance(arg.annotation, ast.Subscript):
                    arg_type = self._extract_subscript_type(arg.annotation)
            args.append((arg_name, arg_type))
        
        # Create MeTTa atom for function signature
        arg_types_str = " ".join([arg_type or 'Any' for _, arg_type in args])
        self.type_atoms.append(f"(: {node.name} (-> {arg_types_str} {return_type or 'Any'}))")
        
        # Visit function body
        for stmt in node.body:
            self.visit(stmt)
            
        self.current_function = prev_function
        self.current_scope.pop()
    
    def visit_Assign(self, node):
        # Extract variable assignments and infer types if possible
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                var_type = self._infer_type_from_value(node.value)
                if var_type:
                    scope_prefix = '.'.join(self.current_scope) if self.current_scope else ""
                    full_name = f"{scope_prefix}.{var_name}" if scope_prefix else var_name
                    self.type_atoms.append(f"(: {full_name} {var_type})")
        
        # Continue visiting children
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node):
        # Handle annotated assignments directly
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            var_type = None
            
            if isinstance(node.annotation, ast.Name):
                var_type = node.annotation.id
            elif isinstance(node.annotation, ast.Attribute):
                var_type = f"{node.annotation.value.id}.{node.annotation.attr}"
            elif isinstance(node.annotation, ast.Subscript):
                var_type = self._extract_subscript_type(node.annotation)
                
            if var_type:
                scope_prefix = '.'.join(self.current_scope) if self.current_scope else ""
                full_name = f"{scope_prefix}.{var_name}" if scope_prefix else var_name
                self.type_atoms.append(f"(: {full_name} {var_type})")
        
        # Continue visiting children
        self.generic_visit(node)
    
    def visit_Call(self, node):
        # Track function calls to help with type inference
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            arg_types = []
            
            for arg in node.args:
                arg_type = self._infer_type_from_value(arg)
                arg_types.append(arg_type or 'Any')
            
            if func_name not in self.function_calls:
                self.function_calls[func_name] = []
            self.function_calls[func_name].append(arg_types)
        
        # Continue visiting children
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        prev_class = self.current_class
        self.current_class = node.name
        self.current_scope.append(f"class:{node.name}")
        
        # Create MeTTa atom for class
        self.type_atoms.append(f"(: {node.name} Type)")
        
        # Visit class body
        for stmt in node.body:
            self.visit(stmt)
            
        self.current_class = prev_class
        self.current_scope.pop()
    
    def _extract_subscript_type(self, node):
        """Extract type from subscript annotations like List[int]."""
        if isinstance(node.value, ast.Name):
            base_type = node.value.id
            if isinstance(node.slice, ast.Index):
                # Python 3.8 and below
                if isinstance(node.slice.value, ast.Name):
                    return f"{base_type}[{node.slice.value.id}]"
                elif isinstance(node.slice.value, ast.Tuple):
                    params = []
                    for elt in node.slice.value.elts:
                        if isinstance(elt, ast.Name):
                            params.append(elt.id)
                    return f"{base_type}[{', '.join(params)}]"
            elif isinstance(node.slice, ast.Name):
                # Python 3.9+
                return f"{base_type}[{node.slice.id}]"
            elif isinstance(node.slice, ast.Tuple):
                params = []
                for elt in node.slice.elts:
                    if isinstance(elt, ast.Name):
                        params.append(elt.id)
                return f"{base_type}[{', '.join(params)}]"
        return "Any"
    
    def _infer_type_from_value(self, value_node):
        """Infer type from a value expression."""
        if isinstance(value_node, ast.Constant):
            if isinstance(value_node.value, str):
                return "String"
            elif isinstance(value_node.value, int):
                return "Number"
            elif isinstance(value_node.value, float):
                return "Number"
            elif isinstance(value_node.value, bool):
                return "Bool"
            elif value_node.value is None:
                return "None"
        elif isinstance(value_node, ast.Str):  # For older Python versions
            return "String" 
        elif isinstance(value_node, ast.Num):  # For older Python versions
            return "Number"
        elif isinstance(value_node, ast.List):
            return "List"
        elif isinstance(value_node, ast.Dict):
            return "Dict"
        elif isinstance(value_node, ast.Set):
            return "Set"
        elif isinstance(value_node, ast.Tuple):
            return "Tuple"
        elif isinstance(value_node, ast.Call):
            if isinstance(value_node.func, ast.Name):
                func_name = value_node.func.id
                # Map built-in functions to their return types
                builtin_return_types = {
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
                if func_name in builtin_return_types:
                    return builtin_return_types[func_name]
                return func_name  # Use function name as a type
        elif isinstance(value_node, ast.Name):
            # For variables, we'd need symbol tracking to get the type
            # This is a simplified approach
            if value_node.id == 'True' or value_node.id == 'False':
                return 'Bool'
            elif value_node.id == 'None':
                return 'None'
            return None  # Can't determine the type without context
        elif isinstance(value_node, ast.BinOp):
            left_type = self._infer_type_from_value(value_node.left)
            right_type = self._infer_type_from_value(value_node.right)
            
            # Type inference for binary operations
            if isinstance(value_node.op, ast.Add):
                if left_type == "String" or right_type == "String":
                    return "String"
                elif left_type == "Number" and right_type == "Number":
                    return "Number"
            elif isinstance(value_node.op, (ast.Sub, ast.Mult, ast.Div)):
                if left_type == "Number" and right_type == "Number":
                    return "Number"
                elif isinstance(value_node.op, ast.Mult) and (
                    (left_type == "String" and right_type == "Number") or
                    (left_type == "Number" and right_type == "String")
                ):
                    return "String"
            
        return None

def extract_type_errors(code: str):
    """
    Find potential type errors in code without executing it.
    Basic implementation looking for common type mismatches.
    """
    tree = ast.parse(code)
    errors = []
    
    class ErrorFinder(ast.NodeVisitor):
        def visit_BinOp(self, node):
            # String and non-string addition
            left_type = infer_type(node.left)
            right_type = infer_type(node.right)
            
            if isinstance(node.op, ast.Add):
                if left_type == "String" and right_type and right_type != "String":
                    errors.append({
                        "type": "TypeError",
                        "message": f"Cannot add string and {right_type.lower()}",
                        "line": getattr(node, 'lineno', 0),
                        "operation": "Add",
                        "left_type": "String",
                        "right_type": right_type
                    })
                elif right_type == "String" and left_type and left_type != "String":
                    errors.append({
                        "type": "TypeError",
                        "message": f"Cannot add {left_type.lower()} and string",
                        "line": getattr(node, 'lineno', 0),
                        "operation": "Add",
                        "left_type": left_type,
                        "right_type": "String"
                    })
            
            # Division by zero potential
            if isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant) and node.right.value == 0:
                errors.append({
                    "type": "ZeroDivisionError",
                    "message": "Division by zero",
                    "line": getattr(node, 'lineno', 0),
                    "operation": "Divide"
                })
            
            self.generic_visit(node)
        
        def visit_Call(self, node):
            # Check for potential type mismatches in function calls
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                # For demonstration: check calls to len() with wrong types
                if func_name == 'len' and len(node.args) > 0:
                    arg_type = infer_type(node.args[0])
                    if arg_type in ['Number', 'Bool']:
                        errors.append({
                            "type": "TypeError",
                            "message": f"len() requires a sequence, not {arg_type}",
                            "line": getattr(node, 'lineno', 0),
                            "operation": "Call",
                            "function": "len",
                            "arg_type": arg_type
                        })
            
            self.generic_visit(node)
    
    def infer_type(node):
        """Simple type inference for error detection."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return "String"
            elif isinstance(node.value, (int, float)):
                return "Number"
            elif isinstance(node.value, bool):
                return "Bool"
            return "Unknown"
        elif isinstance(node, ast.Str):  # For older Python versions
            return "String"
        elif isinstance(node, ast.Num):  # For older Python versions
            return "Number"
        elif isinstance(node, ast.List):
            return "List"
        elif isinstance(node, ast.Dict):
            return "Dict"
        elif isinstance(node, ast.Name):
            if node.id in ['True', 'False']:
                return "Bool"
        return None
    
    ErrorFinder().visit(tree)
    return errors

def decompose_python_file(file_path):
    """
    Parse a Python file and extract type information and potential errors.
    Returns MeTTa expressions as strings.
    """
    try:
        with open(file_path, 'r') as f:
            code = f.read()
            
        tree = ast.parse(code)
        extractor = TypeExtractor()
        extractor.visit(tree)
        
        # Extract potential type errors
        errors = extract_type_errors(code)
        error_atoms = []
        
        for error in errors:
            if error["type"] == "TypeError":
                error_atoms.append(
                    f"(TypeError \"{error['message']}\" {error['operation']} "
                    f"{error.get('left_type', 'Any')} {error.get('right_type', 'Any')})"
                )
            elif error["type"] == "ZeroDivisionError":
                error_atoms.append(
                    f"(ZeroDivisionError \"{error['message']}\" {error['operation']})"
                )
        
        return {
            "type_atoms": extractor.type_atoms,
            "error_atoms": error_atoms
        }
    except Exception as e:
        print(f"Error decomposing file: {e}")
        return {"type_atoms": [], "error_atoms": []}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = decompose_python_file(sys.argv[1])
        print("Type atoms:")
        for atom in result["type_atoms"]:
            print(atom)
        
        print("\nError atoms:")
        for atom in result["error_atoms"]:
            print(atom)
    else:
        print("Please provide a Python file to analyze.")