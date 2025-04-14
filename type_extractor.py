import ast

class TypeExtractor(ast.NodeVisitor):
    """Extract type information from Python AST."""
    
    def __init__(self):
        self.type_atoms = []  # Will store MeTTa-compatible atoms
        self.current_function = None
        self.current_class = None
        self.current_scope = []
        
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
                
        # Handle arguments and their annotations
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = None
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_type = arg.annotation.id
                elif isinstance(arg.annotation, ast.Attribute):
                    arg_type = f"{arg.annotation.value.id}.{arg.annotation.attr}"
            args.append((arg_name, arg_type))
        
        # Create MeTTa atom for function signature
        self.type_atoms.append(f"(: {node.name} (-> {' '.join([arg_type or '%Undefined%' for _, arg_type in args])} {return_type or '%Undefined%'}))")
        
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
                    scope_prefix = '.'.join(self.current_scope)
                    self.type_atoms.append(f"(: {scope_prefix}.{var_name} {var_type})")
        
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
                
            if var_type:
                scope_prefix = '.'.join(self.current_scope)
                self.type_atoms.append(f"(: {scope_prefix}.{var_name} {var_type})")
        
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
                # For simple function calls, use function name as type
                return value_node.func.id
        return None

def extract_type_errors(code: str):
    """
    Find potential type errors in code without executing it.
    Very basic implementation - just looking for obvious mismatches.
    """
    tree = ast.parse(code)
    errors = []
    
    class ErrorFinder(ast.NodeVisitor):
        def visit_BinOp(self, node):
            # Check for str + non-str operations
            if (isinstance(node.left, ast.Constant) and isinstance(node.left.value, str) and
                isinstance(node.right, ast.Constant) and not isinstance(node.right.value, str) and
                isinstance(node.op, ast.Add)):
                errors.append({
                    "type": "TypeError", 
                    "message": "Cannot add string and non-string",
                    "line": node.lineno,
                    "operation": "Add",
                    "left_type": "String",
                    "right_type": self._get_type_name(node.right.value)
                })
            self.generic_visit(node)
        
        def _get_type_name(self, value):
            return type(value).__name__
    
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
            error_atoms.append(f"(TypeError \"{error['message']}\" {error['operation']} {error['left_type']} {error['right_type']})")
        
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