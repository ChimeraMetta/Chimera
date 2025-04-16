# Python Type-Theoretic Decomposer & Donor System

A comprehensive system for analyzing Python code and transforming it into a type-theoretic representation for MeTTa-based reasoning, with a layered approach to code transformations.

## Table of Contents

- [Overview](#overview)
- [Type-Theoretic Decomposer](#type-theoretic-decomposer)
  - [How It Works](#how-it-works)
  - [Implementation Details](#implementation-details)
  - [Type Inference System](#type-inference-system)
  - [Converting to MeTTa Atoms](#converting-to-metta-atoms)
- [Layered Donor System](#layered-donor-system)
  - [Layer 1: Fragment Donors](#layer-1-fragment-donors)
  - [Layer 2: Operation Donors](#layer-2-operation-donors)
  - [Layer 3: Type Conversion Donors](#layer-3-type-conversion-donors)
  - [Layer 4: Function Donors](#layer-4-function-donors)
  - [Context-Aware Donors](#context-aware-donors)
- [MeTTa Reasoning Process](#metta-reasoning-process)
  - [Pattern Matching](#pattern-matching)
  - [Layered Selection](#layered-selection)
  - [Transformation Application](#transformation-application)
- [Runtime Integration](#runtime-integration)
- [Usage Examples](#usage-examples)
- [Conclusion](#conclusion)

## Overview

The system consists of two primary components:

1. **Type-Theoretic Decomposer**: Analyzes Python code to extract structural, type, and relationship information
2. **Layered Donor System**: Provides patterns and transformations organized in layers of increasing abstraction

These components work together, with the decomposer feeding information to MeTTa, which uses the donor system to reason about possible code transformations.

## Type-Theoretic Decomposer

### How It Works

The decomposer analyzes Python code through these steps:

1. **Parse Python Code**: Uses Python's `ast` module to create an Abstract Syntax Tree
2. **Traverse AST**: Extracts type information, code structure, and patterns
3. **Infer Types**: Determines types from annotations, literal values, and operations
4. **Build Relationships**: Maps ontological relationships between code elements
5. **Convert to MeTTa Atoms**: Generates atoms representing the code's structure

### Implementation Details

The core of the static analyzer is the `CodeDecomposer` class:

```python
class CodeDecomposer(ast.NodeVisitor):
    def __init__(self):
        # Structural atoms for MeTTa
        self.atoms = []
        
        # Track scopes and structure
        self.current_function = None
        self.current_class = None
        self.scope_stack = []
        
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
```

Key node visitors include:

#### Function Definition Visitor

```python
def visit_FunctionDef(self, node):
    # Store current state
    prev_function = self.current_function
    self.current_function = node.name
    self.scope_stack.append(f"function:{node.name}")
    
    # Extract parameters and return type
    params = []
    return_type = None
    
    if node.returns:
        return_type = self._get_type_from_annotation(node.returns)
    
    for arg in node.args.args:
        param_type = None
        if arg.annotation:
            param_type = self._get_type_from_annotation(arg.annotation)
        params.append((arg.arg, param_type))
    
    # Create function atom
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
    
    # Visit function body
    for stmt in node.body:
        self.visit(stmt)
    
    # Restore state
    self.current_function = prev_function
    self.scope_stack.pop()
```

#### Binary Operation Visitor

```python
def visit_BinOp(self, node):
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
```

### Type Inference System

The type inference system operates through several strategies:

```python
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
    elif isinstance(node, ast.List):
        return "List"
    elif isinstance(node, ast.Dict):
        return "Dict"
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
    elif isinstance(node, ast.BinOp):
        left_type = self._infer_type_from_value(node.left)
        right_type = self._infer_type_from_value(node.right)
        
        # Type inference for binary operations
        if isinstance(node.op, ast.Add):
            if left_type == "String" or right_type == "String":
                return "String"
            elif left_type == "Number" and right_type == "Number":
                return "Number"
        elif isinstance(node.op, (ast.Sub, ast.Mult, ast.Div)):
            if left_type == "Number" and right_type == "Number":
                return "Number"
    
    return "Any"
```

### Converting to MeTTa Atoms

The extracted information is converted to MeTTa atoms:

```python
def convert_to_metta_atoms(decomposer):
    metta_atoms = []
    
    # Convert basic structure atoms
    for atom in decomposer.atoms:
        if atom["type"] == "function_def":
            # Function definition atoms
            params_str = " ".join([param_type or "Any" for _, param_type in atom["params"]])
            return_str = atom["return_type"] or "Any"
            
            metta_atoms.append(f"(: {atom['name']} (-> {params_str} {return_str}))")
            metta_atoms.append(f"(function-def {atom['name']} \"{atom['scope']}\" {atom['line_start']} {atom['line_end']})")
            
            # Add parameters
            for i, (param_name, param_type) in enumerate(atom["params"]):
                metta_atoms.append(f"(function-param {atom['name']} {i} {param_name} {param_type or 'Any'})")
        
        elif atom["type"] == "bin_op":
            # Binary operation atoms
            metta_atoms.append(f"(bin-op {atom['op']} {atom['left_type']} {atom['right_type']} \"{atom['scope']}\" {atom['line']})")
    
    # Add ontological relationships
    for scope, modules in decomposer.module_relationships.items():
        for module in modules:
            metta_atoms.append(f"(module-depends \"{scope}\" {module})")
    
    # Add string operation patterns
    for i, op in enumerate(decomposer.string_operations):
        metta_atoms.append(f"(string-op-pattern {i} {op['op']} {op.get('left_type', 'Any')} {op.get('right_type', 'Any')} {op['line']})")
    
    return metta_atoms
```

## Layered Donor System

The donor system uses a layered approach inspired by biological immune systems, with each layer providing increasingly comprehensive solutions.

### Layer 1: Fragment Donors

**Purpose**: Provide specific code patterns for common operations.

**Implementation in MeTTa**:
```
;; Basic code fragments
(= (fragment-donor "python_string_concat")
   "str1 + str2")

(= (fragment-donor "python_f_string")
   "f\"{variable}\"")

(= (fragment-donor "python_format_string")
   "\"{}\".format(variable)")

(= (fragment-donor "python_string_join")
   "\", \".join(items)")

(= (fragment-donor "python_list_comprehension")
   "[x for x in items]")

(= (fragment-donor "python_zero_division_check")
   "if divisor != 0:
    result = dividend / divisor
else:
    result = float('inf')")
```

**Characteristics**:
- Smallest unit of reusable code
- No context or type information
- Focused on specific syntax patterns
- Highly reusable across different contexts

**Example Application**:
When encountering string concatenation, the system might suggest using f-strings instead of the `+` operator.

### Layer 2: Operation Donors

**Purpose**: Provide patterns for specific operations with type context.

**Implementation in MeTTa**:
```
;; Operation-specific patterns with type awareness
(= (operation-donor Add String String)
   (fragment-donor "python_string_concat"))

(= (operation-donor Add String Number)
   (fragment-donor "python_f_string"))

(= (operation-donor Add Number String)
   (fragment-donor "python_f_string"))

(= (operation-donor Div Number Number)
   (fragment-donor "python_zero_division_check"))

(= (operation-donor For List Any)
   (fragment-donor "python_list_comprehension"))
```

**Characteristics**:
- Combines operation type with operand types
- More specific than fragment donors
- Addresses type-specific operations
- Handles common error patterns

**Example Application**:
When adding a string and a number, the system suggests using f-strings to avoid type errors.

### Layer 3: Type Conversion Donors

**Purpose**: Provide generic patterns for converting between types.

**Implementation in MeTTa**:
```
;; Type conversion patterns
(= (type-donor String Number)
   "float(string_value)")

(= (type-donor Number String)
   "str(number_value)")

(= (type-donor List String)
   "\", \".join(list_value)")

(= (type-donor String List)
   "string_value.split(separator)")

(= (type-donor Dict List)
   "list(dict_value.items())")
```

**Characteristics**:
- Focuses on type conversions
- Generic across different operations
- Used when types need to be reconciled
- Applicable in various contexts

**Example Application**:
When a number needs to be used in a string context, the system suggests converting it to a string.

### Layer 4: Function Donors

**Purpose**: Provide complete, reusable function implementations.

**Implementation in MeTTa**:
```
;; Complete function implementations
(= (function-donor "string_formatter")
   "def format_string(template: str, *args, **kwargs) -> str:
    return template.format(*args, **kwargs)")

(= (function-donor "safe_division")
   "def safe_divide(a: float, b: float, default: float = float('inf')) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        return default")

(= (function-donor "list_processor")
   "def process_list(items: list, transform_func = None) -> list:
    if transform_func:
        return [transform_func(item) for item in items]
    return items")
```

**Characteristics**:
- Highest level of abstraction
- Complete, self-contained solutions
- Encapsulates best practices
- Handles multiple related patterns

**Example Application**:
When dealing with division that might involve zero, the system suggests extracting the operation into a safe division function.

### Context-Aware Donors

**Purpose**: Provide domain-specific patterns for particular contexts.

**Implementation in MeTTa**:
```
;; Context-specific patterns
(= (context-donor "data_processing" "list_transformation")
   (function-donor "list_processor"))

(= (context-donor "finance" "number_formatting")
   "def format_currency(amount: float, currency: str = '$') -> str:
    return f\"{currency}{amount:.2f}\"")

(= (context-donor "web" "string_escaping")
   "def escape_html(text: str) -> str:
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')")
```

**Characteristics**:
- Domain-specific patterns
- Tailored to particular application areas
- Highest precedence in selection
- Incorporates domain conventions

**Example Application**:
In a financial context, number formatting should use proper currency symbols and decimal precision.

## MeTTa Reasoning Process

### Pattern Matching

MeTTa matches code patterns from the decomposer against its donor system:

```
;; Pattern recognition
(= (recognize-pattern (bin-op "Add" "String" "String" $scope $line))
   ("string_concat" String String))

(= (recognize-pattern (bin-op "Add" "String" "Number" $scope $line))
   ("string_format" String Number))

(= (recognize-pattern (bin-op "Add" "Number" "String" $scope $line))
   ("string_format" Number String))

(= (recognize-pattern (function-call "sum" 1 $scope $line))
   ("list_transform" List Number))
```

### Layered Selection

MeTTa selects donors using a layered approach, starting with the most specific:

```
;; Find appropriate donor based on pattern
(= (find-donor $pattern)
   (case (recognize-pattern $pattern)
     (Empty None)
     (($operation $left_type $right_type)
      (find-donor-by-operation $operation $left_type $right_type))))

;; Layered donor selection logic
(= (find-donor-by-operation $operation $left_type $right_type)
   (let $context-donor (context-donor (current-context) $operation)
        (case $context-donor
          (Empty
           (let $op-donor (operation-donor $operation $left_type $right_type)
                (case $op-donor
                  (Empty
                   (let $type-donor (type-donor $left_type $right_type $operation)
                        (case $type-donor
                          (Empty
                           (case (function-donor-for-operation $operation)
                             (Empty "No donor found for this pattern")
                             ($donor $donor)))
                          ($donor $donor))))
                  ($donor $donor))))
          ($donor $donor))))
```

This selection process follows a clear hierarchy:
1. First, check for context-specific donors
2. If none found, check for operation-specific donors
3. If none found, check for type conversion donors
4. If none found, check for function donors
5. If still none found, report that no donor is available

### Transformation Application

Once a donor is selected, it needs to be applied to the code:

```
;; Transform code based on pattern
(= (transform-code $pattern)
   (let $donor (find-donor $pattern)
        (case (= $donor None)
          (True "No transformation available")
          (False (apply-transformation $donor $pattern)))))

;; Apply transformation
(= (apply-transformation $donor (bin-op "Add" "String" "Number" $scope $line))
   "Use f-strings: f\"{string_variable}{number_variable}\"")

(= (apply-transformation $donor (bin-op "Add" "String" "String" $scope $line))
   "Use string concatenation: string1 + string2")

(= (apply-transformation $donor $pattern)
   $donor)
```

## Runtime Integration

The system can be extended with runtime feedback through decorated functions:

```python
@hybrid_transform(context="finance")
def format_price(item, price):
    """Format an item and price for display."""
    return item + ": $" + price  # Will be transformed to use f-strings
```

When this function runs:
1. First, static analysis is performed
2. Then, runtime monitoring captures any errors
3. Results are fed back to MeTTa to improve future suggestions
4. Successful patterns are reinforced

The system tracks success metrics for transformations:

```
(= (record-fix-attempt $error_signature $fix $success)
   (case (match-atom &self (success-record $error_signature $fix $timestamp $success_count $failure_count))
     (Empty 
      (case $success
        (True (add-atom &self (success-record $error_signature $fix (current-time) 1 0)))
        (False (add-atom &self (success-record $error_signature $fix (current-time) 0 1)))))
     ((success-record $error_signature $fix $timestamp $success_count $failure_count)
      (case $success
        (True (update-atom &self 
                (success-record $error_signature $fix $timestamp $success_count $failure_count)
                (success-record $error_signature $fix (current-time) (+ $success_count 1) $failure_count)))
        (False (update-atom &self
                (success-record $error_signature $fix $timestamp $success_count $failure_count)
                (success-record $error_signature $fix (current-time) $success_count (+ $failure_count 1))))))))
```

## Usage Examples

### Static Analysis of a File

```python
from static_analyzer import decompose_file

# Analyze a Python file
result = decompose_file("my_script.py")

# Print MeTTa atoms
for atom in result["metta_atoms"]:
    print(atom)
```

### Dynamic Transformation of Functions

```python
from hybrid_transformer import hybrid_transform

@hybrid_transform(context="data_processing")
def process_data(items):
    """Process a list of items."""
    result = ""
    for item in items:
        result += str(item) + ", "  # Will be transformed to join operation
    return result.rstrip(", ")
```

### Sample Output

The decomposer generates MeTTa atoms like these:

```
(: add_numbers (-> int int int))
(function-def add_numbers "global" 3 5)
(function-param add_numbers 0 a int)
(function-param add_numbers 1 b int)
(bin-op Add Any Any "function:add_numbers" 5)

(string-op-pattern 0 Add String Any 9)
(loop-pattern 0 for "function:calculate_total" 14)
(function-pattern 3 format_receipt 2 True "global")
```

## Conclusion

The Python Type-Theoretic Decomposer and Layered Donor System provide a comprehensive approach to code analysis and transformation. By decomposing Python code into a type-theoretic structure and using a layered donor system for transformations, the system can:

1. Identify potential issues in code statically
2. Suggest improvements based on best practices
3. Learn from runtime behavior
4. Apply context-specific transformations

The layered approach allows the system to balance between specificity and generality, providing the most appropriate transformation for each code pattern. As the system learns from more code, it builds a better understanding of successful patterns and can provide increasingly effective suggestions.

This system demonstrates how MeTTa reasoning can be applied to real-world programming tasks, creating a bridge between symbolic AI and practical software development.