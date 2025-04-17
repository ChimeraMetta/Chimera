;; Donor system for code transformation
;; Organized in layers from specific to general

;; Define code snippet types
(: CodeSnippet Type)
(: PythonSnippet CodeSnippet)

;; Layer 1: Fragment Donors - Specific code patterns
(= (fragment-donor python_string_concat)
   (PythonSnippet str1_plus_str2))

(= (fragment-donor python_f_string)
   (PythonSnippet f_string_with_variable))

(= (fragment-donor python_format_string)
   (PythonSnippet string_dot_format))

(= (fragment-donor python_string_join)
   (PythonSnippet join_items))

(= (fragment-donor python_list_comprehension)
   (PythonSnippet list_comprehension))

(= (fragment-donor python_zero_division_check)
   (PythonSnippet if_divisor_not_zero))

(= (fragment-donor python_index_check)
   (PythonSnippet if_index_in_bounds))

(= (fragment-donor python_try_except)
   (PythonSnippet try_except_block))

;; Layer 2: Operation Donors - Specific operations with type context
(= (operation-donor Add String String)
   (fragment-donor python_string_concat))

(= (operation-donor Add String Number)
   (fragment-donor python_f_string))

(= (operation-donor Add Number String)
   (fragment-donor python_f_string))

(= (operation-donor Div Number Number)
   (fragment-donor python_zero_division_check))

(= (operation-donor Subscript List Number)
   (fragment-donor python_index_check))

;; Layer 3: Type Conversion Donors - Generic type conversions
(= (type-donor String Number)
   (PythonSnippet float_string_value))

(= (type-donor Number String)
   (PythonSnippet str_number_value))

(= (type-donor List String)
   (PythonSnippet join_list_value))

(= (type-donor String List)
   (PythonSnippet string_split))

(= (type-donor Dict List)
   (PythonSnippet list_dict_items))

;; Layer 4: Function Donors - Complete function implementations
(= (function-donor string_formatter)
   (PythonSnippet format_string_function))

(= (function-donor safe_division)
   (PythonSnippet safe_divide_function))

(= (function-donor safe_index)
   (PythonSnippet safe_get_function))

(= (function-donor list_processor)
   (PythonSnippet process_list_function))

;; Context-specific donors
(= (context-donor data_processing list_transformation)
   (function-donor list_processor))

(= (context-donor finance number_formatting)
   (PythonSnippet format_currency_function))

(= (context-donor web string_escaping)
   (PythonSnippet escape_html_function))

;; Code snippet definitions (actual Python code)
(= (python-snippet str1_plus_str2)
   "str1 + str2")

(= (python-snippet f_string_with_variable)
   "f\"{variable}\"")

(= (python-snippet string_dot_format)
   "\"{}\".format(variable)")

(= (python-snippet join_items)
   "\", \".join(items)")

(= (python-snippet list_comprehension)
   "[x for x in items]")

(= (python-snippet if_divisor_not_zero)
   "if divisor != 0:\n    result = dividend / divisor\nelse:\n    result = float('inf')")

(= (python-snippet if_index_in_bounds)
   "if 0 <= index < len(items):\n    result = items[index]\nelse:\n    result = default")

(= (python-snippet try_except_block)
   "try:\n    result = operation()\nexcept ExceptionType as e:\n    result = default_value")

(= (python-snippet float_string_value)
   "float(string_value)")

(= (python-snippet str_number_value)
   "str(number_value)")

(= (python-snippet join_list_value)
   "\", \".join(list_value)")

(= (python-snippet string_split)
   "string_value.split(separator)")

(= (python-snippet list_dict_items)
   "list(dict_value.items())")

(= (python-snippet format_string_function)
   "def format_string(template: str, *args, **kwargs) -> str:\n    return template.format(*args, **kwargs)")

(= (python-snippet safe_divide_function)
   "def safe_divide(a: float, b: float, default: float = float('inf')) -> float:\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return default")

(= (python-snippet safe_get_function)
   "def safe_get(items: list, index: int, default=None):\n    if 0 <= index < len(items):\n        return items[index]\n    return default")

(= (python-snippet process_list_function)
   "def process_list(items: list, transform_func=None) -> list:\n    if transform_func:\n        return [transform_func(item) for item in items]\n    return items")

(= (python-snippet format_currency_function)
   "def format_currency(amount: float, currency: str = '$') -> str:\n    return f\"{currency}{amount:.2f}\"")

(= (python-snippet escape_html_function)
   "def escape_html(text: str) -> str:\n    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')")