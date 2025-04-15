;; Extended MeTTa Donor System for Python Type Errors
;; A layered system that provides fixes for type errors

;; Layer 1: Code Fragment Donors
;; Specific code fragments for known error patterns
(= (code-donor "String_plus_Number")
   "str(number_value)  ;; Convert number to string before concatenation")

(= (code-donor "Number_plus_String")
   "float(string_value)  ;; Convert string to number for addition")

(= (code-donor "String_format_Number")
   "f\"{string_value}{number_value}\"  ;; Use f-strings for clean formatting")

(= (code-donor "Division_by_Zero_Check")
   "if divisor != 0:
    result = dividend / divisor
else:
    result = float('inf')  # or raise an appropriate exception")

(= (code-donor "Format_Receipt_Item")
   "f\"{item}: ${price:.2f}\"  # Format using f-string")

(= (code-donor "Format_Receipt_Total")
   "f\"Total: ${total:.2f}\"  # Format using f-string")

(= (code-donor "Variable_Type_Check")
   "if isinstance(value, str):
    # Handle string case
    result = string_operation(value)
elif isinstance(value, int):
    # Handle int case
    result = str(value)  # Convert to string if needed
else:
    # Handle other types
    result = str(value)")

;; Layer 2: Operation Error Donors
;; Knowledge about operations and their expected types
(= (operation-donor Add String Number)
   (code-donor "String_plus_Number"))

(= (operation-donor Add Number String)
   (code-donor "Number_plus_String"))

(= (operation-donor Multiply String Number)
   "string_value * number_value  ;; String multiplication is valid in Python")

(= (operation-donor Divide Number Number)
   (code-donor "Division_by_Zero_Check"))

(= (operation-donor Call "len" Number)
   "# len() requires a sequence, not a number
# Convert to a sequence or use a different function
len(str(number))  # Convert to string first if you need length of digits")

;; Layer 3: Type Conversion Donors
;; Generic conversions between types
(= (convert-donor String Number)
   "float(string_value)  ;; Convert string to float")

(= (convert-donor Number String)
   "str(number_value)  ;; Convert number to string")

(= (convert-donor Bool String)
   "str(bool_value)  ;; Convert boolean to string")

(= (convert-donor List String)
   "', '.join(map(str, list_value))  ;; Convert list to comma-separated string")

;; Layer 4: Complete function donors
;; Reference implementations that work correctly
(= (function-donor "add_str_num")
   "def add_str_num(text: str, num: float) -> str:
    return text + str(num)  # Convert number to string")

(= (function-donor "add_num_str")
   "def add_num_str(num: float, text: str) -> float:
    return num + float(text)  # Convert string to number")

(= (function-donor "format_receipt_item")
   "def format_receipt_item(item: str, price: float) -> str:
    return f\"{item}: ${price:.2f}\"  # Format using f-string")

(= (function-donor "safe_division")
   "def safe_division(a: float, b: float, default: float = float('inf')) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        return default")

(= (function-donor "type_safe_operation")
   "def type_safe_operation(value: Union[str, int]) -> str:
    if isinstance(value, str):
        return \"Value: \" + value
    else:
        return \"Value: \" + str(value)")

;; Query functions to find appropriate donors
(= (find-donor-for-error (TypeError $msg $op $left-type $right-type))
   (case (operation-donor $op $left-type $right-type)
     (Empty 
       (case (convert-donor $right-type $left-type)
         (Empty 
           (case (function-donor-for-types $left-type $right-type)
             (Empty "No specific donor found for this error")
             ($donor $donor)))
         ($donor $donor)))
     ($donor $donor)))

(= (find-donor-for-error (ZeroDivisionError $msg $op))
   (case (operation-donor $op Number Number)
     (Empty (function-donor "safe_division"))
     ($donor $donor)))

;; Helper function to find function donors based on types
(= (function-donor-for-types String Number)
   (function-donor "add_str_num"))

(= (function-donor-for-types Number String)
   (function-donor "add_num_str"))

(= (function-donor-for-types String Union)
   (function-donor "type_safe_operation"))

;; Context-specific donors for common patterns
(= (context-donor "receipt_formatting" Add String Number)
   (code-donor "Format_Receipt_Item"))

(= (context-donor "receipt_formatting" Add String Float)
   (code-donor "Format_Receipt_Item"))

;; Function to get most specific donor based on context
(= (get-donor-with-context $context $op $left-type $right-type)
   (case (context-donor $context $op $left-type $right-type)
     (Empty (find-donor-for-error (TypeError "Type error" $op $left-type $right-type)))
     ($donor $donor)))