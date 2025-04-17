;; Codebase Ontology and Reasoning Rules
;; For use with the static analyzer and dynamic monitor

;; =====================
;; Fragment Donors
;; =====================
(= (fragment-donor python-string-concat)
   "str1 + str2")

(= (fragment-donor python-f-string)
   "f\"{variable}\"")

(= (fragment-donor python-format-string)
   "\"{}\".format(variable)")

(= (fragment-donor python-string-join)
   "\", \".join(items)")

(= (fragment-donor python-list-comprehension)
   "[x for x in items]")

(= (fragment-donor python-zero-division-check)
   "if divisor != 0:
    result = dividend / divisor
else:
    result = float('inf')")

(= (fragment-donor python-index-check)
   "if 0 <= index < len(sequence):
    item = sequence[index]
else:
    item = default")

(= (fragment-donor python-dict-lookup)
   "try:
    value = dictionary[key]
except KeyError:
    value = default")

;; =====================
;; Operation Donors
;; =====================
(= (operation-donor Add String String)
   (fragment-donor python-string-concat))

(= (operation-donor Add String Number)
   (fragment-donor python-f-string))

(= (operation-donor Add Number String)
   (fragment-donor python-f-string))

(= (operation-donor Div Number Number)
   (fragment-donor python-zero-division-check))

(= (operation-donor Subscript List Number)
   (fragment-donor python-index-check))

(= (operation-donor Subscript Dict Any)
   (fragment-donor python-dict-lookup))

;; =====================
;; Type Conversion Donors
;; =====================
(= (type-donor String Number)
   "float(string_value)")

(= (type-donor Number String)
   "str(number_value)")

(= (type-donor List String)
   "\", \".join(list_value)")

(= (type-donor String List)
   "string_value.split(separator)")

;; =====================
;; Function Donors
;; =====================
(= (function-donor safe-division)
   "def safe_divide(a: float, b: float, default: float = float('inf')) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        return default")

(= (function-donor safe-index)
   "def safe_get(sequence, index, default=None):
    try:
        return sequence[index]
    except (IndexError, KeyError):
        return default")

(= (function-donor string-formatter)
   "def format_string(template: str, *args, **kwargs) -> str:
    try:
        return template.format(*args, **kwargs)
    except (ValueError, KeyError, IndexError) as e:
        return f\"Error formatting string: {e}\"")

;; =====================
;; Error Fix Rules
;; =====================
(= (error-fix ZeroDivisionError safe-division)
   "Replace division operation with safe_divide function")

(= (error-fix IndexError safe-index)
   "Replace direct indexing with safe_get function")

(= (error-fix TypeError string-conversion)
   "Add explicit type conversion")

(= (error-fix KeyError safe-dict-lookup)
   "Add try-except block for dictionary lookups")

;; =====================
;; Fix Code Templates
;; =====================
(= (fix-code safe-division)
   "def safe_divide(a, b, default=float('inf')):
    try:
        return a / b
    except ZeroDivisionError:
        return default")

(= (fix-code safe-index)
   "def safe_get(sequence, index, default=None):
    try:
        return sequence[index]
    except (IndexError, KeyError):
        return default")

(= (fix-code string-conversion)
   "# Convert values to strings before concatenation
str(value1) + str(value2)")

(= (fix-code safe-dict-lookup)
   "try:
    value = dictionary[key]
except KeyError:
    value = default_value")

;; =====================
;; Pattern Recognition Rules
;; =====================
(= (recognize-pattern (bin-op Add String String $scope $line))
   (string-concat String String))

(= (recognize-pattern (bin-op Add String Number $scope $line))
   (string-format String Number))

(= (recognize-pattern (bin-op Add Number String $scope $line))
   (string-format Number String))

(= (recognize-pattern (bin-op Div Number Number $scope $line))
   (division Number Number))

;; =====================
;; Error Pattern Recognition
;; =====================
(= (recognize-error-pattern ZeroDivisionError (division $a $b))
   (safe-division-needed $a $b))

(= (recognize-error-pattern IndexError (Subscript List Number))
   (safe-index-needed List Number))

(= (recognize-error-pattern TypeError (bin-op Add String Number $scope $line))
   (type-conversion-needed String Number Add))

(= (recognize-error-pattern TypeError (bin-op Add Number String $scope $line))
   (type-conversion-needed Number String Add))

;; =====================
;; Dynamic Error Analysis
;; =====================
(= (analyze-error $exec-id $error-id)
   (case (match &self (error-type $error-id $type))
     (Empty (no-fix-available $error-id))
     ($type (find-error-fix $error-id $type))))

(= (find-error-fix $error-id $error-type)
   (case (match &self (error-fix $error-type $fix) $fix)
     (Empty (no-fix-available $error-id))
     ($fix (suggest-fix $error-id $fix))))

;; =====================
;; Fix Selection Rules
;; =====================
(= (select-fix-for $error-id $func-name $fix)
   (case (match &self (function-context $func-name $context))
     (Empty $fix)  ; No context, use general fix
     ($context (context-aware-fix $error-id $fix $context))))

(= (context-aware-fix $error-id $fix finance)
   (case (match &self (error-type $error-id ZeroDivisionError))
     (Empty $fix)
     ($_ safe-financial-division)))

;; =====================
;; Function Recommendation Rules
;; =====================
(= (function-recommendation $func-name error-pattern $description 0.8)
   (case (match &self (execution-error $exec-id $error-id $time)
               (match &self (execution-start $exec-id $func-name $start-time))
               (match &self (error-type $error-id $error-type)))
     (Empty False)
     ($_ (add-recommendation $func-name $error-type $description))))

(= (add-recommendation $func-name ZeroDivisionError "Consider using a safe division utility to handle zero divisors")
   True)

(= (add-recommendation $func-name IndexError "Add bounds checking before accessing list elements")
   True)

(= (add-recommendation $func-name TypeError "Ensure consistent types by adding explicit conversions")
   True)

;; Performance and complexity recommendations based on execution metrics
(= (function-recommendation $func-name performance $description 0.7)
   (case (match &self (function-metrics $func-name $calls $successes $failures $avg-time))
     (Empty False)
     (($calls $successes $failures $avg-time) 
      (if (> $avg-time 1.0)  ; Threshold for slow functions
          (add-performance-recommendation $func-name $avg-time)
          False))))

(= (add-performance-recommendation $func-name $avg-time)
   "Function has high average execution time. Consider optimizing algorithm or caching results")

;; =====================
;; Code Quality Rules
;; =====================
(= (code-quality-issue $func-name complexity "Function has too many loops" 0.6)
   (case (match &self (loop-pattern $id for $func-scope $line1)
               (match &self (loop-pattern $id2 for $func-scope $line2))
               (match &self (loop-pattern $id3 for $func-scope $line3)))
     (Empty False)
     ($_ (if (!= $line1 $line2) 
            (if (!= $line1 $line3)
                (if (!= $line2 $line3)
                    (contains-scope $func-scope $func-name)
                    False)
                False)
            False))))

(= (contains-scope $scope $func-name)
   (case (match &self (function-def $func-name $scope $start $end))
     (Empty False)
     ($_ True)))

;; =====================
;; Utility Functions
;; =====================
(= (current-time)
   (time))

;; =====================
;; Function Error Patterns
;; =====================
(= (function-error-pattern $func-name $error-type $frequency $description)
   (case (count-errors $func-name $error-type $count)
     (Empty False)
     ($count (if (> $count 2)
                 (error-pattern-description $error-type $description)
                 False))))

(= (count-errors $func-name $error-type $count)
   (case (match &self (execution-start $exec-id $func-name $start)
               (match &self (execution-error $exec-id $error-id $time))
               (match &self (error-type $error-id $error-type))
               (aggregate-count $func-name $error-type))
     (Empty 0)
     ($count $count)))

(= (error-pattern-description ZeroDivisionError "Function frequently encounters division by zero")
   True)

(= (error-pattern-description IndexError "Function frequently tries to access out-of-bounds indices")
   True)

(= (error-pattern-description TypeError "Function frequently has type mismatches")
   True)

;; =====================
;; Codebase Structure Analysis
;; =====================
(= (module-dependency-graph $module $dependencies)
   (case (match &self (module-depends $module $dependency))
     (Empty (list))
     ($dependency (cons $dependency (module-dependency-graph $module (remove $dependency $dependencies))))))

(= (class-hierarchy-graph $base-class $derived-classes)
   (case (match &self (class-hierarchy $base-class $derived))
     (Empty (list))
     ($derived (cons $derived (class-hierarchy-graph $base-class (remove $derived $derived-classes))))))

;; =====================
;; Query Support Functions
;; =====================
(= (find-all-functions $func-list)
   (case (match &self (function-def $name $scope $start $end))
     (Empty $func-list)
     ($name (find-all-functions (add-unique $func-list $name)))))

(= (add-unique $list $item)
   (if (contains $list $item)
       $list
       (append $list $item)))

(= (find-error-prone-functions $threshold)
   (case (match &self (function-metrics $name $calls $successes $failures $avg-time))
     (Empty (list))
     (($name $calls $successes $failures $avg-time)
      (if (> (/ $failures $calls) $threshold)
          (cons $name (find-error-prone-functions $threshold))
          (find-error-prone-functions $threshold)))))

;; =====================
;; Type Safety Analysis
;; =====================
(= (type-safety-issue $func-name $description 0.75)
   (case (match &self (execution-error $exec-id $error-id $time)
               (match &self (execution-start $exec-id $func-name $start))
               (match &self (error-type $error-id TypeError)))
     (Empty False)
     ($_ "Function may have type conversion issues. Consider adding explicit type checking")))

;; =====================
;; Code Relationship Analysis
;; =====================
(= (find-related-functions $func-name $related-list)
   (case (match &self (function-depends $func-name $called-func))
     (Empty $related-list)
     ($called-func (find-related-functions $func-name (add-unique $related-list $called-func)))))

(= (find-calling-functions $func-name $calling-list)
   (case (match &self (function-depends $caller $func-name))
     (Empty $calling-list)
     ($caller (find-calling-functions $func-name (add-unique $calling-list $caller)))))

;; =====================
;; Architectural Insights
;; =====================
(= (architectural-insight $description 0.7)
   (case (high-fan-in-functions 5 $functions)
     (Empty False)
     ($functions "Several functions have high fan-in, suggesting they might be bottlenecks or key utilities")))

(= (high-fan-in-functions $threshold $result)
   (case (match &self (function-def $name $scope $start $end))
     (Empty $result)
     ($name (case (count-callers $name $count)
              (if (> $count $threshold)
                  (high-fan-in-functions $threshold (add-unique $result $name))
                  (high-fan-in-functions $threshold $result))))))

(= (count-callers $func-name $count)
   (case (match &self (function-depends $caller $func-name))
     (Empty 0)
     ($_ (+ 1 (count-callers $func-name (- $count 1))))))