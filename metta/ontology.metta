;; Core Codebase Ontology Rules
;; These rules focus on relationships and architecture rather than errors

;; =====================
;; Module Structure
;; =====================

;; Define module dependency relationship
(= (module-depends-on $module $dependency)
   (match &self (import $dependency $module-scope $line)
          (contains-scope $module-scope $module)))

;; Define module import relationship
(= (module-imports $module $imported-module $imported-name)
   (match &self (import-from $imported-module $imported-name $module-scope $line)
          (contains-scope $module-scope $module)))

;; Check if a scope contains a module name
(= (contains-scope $scope $module)
   (= $scope $module))

;; Get all dependencies for a module
(= (all-module-dependencies $module $dependencies)
   (case (match &self (module-depends-on $module $dependency))
     (Empty $dependencies)
     ($dependency (all-module-dependencies $module (cons $dependency $dependencies)))))

;; =====================
;; Function Relationships
;; =====================

;; Direct function call relationship based on scope and line numbers
(= (direct-function-call $caller $callee)
   (match &self (function-def $caller $caller-scope $start $end)
          (function-call $callee $args $scope $line)
          (and (= $scope $caller-scope)
               (>= $line $start)
               (<= $line $end))))

;; Alternative approach using function-depends relationship which might be 
;; populated by the static analyzer
(= (function-calls $caller $callee)
   (or (direct-function-call $caller $callee)
       (match &self (function-depends $caller $callee))))

;; Transitive function call relationship (call chain)
(= (calls-transitively $caller $callee)
   (or (function-calls $caller $callee)
       (match &self (function-calls $caller $intermediate)
              (calls-transitively $intermediate $callee))))

;; Find all functions called by a specific function (direct and indirect)
(= (all-called-by $caller $result)
   (match &self (calls-transitively $caller $callee)
          (cons $callee $result)))

;; Find all functions that call a specific function (direct and indirect)
(= (all-callers-of $callee $result)
   (match &self (calls-transitively $caller $callee)
          (cons $caller $result)))

;; Count the depth of a call chain
(= (call-chain-depth $caller $callee $depth)
   (case (function-calls $caller $callee)
     (True (= $depth 1))
     (False (case (match &self (function-calls $caller $intermediate)
                          (call-chain-depth $intermediate $callee $intermediate-depth))
              (Empty (= $depth 9999))  ; No path found
              ($intermediate-depth (= $depth (+ 1 $intermediate-depth)))))))

;; Identify critical functions - those that are part of many call chains
(= (critical-function $func $threshold)
   (> (+ (count-callers $func) (count-callees $func)) $threshold))

;; Count number of functions that call this function
(= (count-callers $func)
   (case (match &self (function-calls $caller $func))
     (Empty 0)
     ($caller (+ 1 (count-callers $func)))))

;; Count number of functions called by this function
(= (count-callees $func)
   (case (match &self (function-calls $func $callee))
     (Empty 0)
     ($callee (+ 1 (count-callees $func)))))

;; Recursive call detection
(= (recursive-function $func)
   (function-calls $func $func))

;; Mutually recursive functions
(= (mutually-recursive-functions $func1 $func2)
   (and (function-calls $func1 $func2)
        (function-calls $func2 $func1)))

;; Identify potential callback patterns
(= (callback-pattern $func $callback-param)
   (and (match &self (function-param $func $idx $callback-param Any))
        (match &self (function-call $callback-param $args $func-scope $line)
               (function-def $func $func-scope $start $end)
               (and (>= $line $start) (<= $line $end)))))

;; Define function data dependency relationship
(= (function-uses-type $func $type)
   (match &self (function-param $func $idx $name $type)))

;; Function with a high number of calls to other functions (high fan-out)
(= (high-fan-out-function $func $threshold)
   (> (count-calls-from $func) $threshold))

;; Function called by many other functions (high fan-in)
(= (high-fan-in-function $func $threshold)
   (> (count-calls-to $func) $threshold))

;; Count calls from a function
(= (count-calls-from $func)
   (case (match &self (function-calls $func $callee))
     (Empty 0)
     ($callee (+ 1 (count-calls-from $func)))))

;; Count calls to a function
(= (count-calls-to $func)
   (case (match &self (function-calls $caller $func))
     (Empty 0)
     ($caller (+ 1 (count-calls-to $func)))))

;; =====================
;; Class Relationships
;; =====================

;; Define class inheritance relationship
(= (class-extends $derived $base)
   (match &self (class-inherits $derived $base)))

;; Define class method relationship
(= (class-has-method $class $method)
   (match &self (function-def $method $scope $start $end)
          (= $scope (concat "class:" $class))))

;; Define class uses relationship (when a class is used as a parameter type)
(= (function-uses-class $func $class)
   (match &self (function-param $func $idx $name $class)))

;; =====================
;; Data Flow Patterns
;; =====================

;; Define data flow between functions (output of one is input to another)
(= (data-flows $source-func $target-func $type)
   (and (function-returns $source-func $type)
        (function-accepts $target-func $type)))

;; Determine if a function returns a specific type
(= (function-returns $func $type)
   (match &self (: $func (-> $params $type))))

;; Determine if a function accepts a specific type as parameter
(= (function-accepts $func $type)
   (match &self (function-param $func $idx $name $type)))

;; =====================
;; Architectural Patterns
;; =====================

;; Identify layers in the architecture
(= (architectural-layer $name $funcs)
   (case (match &self (function-def $func $scope $start $end)
                (contains-layer-scope $scope $name))
     (Empty $funcs)
     ($func (architectural-layer $name (cons $func $funcs)))))

;; Helper to check if scope matches a layer
(= (contains-layer-scope $scope $layer)
   (starts-with $scope $layer))

;; Identify core utilities (functions called by many others)
(= (core-utility $func $threshold)
   (high-fan-in-function $func $threshold))

;; Identify boundary functions (those that accept external types)
(= (boundary-function $func)
   (function-accepts $func "Dict"))

;; =====================
;; Usage Patterns
;; =====================

;; Identify common type usage patterns
(= (common-type-usage $type)
   (> (count-type-usages $type) 3))

;; Count type usages
(= (count-type-usages $type)
   (case (match &self (function-param $func $idx $name $type))
     (Empty 0)
     ($func (+ 1 (count-type-usages $type)))))

;; Identify common operation patterns
(= (common-operation $op $left-type $right-type)
   (> (count-operations $op $left-type $right-type) 2))

;; Count operations of specific types
(= (count-operations $op $left-type $right-type)
   (case (match &self (bin-op $op $left-type $right-type $scope $line))
     (Empty 0)
     ($scope (+ 1 (count-operations $op $left-type $right-type)))))

;; =====================
;; Holistic Insights
;; =====================

;; Determine programming paradigm
(= (programming-paradigm $paradigm $confidence)
   (case (count-classes) 
     (0 (and (= $paradigm "Functional") (= $confidence 0.9)))
     ($classes (case (count-functions)
                 (0 (and (= $paradigm "Data-driven") (= $confidence 0.8)))
                 ($funcs (if (> (/ $funcs $classes) 5)
                             (and (= $paradigm "Mostly functional") (= $confidence 0.7))
                             (if (< (/ $funcs $classes) 2)
                                 (and (= $paradigm "Mostly object-oriented") (= $confidence 0.7))
                                 (and (= $paradigm "Mixed paradigm") (= $confidence 0.6)))))))))

;; Count classes
(= (count-classes)
   (case (match &self (class-def $name $scope $start $end))
     (Empty 0)
     ($name (+ 1 (count-classes)))))

;; Count functions
(= (count-functions)
   (case (match &self (function-def $name $scope $start $end))
     (Empty 0)
     ($name (+ 1 (count-functions)))))

;; Identify architectural patterns
(= (architectural-insight "Layered architecture detected" 0.8)
   (> (count-layers) 2))

(= (architectural-insight "Core utilities identified" 0.7)
   (core-utilities-exist 3))

(= (architectural-insight "Data processing pipeline pattern detected" 0.7)
   (data-pipeline-exists))

;; Helper for architectural insights
(= (count-layers)
   (case (match &self (architectural-layer $name $funcs))
     (Empty 0)
     ($name (+ 1 (count-layers)))))

(= (core-utilities-exist $threshold)
   (case (match &self (core-utility $func $threshold))
     (Empty False)
     ($func True)))

(= (data-pipeline-exists)
   (case (match &self (data-flows $src $tgt $type))
     (Empty False)
     ($flow True)))

;; =====================
;; Function Complexity Analysis
;; =====================

;; Function complexity based on operation count
(= (function-operation-complexity $func $count)
   (case (match &self (function-def $func $scope $start $end)
                (match &self (bin-op $op $left $right $scope $line)
                       (and (>= $line $start) (<= $line $end)))
                (aggregate-count))
     (Empty 0)
     ($count $count)))

;; Function complexity based on loop count
(= (function-loop-complexity $func $count)
   (case (match &self (function-def $func $scope $start $end)
                (match &self (loop-pattern $id $type $loop-scope $line)
                       (and (= $scope $loop-scope) (>= $line $start) (<= $line $end)))
                (aggregate-count))
     (Empty 0)
     ($count $count)))

;; Function is complex
(= (complex-function $func)
   (or (> (function-operation-complexity $func 0) 10)
       (> (function-loop-complexity $func 0) 2)))

;; =====================
;; Type System Analysis
;; =====================

;; Count parameter types used in the codebase
(= (type-frequency $type $count)
   (case (match &self (function-param $func $idx $name $type)
                (aggregate-count))
     (Empty 0)
     ($count $count)))

;; Common types (used in multiple functions)
(= (common-type $type)
   (> (type-frequency $type 0) 3))

;; Function pairs that can compose (output of one matches input of another)
(= (composable-functions $func1 $func2)
   (case (match &self (: $func1 (-> $params1 $return))
                (match &self (function-param $func2 $idx $name $param-type)
                       (= $return $param-type)))
     (Empty False)
     ($return True)))

;; =====================
;; Code Patterns & Idioms
;; =====================

;; Identify iterate-and-accumulate pattern
(= (iterate-accumulate-pattern $func)
   (match &self (function-def $func $scope $start $end)
          (loop-pattern $id for $scope $loop-line)
          (bin-op Add $target $source $scope $op-line)
          (and (>= $loop-line $start) (<= $loop-line $end)
               (>= $op-line $start) (<= $op-line $end)
               (> $op-line $loop-line))))

;; Identify string formatting pattern  
(= (string-formatting-pattern $func)
   (match &self (function-def $func $scope $start $end)
          (bin-op Add String $right $scope $line)
          (and (>= $line $start) (<= $line $end))))

;; Identify dict lookup pattern
(= (dict-lookup-pattern $func)
   (match &self (function-def $func $scope $start $end)
          (function-call "get" 3 $scope $line)
          (and (>= $line $start) (<= $line $end))))

;; =====================
;; Module Structure Analysis
;; =====================

;; Count functions per module/scope
(= (module-function-count $scope $count)
   (case (match &self (function-def $name $scope $start $end)
                (aggregate-count))
     (Empty 0)
     ($count $count)))

;; Module with high function count (potential monolith)
(= (high-function-module $scope $threshold)
   (> (module-function-count $scope 0) $threshold))

;; Module dependencies count
(= (module-dependency-count $scope $count)
   (case (match &self (import $module $scope $line)
                (aggregate-count))
     (Empty 0)
     ($count $count)))

;; Highly dependent module
(= (highly-dependent-module $scope $threshold)
   (> (module-dependency-count $scope 0) $threshold))

;; =====================
;; Domain Analysis 
;; =====================

;; Domain-specific type identification
(= (domain-specific-type $type)
   (and (not (= $type "String"))
        (not (= $type "Number"))
        (not (= $type "Bool"))
        (not (= $type "List"))
        (not (= $type "Dict"))
        (not (= $type "Tuple"))
        (not (= $type "Set"))
        (not (= $type "Any"))
        (not (= $type "None"))))

;; Module might represent a domain concept
(= (domain-module $scope)
   (> (module-function-count $scope 0) 5))

;; ===================== 
;; Type safety validation rules
;; =====================

;; Detect binary operation type mismatch
(= (binary-op-type-mismatch $op $left-type $right-type $scope $line)
   (match &self (bin-op $op $left-type $right-type $scope $line)
          (not (compatible-types-for-op $op $left-type $right-type))))

;; Define type compatibility for different operations
(= (compatible-types-for-op Add String String) True)
(= (compatible-types-for-op Add String Number) False)
(= (compatible-types-for-op Add Number String) False)
(= (compatible-types-for-op Add Number Number) True)
(= (compatible-types-for-op Sub Number Number) True)
(= (compatible-types-for-op Sub Any Any) False)
(= (compatible-types-for-op Mult Number Number) True)
(= (compatible-types-for-op Mult String Number) True)
(= (compatible-types-for-op Mult Number String) True)
(= (compatible-types-for-op Div Number Number) True)
(= (compatible-types-for-op Div Any Any) False)

;; Detect function call parameter type mismatch
(= (function-param-type-mismatch $func $arg-idx $expected-type $actual-type $call-scope $call-line)
   (match &self (function-call $func $args $call-scope $call-line)
          (function-param $func $arg-idx $param-name $expected-type)
          (parameter-type-at-call $func $arg-idx $call-scope $call-line $actual-type)
          (not (compatible-types $expected-type $actual-type))))

;; Helper to determine parameter type at call site (simplified)
(= (parameter-type-at-call $func $idx $scope $line $type)
   (match &self (function-call-param-type $func $idx $scope $line $type)))

;; Type compatibility rules
(= (compatible-types $type1 $type2) 
   (= $type1 $type2))
(= (compatible-types Any $type) True)
(= (compatible-types $type Any) True)
(= (compatible-types Number String) False)
(= (compatible-types String Number) False)
(= (compatible-types Bool Number) False)
(= (compatible-types Number Bool) False)

;; Detect potential division by zero
(= (potential-division-by-zero $scope $line)
   (match &self (bin-op Div $left-type $right-type $scope $line)
          (match &self (zero-value-in-expression $right-type $scope $line))))

;; Simplified zero value detection
(= (zero-value-in-expression $type $scope $line)
   (match &self (literal-zero-value $scope $line)))

;; Detect null/None dereference
(= (potential-null-dereference $scope $line)
   (match &self (function-call $func $args $scope $line)
          (match &self (null-value-in-args $func $args $scope $line))))

;; Return type mismatch
(= (return-type-mismatch $func $expected $actual $line)
   (match &self (function-def $func $scope $start $end)
          (: $func (-> $params $expected))
          (function-return $func $actual $line)
          (not (compatible-types $expected $actual))))

;; =====================
;; Additional Utility Rules
;; =====================

;; Start with utility for string comparison
(= (starts-with $str $prefix)
   (= $str $prefix))  ;; Simplified for demonstration

;; Check if a list contains an element
(= (contains $list $item)
   (case $list
     (Empty False)
     ((cons $head $tail) (if (= $head $item)
                             True
                             (contains $tail $item)))))

;; String concatenation (simplified)
(= (concat $a $b)
   (concat-strings $a $b))

;; Add an item to a list if not already present
(= (add-unique $list $item)
   (if (contains $list $item)
       $list
       (cons $item $list)))

;; Remove an item from a list
(= (remove $item $list)
   (case $list
     (Empty Empty)
     ((cons $head $tail) (if (= $head $item)
                             (remove $item $tail)
                             (cons $head (remove $item $tail))))))