;; Base MeTTa definitions for Python Code Analysis
;; Core types and predicates for code reasoning

;; Define atom types
(: Function Type)
(: Class Type)
(: Variable Type)
(: Operation Type)
(: Expression Type)

;; Define relation types
(: contains (-> Any Any Bool))
(: depends-on (-> Any Any Bool))
(: inherits (-> Any Any Bool))
(: calls (-> Any Any Bool))
(: shares-type (-> Any Any Bool))

;; Define operation types
(: bin-op (-> Symbol Symbol Symbol Symbol Number))
(: function-def (-> Symbol Symbol Number Number))
(: function-param (-> Symbol Number Symbol Symbol))
(: function-call (-> Symbol Number Symbol Number))
(: variable-assign (-> Symbol Symbol Number))
(: class-def (-> Symbol Symbol Number Number))
(: class-inherits (-> Symbol Symbol))
(: module-depends (-> Symbol Symbol))
(: function-depends (-> Symbol Symbol))

;; Define execution patterns
(: execution (-> Symbol Symbol Bool))
(: execution-param (-> Symbol Symbol Symbol))
(: execution-return (-> Symbol Symbol))
(: execution-error (-> Symbol Symbol))
(: error (-> Symbol Symbol))
(: error-pattern (-> Symbol Symbol Number))
(: function-metrics (-> Symbol Number Number Number Number))

;; Define helper functions
(= (is-complex-function $func)
   (match &self (function-metrics $func $calls $successes $failures $avg_time)
      (> $avg_time 0.1)))

(= (has-high-failure-rate $func)
   (match &self (function-metrics $func $calls $successes $failures $avg_time)
      (> (/ $failures (+ $successes $failures)) 0.1)))

(= (get-call-graph $func)
   (match &self (function-depends $func $called) $called))

(= (get-dependencies $module)
   (match &self (module-depends $module $dependency) $dependency))

;; Base code patterns
(= (code-pattern string_add string_concatenation)
   (match &self (bin-op Add String $right $scope $line) 
      (code-issue string_concatenation $scope $line)))

(= (code-pattern zero_div zero_division_risk)
   (match &self (bin-op Div $left $right $scope $line)
      (code-issue zero_division_risk $scope $line)))

(= (code-pattern index_access index_error_risk)
   (match &self (function-call get 2 $scope $line)
      (code-issue index_error_risk $scope $line)))

;; Diagnostic functions
(= (count-functions)
   (let $funcs (match &self (function-def $name $scope $start $end) $name)
      (length $funcs)))

(= (count-classes)
   (let $classes (match &self (class-def $name $scope $start $end) $name)
      (length $classes)))

(= (get-complex-functions)
   (match &self (is-complex-function $func) $func))

(= (get-error-prone-functions)
   (match &self (has-high-failure-rate $func) $func))

(= (get-cyclic-dependencies)
   (match &self 
      (and (module-depends $m1 $m2) 
           (module-depends $m2 $m1))
      ($m1 $m2)))