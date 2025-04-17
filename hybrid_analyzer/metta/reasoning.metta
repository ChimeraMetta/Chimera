;; Reasoning rules for code analysis and transformation
;; Links patterns to donors and provides recommendations

;; Define pattern types
(: PatternMatch Type)
(: ErrorFix Type)
(: Recommendation Type)

;; Pattern recognition
(= (recognize-pattern (bin-op Add String String $scope $line))
   (PatternMatch string_concat String String))

(= (recognize-pattern (bin-op Add String Number $scope $line))
   (PatternMatch string_format String Number))

(= (recognize-pattern (bin-op Add Number String $scope $line))
   (PatternMatch string_format Number String))

(= (recognize-pattern (bin-op Div Number Number $scope $line))
   (PatternMatch safe_division Number Number))

(= (recognize-pattern (function-call get 2 $scope $line))
   (PatternMatch safe_index List Number))

(= (recognize-pattern (function-call sum 1 $scope $line))
   (PatternMatch list_transform List Number))

;; Error pattern recognition
(= (recognize-error ZeroDivisionError $func)
   (ErrorFix zero_division $func (function-donor safe_division)))

(= (recognize-error IndexError $func)
   (ErrorFix index_bounds $func (function-donor safe_index)))

(= (recognize-error TypeError $func)
   (ErrorFix type_conversion $func type_conversion_fix))

(= (recognize-error AttributeError $func)
   (ErrorFix null_check $func null_check_fix))

;; Find fix based on error pattern
(= (find-fix $func $error_type)
   (case $error_type
      (ZeroDivisionError (recognize-error ZeroDivisionError $func))
      (IndexError (recognize-error IndexError $func))
      (TypeError (recognize-error TypeError $func))
      (AttributeError (recognize-error AttributeError $func))
      ($_ no_known_fix)))

;; Find donor based on pattern
(= (find-donor $pattern)
   (case (recognize-pattern $pattern)
     (Empty None)
     (($operation $left_type $right_type)
      (find-donor-by-operation $operation $left_type $right_type))))

;; Layered donor selection logic 
(= (find-donor-by-operation $operation $left_type $right_type)
   (let $context (current-context)
      (let $context-donor (context-donor $context $operation)
         (case $context-donor
            (Empty
             (let $op-donor (operation-donor $operation $left_type $right_type)
                (case $op-donor
                   (Empty
                    (let $type-donor (type-donor $left_type $right_type)
                         (case $type-donor
                            (Empty
                             (case (function-donor-for-operation $operation)
                                (Empty donor_not_found)
                                ($donor $donor)))
                            ($donor $donor))))
                   ($donor $donor))))
            ($donor $donor)))))

;; Code diagnostics
(= (function-complexity $func)
   (match &self (function-metrics $func $calls $successes $failures $avg_time)
      (case (> $avg_time 0.1)
         (True High)
         (False (case (> $avg_time 0.05)
                  (True Medium)
                  (False Low))))))

(= (function-reliability $func)
   (match &self (function-metrics $func $calls $successes $failures $avg_time)
      (let $failure_rate (/ $failures (+ $successes $failures 0.0001))
         (case (> $failure_rate 0.1)
            (True Low)
            (False (case (> $failure_rate 0.01)
                     (True Medium)
                     (False High)))))))

(= (codebase-report)
   (let $func_count (count-functions)
      (let $class_count (count-classes)
         (let $complex_funcs (get-complex-functions)
            (let $error_funcs (get-error-prone-functions)
               (report $func_count $class_count $complex_funcs $error_funcs))))))

;; Generate recommendations based on static and dynamic analysis
(= (generate-recommendations $func)
   (let $metrics (match &self (function-metrics $func $calls $successes $failures $avg_time)
                    ($calls $successes $failures $avg_time))
      (let $errors (match &self (error-pattern $func $error_type $count) 
                      ($error_type $count))
         (let $patterns (match &self (code-pattern $pattern $desc)
                           (match &self (bin-op $op $left $right $scope $func)
                              ($pattern $desc)))
            (recommendation-list $func $metrics $errors $patterns)))))

;; Specific fix descriptions
(= (fix-description zero_division)
   (Recommendation 
      add_zero_check
      "Add zero check before division"
      (python-snippet if_divisor_not_zero)))

(= (fix-description index_bounds)
   (Recommendation
      add_bounds_check
      "Add bounds checking before accessing list elements"
      (python-snippet if_index_in_bounds)))

(= (fix-description type_conversion)
   (Recommendation
      add_type_conversion
      "Add explicit type conversion"
      (python-snippet str_number_value)))

(= (fix-description null_check)
   (Recommendation
      add_null_check
      "Add None check before attribute access"
      "if obj is not None:\n    result = obj.attribute\nelse:\n    result = default_value"))