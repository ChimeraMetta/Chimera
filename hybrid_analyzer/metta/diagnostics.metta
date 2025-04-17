;; Codebase diagnostics and reporting functions
;; Provides deeper insights into code structure and quality

;; Define diagnostic types
(: FunctionCategory Type)
(: ComplexityLevel Type)
(: QualityIssue Type)
(: CodeReport Type)

;; Function complexity analysis
(= (detect-complex-functions)
   (let $all_funcs (match &self (function-def $name $scope $start $end) 
                     ($name $scope $start (- $end $start)))
      (filter (lambda ($f) (> (nth 3 $f) 20)) $all_funcs)))

(= (categorize-function-length $func_info)
   (let $length (nth 3 $func_info)
      (case (> $length 50)
         (True (cons $func_info VeryLong))
         (False (case (> $length 30)
                  (True (cons $func_info Long))
                  (False (case (> $length 15)
                           (True (cons $func_info Medium))
                           (False (cons $func_info Short)))))))))

;; Module coupling analysis
(= (module-coupling-score)
   (let $modules (match &self (module-depends $m1 $m2) $m1)
      (let $unique_modules (unique $modules)
         (let $dependencies (count (match &self (module-depends $m1 $m2) ($m1 $m2)))
            (/ $dependencies (length $unique_modules))))))

(= (detect-cyclic-dependencies)
   (match &self 
      (and (module-depends $m1 $m2) 
           (module-depends $m2 $m1))
      ($m1 $m2)))

;; Type consistency analysis
(= (detect-inconsistent-types $var_name)
   (let $assignments (match &self (variable-assign $var_name $type $line) ($type $line))
      (let $unique_types (unique (map (lambda ($a) (nth 0 $a)) $assignments))
         (if (> (length $unique_types) 1)
             (QualityIssue inconsistent_type $var_name $unique_types $assignments)
             ()))))

(= (detect-all-inconsistent-types)
   (let $var_names (unique (match &self (variable-assign $name $type $line) $name))
      (filter-empty (map detect-inconsistent-types $var_names))))

;; Code duplication detection (simplified)
(= (detect-similar-functions)
   (let $all_funcs (match &self (function-def $name $scope $start $end) 
                     ($name $scope $start $end))
      (let $func_pairs (combinations $all_funcs 2)
         (filter similar-function-pair $func_pairs))))

(= (similar-function-pair $pair)
   (let $f1 (nth 0 $pair)
      (let $f2 (nth 1 $pair)
         (and (similar-length $f1 $f2)
              (shared-dependencies $f1 $f2)))))

(= (similar-length $f1 $f2)
   (let $len1 (- (nth 3 $f1) (nth 2 $f1))
      (let $len2 (- (nth 3 $f2) (nth 2 $f2))
         (< (abs (- $len1 $len2)) 5))))

(= (shared-dependencies $f1 $f2)
   (let $deps1 (match &self (function-depends (nth 0 $f1) $dep) $dep)
      (let $deps2 (match &self (function-depends (nth 0 $f2) $dep) $dep)
         (> (length (intersection $deps1 $deps2)) 2))))

;; Code quality metrics
(= (calculate-quality-metrics)
   (let $func_count (count-functions)
      (let $class_count (count-classes)
         (let $complex_funcs (length (detect-complex-functions))
            (let $cyclic_deps (length (detect-cyclic-dependencies))
               (let $inconsistent_types (length (detect-all-inconsistent-types))
                  (CodeReport quality_metrics $func_count $class_count $complex_funcs $cyclic_deps $inconsistent_types)))))))

;; Codebase diagnostics report
(= (generate-diagnostics-report)
   (let $metrics (calculate-quality-metrics)
      (let $coupling (module-coupling-score)
         (let $complex_funcs (detect-complex-functions)
            (let $type_issues (detect-all-inconsistent-types)
               (let $cyclic_deps (detect-cyclic-dependencies)
                  (CodeReport diagnostics $metrics $coupling $complex_funcs $type_issues $cyclic_deps)))))))

;; Function execution statistics
(= (execution-statistics $func)
   (match &self (function-metrics $func $calls $successes $failures $avg_time)
      (CodeReport execution_stats $func $calls $successes $failures $avg_time 
                 (/ $failures (max 1 (+ $successes $failures)))
                 (* $avg_time 1000))))

(= (all-execution-statistics)
   (match &self (function-metrics $func $calls $successes $failures $avg_time)
      (CodeReport execution_stats $func $calls $successes $failures $avg_time 
                 (/ $failures (max 1 (+ $successes $failures)))
                 (* $avg_time 1000))))

;; Error pattern analysis
(= (common-error-patterns)
   (let $patterns (match &self (error-pattern $func $error_type $count) 
                    (ErrorPattern $func $error_type $count))
      (sort-by-count $patterns)))

(= (sort-by-count $patterns)
   (sort (lambda ($p1 $p2) (> (nth 2 $p1) (nth 2 $p2))) $patterns))

;; Architectural insights
(= (module-dependency-graph)
   (match &self (module-depends $m1 $m2) 
      (DependencyEdge $m1 $m2)))

(= (class-hierarchy-graph)
   (match &self (class-inherits $child $parent) 
      (InheritanceEdge $parent $child)))

;; Recommendations for code structure improvement
(= (recommend-code-structure-improvements)
   (let $complex_funcs (detect-complex-functions)
      (let $cyclic_deps (detect-cyclic-dependencies)
         (let $type_issues (detect-all-inconsistent-types)
            (CodeReport structure_recommendations $complex_funcs $cyclic_deps $type_issues)))))

;; Specific structure recommendations
(= (recommend-for-complex-function $func $length)
   (Recommendation
      refactor_complex_function
      (RefactorTarget $func)
      (RecommendationReason function_too_long $length multiple_responsibilities)
      (RefactorAction split_into_smaller_functions single_responsibility)))

(= (recommend-for-cyclic-dependency $m1 $m2)
   (Recommendation
      break_cyclic_dependency
      (RefactorTarget $m1 $m2)
      (RecommendationReason circular_dependencies tight_coupling)
      (RefactorAction extract_common_functionality new_module)))

(= (recommend-for-inconsistent-types $var $types)
   (Recommendation
      enforce_type_consistency
      (RefactorTarget $var)
      (RecommendationReason inconsistent_types $types)
      (RefactorAction add_type_annotations use_consistent_return_type)))