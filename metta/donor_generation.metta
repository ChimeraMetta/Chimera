;; ========================================
;; Donor Generation Ontology - MeTTa Rules
;; Pure MeTTa-powered donor generation system
;; ========================================

;; ========================================
;; Type Definitions
;; ========================================

(: Pattern Type)
(: Strategy Type)
(: Property Type)
(: DonorCandidate Type)
(: Transformation Type)

;; Pattern types
(: iterate-accumulate-pattern Pattern)
(: search-pattern Pattern)
(: transformation-pattern Pattern)
(: bounds-checking-pattern Pattern)

;; Strategy types
(: operation-substitution Strategy)
(: accumulator-variation Strategy)
(: condition-variation Strategy)
(: structure-preservation Strategy)
(: property-guided Strategy)
(: pattern-expansion Strategy)

;; Property types
(: bounds-checked Property)
(: iterative Property)
(: accumulative Property)
(: error-handling Property)
(: termination-guaranteed Property)
(: safe Property)
(: recursive Property)
(: windowed Property)
(: multi-dimensional Property)

;; Transformation types
(: max-to-min Transformation)
(: min-to-max Transformation)
(: to-sum Transformation)
(: to-count Transformation)
(: to-product Transformation)
(: to-average Transformation)
(: value-to-index Transformation)
(: to-predicate Transformation)
(: to-threshold Transformation)
(: enhanced-bounds-check Transformation)
(: recursive-variant Transformation)
(: multi-range Transformation)
(: windowed Transformation)

;; ========================================
;; Pattern Detection Rules
;; ========================================

;; Iterate-accumulate pattern detection
(= (has-iterate-accumulate-pattern $func)
   (and (has-loop-structure $func)
        (has-accumulator-variable $func)
        (has-bounds-checking $func)))

;; Loop structure detection
(= (has-loop-structure $func)
   (match &self (loop-pattern $id $type $scope $line)
          (contains-function $scope $func)))

;; Bounds checking detection - requires both upper and lower bounds
(= (has-bounds-checking $func)
   (and (match &self (bin-op Lt $left $right $scope $line)
               (contains-function $scope $func))
        (match &self (bin-op Gt $left $right $scope $line)
               (contains-function $scope $func))))

;; Accumulator variable detection (inferred from comparison operations)
(= (has-accumulator-variable $func)
   (or (match &self (bin-op Gt $left $right $scope $line)
              (contains-function $scope $func))
       (match &self (bin-op Lt $left $right $scope $line)
              (contains-function $scope $func))))

;; Search pattern detection
(= (has-search-pattern $func)
   (and (has-conditional-return $func)
        (has-comparison-operations $func)
        (not (has-accumulator-variable $func))))

;; Conditional return detection
(= (has-conditional-return $func)
   (match &self (function-return $func $type $line)
          (> $line 1)))

;; Comparison operations detection
(= (has-comparison-operations $func)
   (match &self (bin-op $op $left $right $scope $line)
          (and (contains-function $scope $func)
               (comparison-operator $op))))

;; Define comparison operators
(= (comparison-operator Lt) True)
(= (comparison-operator Gt) True)
(= (comparison-operator LtE) True)
(= (comparison-operator GtE) True)
(= (comparison-operator Eq) True)
(= (comparison-operator NotEq) True)

;; Helper: Check if scope contains function
(= (contains-function $scope $func)
   (or (= $scope $func)
       (= $scope (concat $func "-scope"))
       (= $scope (concat "function-" $func))
       (= $scope (concat "module-main"))))

;; Enhanced pattern detection with confidence scoring
(= (pattern-confidence iterate-accumulate-pattern $func $confidence)
   (if (and (has-loop-structure $func)
            (has-accumulator-variable $func)
            (has-bounds-checking $func))
       (= $confidence 0.9)
       (if (and (has-loop-structure $func)
                (has-accumulator-variable $func))
           (= $confidence 0.7)
           (= $confidence 0.0))))

;; ========================================
;; Property Assignment Rules
;; ========================================

;; Functions with bounds checking have bounds-checked property
(= (function-has-property $func bounds-checked)
   (has-bounds-checking $func))

;; Functions with loops have iterative property
(= (function-has-property $func iterative)
   (has-loop-structure $func))

;; Functions with accumulator variables have accumulative property
(= (function-has-property $func accumulative)
   (has-accumulator-variable $func))

;; Functions with conditional returns have error-handling property
(= (function-has-property $func error-handling)
   (has-conditional-return $func))

;; All functions are assumed to have termination-guaranteed property
(= (function-has-property $func termination-guaranteed)
   (match &self (function-def $func $scope $start $end) True))

;; Functions with max-finding operations
(= (function-has-property $func max-finding)
   (and (match &self (bin-op Gt $left $right $scope $line)
               (contains-function $scope $func))
        (match &self (function-def $func $scope $start $end) True)))

;; Functions with min-finding operations
(= (function-has-property $func min-finding)
   (and (match &self (bin-op Lt $left $right $scope $line)
               (contains-function $scope $func))
        (match &self (function-def $func $scope $start $end) True)))

;; ========================================
;; Generation Strategy Rules
;; ========================================

;; Operation substitution applies to functions with comparison operations
(= (strategy-applies operation-substitution $func)
   (has-comparison-operations $func))

;; Accumulator variation applies to iterate-accumulate patterns
(= (strategy-applies accumulator-variation $func)
   (has-iterate-accumulate-pattern $func))

;; Condition variation applies to functions with comparisons
(= (strategy-applies condition-variation $func)
   (has-comparison-operations $func))

;; Structure preservation applies to all functions
(= (strategy-applies structure-preservation $func)
   (match &self (function-def $func $scope $start $end) True))

;; Property guided applies to functions with specific properties
(= (strategy-applies property-guided $func)
   (function-has-property $func bounds-checked))

;; Pattern expansion applies to iterate-accumulate patterns
(= (strategy-applies pattern-expansion $func)
   (has-iterate-accumulate-pattern $func))

;; ========================================
;; Transformation Applicability Rules
;; ========================================

;; Operation substitution transformations
(= (operation-substitution-transform $func max-to-min)
   (and (function-has-property $func max-finding)
        (strategy-applies operation-substitution $func)))

(= (operation-substitution-transform $func min-to-max)
   (and (function-has-property $func min-finding)
        (strategy-applies operation-substitution $func)))

;; Accumulator variation transformations
(= (accumulator-variation-transform $func to-sum)
   (has-iterate-accumulate-pattern $func))

(= (accumulator-variation-transform $func to-count)
   (has-iterate-accumulate-pattern $func))

(= (accumulator-variation-transform $func to-product)
   (has-iterate-accumulate-pattern $func))

(= (accumulator-variation-transform $func to-average)
   (has-iterate-accumulate-pattern $func))

;; Structure preservation transformations
(= (structure-preservation-transform $func value-to-index)
   (and (has-iterate-accumulate-pattern $func)
        (function-has-property $func accumulative)))

;; Condition variation transformations
(= (condition-variation-transform $func to-predicate)
   (has-comparison-operations $func))

(= (condition-variation-transform $func to-threshold)
   (has-comparison-operations $func))

;; Property guided transformations
(= (property-guided-transform $func enhanced-bounds-check)
   (function-has-property $func bounds-checked))

(= (property-guided-transform $func recursive-variant)
   (and (function-has-property $func iterative)
        (function-has-property $func termination-guaranteed)))

;; Pattern expansion transformations
(= (pattern-expansion-transform $func multi-range)
   (has-iterate-accumulate-pattern $func))

(= (pattern-expansion-transform $func windowed)
   (has-iterate-accumulate-pattern $func))

;; ========================================
;; Candidate Generation Rules
;; ========================================

;; Generate candidate using operation substitution
(= (generate-candidate $func operation-substitution max-to-min $candidate)
   (and (operation-substitution-transform $func max-to-min)
        (= $candidate (create-min-variant $func))))

(= (generate-candidate $func operation-substitution min-to-max $candidate)
   (and (operation-substitution-transform $func min-to-max)
        (= $candidate (create-max-variant $func))))

;; Generate candidate using accumulator variation
(= (generate-candidate $func accumulator-variation to-sum $candidate)
   (and (accumulator-variation-transform $func to-sum)
        (= $candidate (create-sum-variant $func))))

(= (generate-candidate $func accumulator-variation to-count $candidate)
   (and (accumulator-variation-transform $func to-count)
        (= $candidate (create-count-variant $func))))

(= (generate-candidate $func accumulator-variation to-product $candidate)
   (and (accumulator-variation-transform $func to-product)
        (= $candidate (create-product-variant $func))))

(= (generate-candidate $func accumulator-variation to-average $candidate)
   (and (accumulator-variation-transform $func to-average)
        (= $candidate (create-average-variant $func))))

;; Generate candidate using structure preservation
(= (generate-candidate $func structure-preservation value-to-index $candidate)
   (and (structure-preservation-transform $func value-to-index)
        (= $candidate (create-index-variant $func))))

;; Generate candidate using condition variation
(= (generate-candidate $func condition-variation to-predicate $candidate)
   (and (condition-variation-transform $func to-predicate)
        (= $candidate (create-predicate-variant $func))))

(= (generate-candidate $func condition-variation to-threshold $candidate)
   (and (condition-variation-transform $func to-threshold)
        (= $candidate (create-threshold-variant $func))))

;; Generate candidate using property guided
(= (generate-candidate $func property-guided enhanced-bounds-check $candidate)
   (and (property-guided-transform $func enhanced-bounds-check)
        (= $candidate (create-enhanced-bounds-variant $func))))

(= (generate-candidate $func property-guided recursive-variant $candidate)
   (and (property-guided-transform $func recursive-variant)
        (= $candidate (create-recursive-variant $func))))

;; Generate candidate using pattern expansion
(= (generate-candidate $func pattern-expansion multi-range $candidate)
   (and (pattern-expansion-transform $func multi-range)
        (= $candidate (create-multi-range-variant $func))))

(= (generate-candidate $func pattern-expansion windowed $candidate)
   (and (pattern-expansion-transform $func windowed)
        (= $candidate (create-windowed-variant $func))))

;; ========================================
;; Ranking and Scoring Rules
;; ========================================

;; Base confidence scores for strategies
(= (strategy-confidence operation-substitution 0.9))
(= (strategy-confidence accumulator-variation 0.85))
(= (strategy-confidence structure-preservation 0.95))
(= (strategy-confidence condition-variation 0.8))
(= (strategy-confidence property-guided 0.85))
(= (strategy-confidence pattern-expansion 0.65))

;; Property preservation scoring
(= (property-preservation-score $candidate $original $score)
   (/ (count-preserved-properties $candidate $original)
      (count-total-properties $original)))

;; Count preserved properties
(= (count-preserved-properties $candidate $original $count)
   (case (match &self (candidate-preserves-property $candidate $original $prop))
     (Empty 0)
     ($prop (+ 1 (count-preserved-properties $candidate $original $count)))))

;; Count total properties
(= (count-total-properties $func $count)
   (case (match &self (function-has-property $func $prop))
     (Empty 0)
     ($prop (+ 1 (count-total-properties $func $count)))))

;; Candidate ranking
(= (rank-candidate $candidate $final-score)
   (+ (strategy-confidence (candidate-strategy $candidate))
      (* 0.2 (property-preservation-score $candidate (original-function)))
      (candidate-quality-bonus $candidate)))

;; Quality bonuses
(= (candidate-quality-bonus $candidate 0.1)
   (candidate-has-docstring $candidate))

(= (candidate-quality-bonus $candidate 0.05)
   (candidate-has-error-handling $candidate))

(= (candidate-quality-bonus $candidate 0.03)
   (candidate-has-type-hints $candidate))

;; ========================================
;; Property Preservation Rules
;; ========================================

;; Operation substitution preserves most properties
(= (transformation-preserves-property operation-substitution max-to-min bounds-checked) True)
(= (transformation-preserves-property operation-substitution max-to-min iterative) True)
(= (transformation-preserves-property operation-substitution max-to-min error-handling) True)
(= (transformation-preserves-property operation-substitution max-to-min termination-guaranteed) True)

(= (transformation-preserves-property operation-substitution min-to-max bounds-checked) True)
(= (transformation-preserves-property operation-substitution min-to-max iterative) True)
(= (transformation-preserves-property operation-substitution min-to-max error-handling) True)
(= (transformation-preserves-property operation-substitution min-to-max termination-guaranteed) True)

;; Accumulator variation preserves structure properties
(= (transformation-preserves-property accumulator-variation to-sum bounds-checked) True)
(= (transformation-preserves-property accumulator-variation to-sum iterative) True)
(= (transformation-preserves-property accumulator-variation to-sum error-handling) True)
(= (transformation-preserves-property accumulator-variation to-sum termination-guaranteed) True)

(= (transformation-preserves-property accumulator-variation to-count bounds-checked) True)
(= (transformation-preserves-property accumulator-variation to-count iterative) True)
(= (transformation-preserves-property accumulator-variation to-count error-handling) True)
(= (transformation-preserves-property accumulator-variation to-count termination-guaranteed) True)

;; Structure preservation preserves ALL properties
(= (transformation-preserves-property structure-preservation value-to-index $prop) True)

;; Property guided explicitly preserves target properties
(= (transformation-preserves-property property-guided enhanced-bounds-check bounds-checked) True)
(= (transformation-preserves-property property-guided enhanced-bounds-check safe) True)

(= (transformation-preserves-property property-guided recursive-variant termination-guaranteed) True)
(= (transformation-preserves-property property-guided recursive-variant recursive) True)

;; Pattern expansion adds new properties
(= (transformation-preserves-property pattern-expansion multi-range bounds-checked) True)
(= (transformation-preserves-property pattern-expansion multi-range multi-dimensional) True)

(= (transformation-preserves-property pattern-expansion windowed bounds-checked) True)
(= (transformation-preserves-property pattern-expansion windowed windowed) True)

;; ========================================
;; Candidate Validation Rules
;; ========================================

;; A candidate is valid if it preserves essential properties
(= (valid-candidate $candidate $original)
   (and (preserves-essential-properties $candidate $original)
        (maintains-correctness $candidate $original)
        (has-proper-error-handling $candidate)))

;; Essential properties that must be preserved
(= (essential-property bounds-checked) True)
(= (essential-property error-handling) True)
(= (essential-property termination-guaranteed) True)

;; Check if candidate preserves essential properties
(= (preserves-essential-properties $candidate $original)
   (forall $prop (and (essential-property $prop)
                     (function-has-property $original $prop))
           (candidate-has-property $candidate $prop)))

;; Correctness maintenance (simplified)
(= (maintains-correctness $candidate $original)
   (same-input-output-behavior $candidate $original))

;; Error handling check
(= (has-proper-error-handling $candidate)
   (candidate-has-property $candidate error-handling))

;; ========================================
;; Donor Selection Rules
;; ========================================

;; Select best donors based on multiple criteria
(= (select-best-donors $candidates $count $selected)
   (take $count (sort-by-score $candidates)))

;; Sort candidates by their final score
(= (sort-by-score $candidates $sorted)
   (sort $candidates (lambda ($a $b) (> (final-score $a) (final-score $b)))))

;; Calculate final score for a candidate
(= (final-score $candidate)
   (+ (base-confidence $candidate)
      (* 0.3 (property-preservation-score $candidate))
      (* 0.2 (code-quality-score $candidate))
      (* 0.1 (novelty-score $candidate))))

;; Code quality scoring
(= (code-quality-score $candidate $score)
   (+ (if (candidate-has-docstring $candidate) 0.2 0.0)
      (if (candidate-has-error-handling $candidate) 0.3 0.0)
      (if (candidate-has-type-hints $candidate) 0.1 0.0)
      (if (candidate-follows-naming-conventions $candidate) 0.1 0.0)))

;; Novelty scoring (encourages diversity)
(= (novelty-score $candidate $score)
   (case (candidate-strategy $candidate)
     (operation-substitution 0.1)
     (accumulator-variation 0.3)
     (structure-preservation 0.2)
     (condition-variation 0.4)
     (property-guided 0.2)
     (pattern-expansion 0.5)))

;; ========================================
;; Compatibility Rules
;; ========================================

;; Check if a donor is compatible with a target context
(= (donor-compatible $donor $target-context)
   (and (type-compatible $donor $target-context)
        (property-compatible $donor $target-context)
        (interface-compatible $donor $target-context)))

;; Type compatibility
(= (type-compatible $donor $target-context)
   (and (compatible-input-types $donor $target-context)
        (compatible-output-types $donor $target-context)))

;; Property compatibility
(= (property-compatible $donor $target-context)
   (forall $prop (required-property $target-context $prop)
           (candidate-has-property $donor $prop)))

;; Interface compatibility
(= (interface-compatible $donor $target-context)
   (and (compatible-parameter-count $donor $target-context)
        (compatible-calling-convention $donor $target-context)))

;; ========================================
;; Learning and Adaptation Rules
;; ========================================

;; Learn from successful donor applications
(= (learn-from-success $donor $target $outcome)
   (if (= $outcome success)
       (increase-confidence $donor (candidate-strategy $donor))
       (decrease-confidence $donor (candidate-strategy $donor))))

;; Adapt generation strategies based on success rate
(= (adapt-strategy $strategy $success-rate)
   (if (> $success-rate 0.8)
       (increase-strategy-weight $strategy)
       (if (< $success-rate 0.4)
           (decrease-strategy-weight $strategy)
           (maintain-strategy-weight $strategy))))

;; Update transformation rules based on feedback
(= (update-transformation-rule $transformation $feedback)
   (case $feedback
     (positive (increase-transformation-confidence $transformation))
     (negative (decrease-transformation-confidence $transformation))
     (neutral (maintain-transformation-confidence $transformation))))

;; ========================================
;; Helper Functions and Placeholders
;; ========================================

;; These would be implemented as grounded functions in a full system
;; For now, they serve as placeholders for the MeTTa reasoning

;; Code generation placeholders
(= (create-min-variant $func) (min-variant-code $func))
(= (create-max-variant $func) (max-variant-code $func))
(= (create-sum-variant $func) (sum-variant-code $func))
(= (create-count-variant $func) (count-variant-code $func))
(= (create-product-variant $func) (product-variant-code $func))
(= (create-average-variant $func) (average-variant-code $func))
(= (create-index-variant $func) (index-variant-code $func))
(= (create-predicate-variant $func) (predicate-variant-code $func))
(= (create-threshold-variant $func) (threshold-variant-code $func))
(= (create-enhanced-bounds-variant $func) (enhanced-bounds-variant-code $func))
(= (create-recursive-variant $func) (recursive-variant-code $func))
(= (create-multi-range-variant $func) (multi-range-variant-code $func))
(= (create-windowed-variant $func) (windowed-variant-code $func))

;; Analysis placeholders
(= (candidate-has-docstring $candidate) (has-docstring-check $candidate))
(= (candidate-has-error-handling $candidate) (has-error-handling-check $candidate))
(= (candidate-has-type-hints $candidate) (has-type-hints-check $candidate))
(= (candidate-follows-naming-conventions $candidate) (naming-convention-check $candidate))

;; Utility placeholders
(= (same-input-output-behavior $a $b) (behavioral-equivalence-check $a $b))
(= (compatible-input-types $a $b) (input-type-compatibility-check $a $b))
(= (compatible-output-types $a $b) (output-type-compatibility-check $a $b))
(= (compatible-parameter-count $a $b) (parameter-count-check $a $b))
(= (compatible-calling-convention $a $b) (calling-convention-check $a $b))

;; ========================================
;; End of Donor Generation Ontology
;; ========================================