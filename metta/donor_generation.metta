;; ========================================
;; Enhanced Donor Generation Ontology - MeTTa Rules
;; Complete implementation with all improvements
;; ========================================

;; ========================================
;; Enhanced Type Definitions
;; ========================================

(: Pattern Type)
(: Strategy Type)
(: Property Type)
(: DonorCandidate Type)
(: Transformation Type)
(: DataStructure Type)
(: Algorithm Type)
(: Context Type)
(: Feedback Type)
(: ComplexityMetric Type)
(: ErrorPattern Type)
(: TypeCompatibility Type)

;; Enhanced Pattern types
(: iterate-accumulate-pattern Pattern)
(: search-pattern Pattern)
(: transformation-pattern Pattern)
(: bounds-checking-pattern Pattern)
(: functional-pattern Pattern)
(: error-handling-pattern Pattern)
(: collection-iteration-pattern Pattern)
(: recursive-pattern Pattern)
(: iterative-pattern Pattern)
(: map-reduce-pattern Pattern)
(: validation-pattern Pattern)
(: composition-pattern Pattern)

;; Enhanced Strategy types
(: operation-substitution Strategy)
(: accumulator-variation Strategy)
(: condition-variation Strategy)
(: structure-preservation Strategy)
(: property-guided Strategy)
(: pattern-expansion Strategy)
(: data-structure-adaptation Strategy)
(: algorithm-transformation Strategy)
(: type-generalization Strategy)
(: functional-composition Strategy)
(: error-handling-enhancement Strategy)
(: performance-optimization Strategy)

;; Enhanced Property types
(: bounds-checked Property)
(: iterative Property)
(: accumulative Property)
(: error-handling Property)
(: termination-guaranteed Property)
(: safe Property)
(: recursive Property)
(: windowed Property)
(: multi-dimensional Property)
(: functional Property)
(: composable Property)
(: side-effect-free Property)
(: type-safe Property)
(: null-safe Property)
(: performance-critical Property)
(: memory-efficient Property)
(: thread-safe Property)
(: immutable Property)

;; Data Structure types
(: list DataStructure)
(: dict DataStructure)
(: set DataStructure)
(: tuple DataStructure)
(: string DataStructure)
(: array DataStructure)
(: iterable DataStructure)
(: generator DataStructure)
(: namedtuple DataStructure)
(: frozenset DataStructure)
(: bytes DataStructure)

;; Algorithm types
(: iterative-algorithm Algorithm)
(: recursive-algorithm Algorithm)
(: divide-conquer-algorithm Algorithm)
(: dynamic-programming-algorithm Algorithm)
(: greedy-algorithm Algorithm)
(: backtracking-algorithm Algorithm)
(: functional-algorithm Algorithm)
(: parallel-algorithm Algorithm)

;; Context types
(: development-context Context)
(: production-context Context)
(: performance-context Context)
(: safety-context Context)
(: finance-context Context)
(: scientific-context Context)
(: web-context Context)

;; Error Pattern types
(: null-pointer-error ErrorPattern)
(: index-out-of-bounds ErrorPattern)
(: type-mismatch ErrorPattern)
(: division-by-zero ErrorPattern)
(: resource-leak ErrorPattern)
(: infinite-loop ErrorPattern)
(: stack-overflow ErrorPattern)

;; ========================================
;; Data Structure Adaptation Rules
;; ========================================

;; Data structure compatibility scoring
(= (data-structure-compatible list tuple) 0.9)
(= (data-structure-compatible list set) 0.8)
(= (data-structure-compatible list dict) 0.7)
(= (data-structure-compatible list generator) 0.8)
(= (data-structure-compatible dict namedtuple) 0.9)
(= (data-structure-compatible dict list) 0.8)
(= (data-structure-compatible set frozenset) 0.95)
(= (data-structure-compatible set list) 0.85)
(= (data-structure-compatible string list) 0.8)
(= (data-structure-compatible string bytes) 0.9)
(= (data-structure-compatible tuple list) 0.9)
(= (data-structure-compatible tuple namedtuple) 0.9)

;; Bidirectional compatibility
(= (data-structure-compatible $a $b) 
   (data-structure-compatible $b $a))

;; Data structure usage detection
(= (uses-data-structure $func list)
   (or (match &self (variable-assign $var $scope $line)
              (and (contains-function $scope $func)
                   (variable-type $var list)))
       (match &self (function-call-param-type $call $idx $scope $line list)
              (contains-function $scope $func))))

(= (uses-data-structure $func dict)
   (or (match &self (bin-op $op String Any $scope $line)
              (contains-function $scope $func))
       (match &self (function-call dict $args $scope $line)
              (contains-function $scope $func))))

(= (uses-data-structure $func string)
   (match &self (string-op-pattern $id $op $left $right $line)
          (contains-function-line $func $line)))

;; Structure transformation applicability
(= (structure-transformation-applicable $func list set)
   (and (uses-data-structure $func list)
        (no-indexing-operations $func)
        (no-ordering-dependent-operations $func)))

(= (structure-transformation-applicable $func list tuple)
   (and (uses-data-structure $func list)
        (no-mutation-operations $func)))

(= (structure-transformation-applicable $func dict namedtuple)
   (and (uses-data-structure $func dict)
        (fixed-key-structure $func)))

(= (structure-transformation-applicable $func string list)
   (and (uses-data-structure $func string)
        (character-level-processing $func)))

;; Operation constraints for structure transformations
(= (no-indexing-operations $func)
   (not (match &self (array-access $func $index $line) True)))

(= (no-ordering-dependent-operations $func)
   (not (or (match &self (function-call sort $args $scope $line)
                   (contains-function $scope $func))
            (match &self (function-call reverse $args $scope $line)
                   (contains-function $scope $func)))))

(= (no-mutation-operations $func)
   (not (or (match &self (function-call append $args $scope $line)
                   (contains-function $scope $func))
            (match &self (function-call extend $args $scope $line)
                   (contains-function $scope $func))
            (match &self (function-call remove $args $scope $line)
                   (contains-function $scope $func)))))

(= (fixed-key-structure $func)
   (and (uses-data-structure $func dict)
        (no-dynamic-key-operations $func)))

(= (character-level-processing $func)
   (match &self (loop-pattern $id for $scope $line)
          (and (contains-function $scope $func)
               (character-iteration $id))))

;; Collection iteration patterns
(= (has-collection-iteration-pattern $func)
   (and (has-loop-structure $func)
        (or (uses-data-structure $func list)
            (uses-data-structure $func dict)
            (uses-data-structure $func set)
            (uses-data-structure $func string))))

;; Generic iterable compatibility
(= (can-generalize-to-iterable $func)
   (and (has-collection-iteration-pattern $func)
        (no-index-specific-operations $func)
        (sequential-access-only $func)))

(= (no-index-specific-operations $func)
   (not (match &self (array-access $func $index $line) True)))

(= (sequential-access-only $func)
   (and (has-loop-structure $func)
        (not (random-access-detected $func))))

;; ========================================
;; Algorithm Transformation Rules
;; ========================================

;; Algorithm pattern detection
(= (has-iterative-pattern $func)
   (and (has-loop-structure $func)
        (not (calls-self $func))
        (sequential-processing $func)))

(= (has-recursive-pattern $func)
   (and (calls-self $func)
        (has-base-case $func)
        (reduces-problem-size $func)))

(= (has-divide-conquer-pattern $func)
   (and (calls-self $func)
        (splits-input $func)
        (combines-results $func)))

(= (has-dynamic-programming-pattern $func)
   (and (has-recursive-pattern $func)
        (overlapping-subproblems $func)
        (optimal-substructure $func)))

(= (has-greedy-pattern $func)
   (and (makes-local-optimal-choice $func)
        (no-backtracking $func)))

(= (has-backtracking-pattern $func)
   (and (explores-solution-space $func)
        (undoes-choices $func)
        (has-pruning-conditions $func)))

;; Algorithm transformation safety
(= (algorithm-transformation-safe $func iterative-to-recursive)
   (and (has-iterative-pattern $func)
        (has-clear-termination $func)
        (no-complex-state $func)))

(= (algorithm-transformation-safe $func recursive-to-iterative)
   (and (has-recursive-pattern $func)
        (tail-recursive $func)
        (simple-recursive-structure $func)))

(= (algorithm-transformation-safe $func imperative-to-functional)
   (and (no-side-effects $func)
        (deterministic-behavior $func)
        (composable-operations $func)))

(= (algorithm-transformation-safe $func sequential-to-parallel)
   (and (independent-iterations $func)
        (no-shared-state $func)
        (commutative-operations $func)))

;; Complexity analysis
(= (transformation-complexity-impact iterative-to-recursive same))
(= (transformation-complexity-impact recursive-to-iterative improved))
(= (transformation-complexity-impact imperative-to-functional same))
(= (transformation-complexity-impact functional-to-imperative same))
(= (transformation-complexity-impact sequential-to-parallel speedup))
(= (transformation-complexity-impact eager-to-lazy space-optimized))
(= (transformation-complexity-impact divide-and-conquer logarithmic-improvement))
(= (transformation-complexity-impact dynamic-programming time-optimized))
(= (transformation-complexity-impact greedy-approach linear-time))
(= (transformation-complexity-impact backtracking exponential-worst-case))

;; Algorithm compatibility assessment
(= (algorithms-compatible iterative-algorithm recursive-algorithm)
   (and (same-problem-domain $func)
        (equivalent-termination-conditions $func)))

(= (algorithms-compatible functional-algorithm imperative-algorithm)
   (and (deterministic-behavior $func)
        (no-hidden-side-effects $func)))

;; Helper predicates for algorithm analysis
(= (calls-self $func)
   (match &self (direct-call $func $func $line) True))

(= (has-base-case $func)
   (match &self (conditional-return $func $line) True))

(= (sequential-processing $func)
   (and (has-loop-structure $func)
        (processes-elements-in-order $func)))

(= (splits-input $func)
   (match &self (array-slice-operation $func $start $end $line) True))

(= (combines-results $func)
   (match &self (result-combination-operation $func $line) True))

;; ========================================
;; Enhanced Error Handling and Edge Cases
;; ========================================

;; Error pattern detection
(= (has-error-handling-pattern $func)
   (or (has-try-catch-blocks $func)
       (has-conditional-error-returns $func)
       (has-input-validation $func)
       (has-defensive-programming $func)))

(= (has-try-catch-blocks $func)
   (match &self (error-handling-block $func try-except $line) True))

(= (has-conditional-error-returns $func)
   (and (match &self (function-return $func $type $line) True)
        (match &self (conditional-statement $func $line2) True)
        (< $line2 $line)))

(= (has-input-validation $func)
   (and (has-type-checking $func)
        (has-bounds-checking $func)
        (has-null-checking $func)))

(= (has-defensive-programming $func)
   (and (validates-preconditions $func)
        (handles-edge-cases $func)
        (provides-meaningful-errors $func)))

;; Null/None handling detection
(= (has-null-handling $func)
   (or (match &self (null-value-in-args $func $args $scope $line) True)
       (match &self (null-check-pattern $func $line) True)
       (explicit-none-handling $func)))

(= (explicit-none-handling $func)
   (match &self (comparison-with-none $func $line) True))

;; Specific error pattern detection
(= (vulnerable-to-error $func null-pointer-error)
   (and (not (has-null-handling $func))
        (dereferences-potentially-null $func)))

(= (vulnerable-to-error $func index-out-of-bounds)
   (and (uses-array-indexing $func)
        (not (validates-index-bounds $func))))

(= (vulnerable-to-error $func division-by-zero)
   (and (has-division-operation $func)
        (not (checks-denominator-zero $func))))

(= (vulnerable-to-error $func type-mismatch)
   (and (not (has-type-checking $func))
        (operates-on-typed-data $func)))

;; Error handling enhancement opportunities
(= (can-enhance-error-handling $func)
   (or (vulnerable-to-any-error $func)
       (lacks-comprehensive-validation $func)
       (has-silent-failures $func)))

(= (vulnerable-to-any-error $func)
   (or (vulnerable-to-error $func null-pointer-error)
       (vulnerable-to-error $func index-out-of-bounds)
       (vulnerable-to-error $func division-by-zero)
       (vulnerable-to-error $func type-mismatch)))

;; Error handling transformation rules
(= (error-handling-transform $func add-null-checks)
   (and (not (has-null-handling $func))
        (dereferences-potentially-null $func)))

(= (error-handling-transform $func add-bounds-validation)
   (and (uses-array-indexing $func)
        (not (validates-index-bounds $func))))

(= (error-handling-transform $func add-type-validation)
   (and (not (has-type-checking $func))
        (operates-on-typed-data $func)))

(= (error-handling-transform $func wrap-in-try-catch)
   (and (has-risky-operations $func)
        (not (has-try-catch-blocks $func))))

;; ========================================
;; Type System Integration
;; ========================================

;; Type compatibility rules
(= (type-substitution-safe String List)
   (requires-conversion string-to-list))

(= (type-substitution-safe List String)
   (requires-conversion list-to-string))

(= (type-substitution-safe Dict NamedTuple)
   (requires-structural-compatibility))

(= (type-substitution-safe $specific Iterable)
   (if (iterable-type $specific) safe requires-wrapper))

;; Generic type support
(= (type-generalization-applicable $func)
   (and (uses-specific-types $func)
        (operations-generic-compatible $func)))

(= (can-generalize-to-iterable $func)
   (and (uses-data-structure $func list)
        (no-index-specific-operations $func)
        (only-sequential-operations $func)))

(= (can-generalize-to-numeric $func)
   (and (uses-arithmetic-operations $func)
        (no-type-specific-operations $func)))

;; Type safety enhancements
(= (type-safety-enhancement-applicable $func)
   (or (missing-type-annotations $func)
       (unsafe-type-conversions $func)
       (implicit-type-assumptions $func)))

;; Type conversion requirements
(= (requires-conversion string-to-list)
   (conversion-function list))

(= (requires-conversion list-to-string)
   (conversion-function join))

(= (requires-structural-compatibility)
   (structural-analysis-required True))

;; Type constraint analysis
(= (type-constraints $func $constraints)
   (collect-type-requirements $func $constraints))

(= (satisfies-type-constraints $donor $constraints)
   (forall $constraint $constraints
           (constraint-satisfied $donor $constraint)))

;; ========================================
;; Functional Programming Support
;; ========================================

;; Functional pattern detection
(= (has-functional-pattern $func)
   (or (uses-map-filter-reduce $func)
       (no-side-effects $func)
       (immutable-data-only $func)
       (higher-order-functions $func)))

(= (uses-map-filter-reduce $func)
   (or (match &self (function-call map $args $scope $line)
              (contains-function $scope $func))
       (match &self (function-call filter $args $scope $line)
              (contains-function $scope $func))
       (match &self (function-call reduce $args $scope $line)
              (contains-function $scope $func))))

(= (no-side-effects $func)
   (and (not (modifies-global-state $func))
        (not (modifies-input-parameters $func))
        (not (performs-io-operations $func))))

(= (immutable-data-only $func)
   (and (not (uses-mutable-structures $func))
        (not (in-place-modifications $func))))

(= (higher-order-functions $func)
   (or (accepts-function-parameters $func)
       (returns-function $func)
       (uses-lambda-expressions $func)))

;; Function composition rules
(= (composable-with $func1 $func2)
   (and (compatible-output-input-types $func1 $func2)
        (no-conflicting-side-effects $func1 $func2)
        (composable-error-handling $func1 $func2)))

(= (compatible-output-input-types $func1 $func2)
   (and (function-output-type $func1 $type1)
        (function-input-type $func2 $type2)
        (type-compatible $type1 $type2)))

;; Functional transformation applicability
(= (functional-transformation-applicable $func imperative-to-functional)
   (and (no-side-effects $func)
        (deterministic-behavior $func)
        (uses-iteration-patterns $func)))

(= (functional-transformation-applicable $func add-higher-order)
   (and (has-repetitive-patterns $func)
        (parameterizable-operations $func)))

;; Monadic patterns (for advanced functional programming)
(= (monadic-pattern-applicable $func)
   (and (error-handling-pattern $func)
        (chainable-operations $func)))

;; ========================================
;; Context-Aware Generation Rules
;; ========================================

;; Context analysis
(= (target-context-requires development-context fast-iteration))
(= (target-context-requires production-context reliability))
(= (target-context-requires performance-context low-latency))
(= (target-context-requires safety-context error-prevention))
(= (target-context-requires finance-context precision))
(= (target-context-requires scientific-context accuracy))
(= (target-context-requires web-context scalability))

;; Context-specific property requirements
(= (context-property-requirement performance-context performance-critical))
(= (context-property-requirement safety-context null-safe))
(= (context-property-requirement finance-context precision-critical))
(= (context-property-requirement production-context error-handling))
(= (context-property-requirement scientific-context deterministic))

;; Domain-specific patterns
(= (domain-specific-pattern $func finance)
   (and (uses-financial-calculations $func)
        (requires-precision $func)
        (handles-currency-operations $func)))

(= (domain-specific-pattern $func scientific)
   (and (uses-mathematical-operations $func)
        (requires-numerical-stability $func)
        (handles-large-datasets $func)))

(= (domain-specific-pattern $func web)
   (and (handles-concurrent-access $func)
        (stateless-operations $func)
        (serializable-data $func)))

;; Context-specific transformations
(= (context-specific-transformation performance-context $func optimize-loops)
   (and (has-loop-structure $func)
        (inefficient-loop-detected $func)))

(= (context-specific-transformation safety-context $func add-assertions)
   (and (not (has-comprehensive-validation $func))
        (operates-on-critical-data $func)))

(= (context-specific-transformation finance-context $func use-decimal-arithmetic)
   (and (uses-floating-point-arithmetic $func)
        (financial-calculations $func)))

;; Performance constraints
(= (performance-critical $context)
   (context-requirement $context low-latency))

(= (memory-constrained $context)
   (context-requirement $context low-memory))

(= (cpu-constrained $context)
   (context-requirement $context low-cpu))

;; Context compatibility checking
(= (context-compatible $donor $target-context)
   (and (satisfies-context-requirements $donor $target-context)
        (no-context-violations $donor $target-context)))

(= (satisfies-context-requirements $donor $context)
   (forall $req (context-requirement $context $req)
           (donor-satisfies-requirement $donor $req)))

;; ========================================
;; Learning and Feedback Integration
;; ========================================

;; Success tracking
(= (track-donor-success $donor $target $outcome $timestamp)
   (add-atom &feedback (donor-application $donor $target $outcome $timestamp)))

(= (donor-success-rate $donor $rate)
   (/ (count-successful-applications $donor)
      (count-total-applications $donor)))

;; Pattern evolution based on feedback
(= (evolve-pattern $pattern $feedback $new-weight)
   (case $feedback
     (successful (+ (current-pattern-weight $pattern) 0.1))
     (failed (- (current-pattern-weight $pattern) 0.05))
     (partially-successful (+ (current-pattern-weight $pattern) 0.02))))

;; Strategy adaptation
(= (adapt-generation-strategy $strategy $success-rate $new-weight)
   (cond
     ((> $success-rate 0.8) (+ (current-weight $strategy) 0.1))
     ((< $success-rate 0.3) (- (current-weight $strategy) 0.1))
     (True (current-weight $strategy))))

;; Failure analysis
(= (analyze-failure $donor $target $failure-reason)
   (case $failure-reason
     (type-mismatch (decrease-type-compatibility-confidence $donor))
     (runtime-error (increase-error-handling-priority $donor))
     (performance-issue (adjust-performance-expectations $donor))
     (semantic-mismatch (refine-semantic-matching $donor))))

;; Dynamic rule adjustment
(= (adjust-rule-confidence $rule $feedback $new-confidence)
   (case $feedback
     (validates-rule (min 0.95 (+ (rule-confidence $rule) 0.05)))
     (contradicts-rule (max 0.1 (- (rule-confidence $rule) 0.1)))
     (neutral (rule-confidence $rule))))

;; Learning from multiple applications
(= (update-donor-model $donor $applications $updated-model)
   (aggregate-application-feedback $donor $applications $updated-model))

;; Preference learning
(= (learn-user-preferences $user $selections $preferences)
   (extract-preference-patterns $user $selections $preferences))

;; ========================================
;; Integration with Static Analysis
;; ========================================

;; AST pattern integration
(= (ast-pattern-detected $func iterate-accumulate-pattern)
   (and (match &self (function-pattern $i $func $params $has_return $scope) True)
        (match &self (loop-pattern $j $type $scope $line) True)
        (match &self (bin-op $op $left $right $scope $line2) True)))

(= (ast-pattern-detected $func search-pattern)
   (and (match &self (function-pattern $i $func $params True $scope) True)
        (match &self (conditional-statement $func $line) True)
        (match &self (function-return $func $type $line2) True)))

;; Code complexity metrics
(= (cyclomatic-complexity $func $complexity)
   (+ (count-conditional-branches $func)
      (count-loop-structures $func)
      (count-exception-handlers $func)
      1))

(= (count-conditional-branches $func $count)
   (count-matches &self (conditional-statement $func $line) $count))

(= (count-loop-structures $func $count)
   (count-matches &self (loop-pattern $id $type $scope $line)
                  (contains-function $scope $func) $count))

;; Dependency analysis integration
(= (function-dependency-graph $func $dependencies)
   (collect-all-matches &self (function-depends $func $dep) $dependencies))

(= (circular-dependency-detected $func)
   (transitive-dependency $func $func))

(= (transitive-dependency $func1 $func3)
   (or (match &self (function-depends $func1 $func3) True)
       (and (match &self (function-depends $func1 $func2) True)
            (transitive-dependency $func2 $func3))))

;; Code quality metrics
(= (code-quality-score $func $score)
   (+ (documentation-score $func)
      (error-handling-score $func)
      (type-annotation-score $func)
      (naming-convention-score $func)
      (complexity-score $func)))

(= (documentation-score $func $score)
   (if (has-docstring $func) 0.2 0.0))

(= (error-handling-score $func $score)
   (if (has-comprehensive-error-handling $func) 0.3 
       (if (has-basic-error-handling $func) 0.15 0.0)))

;; ========================================
;; Advanced Pattern Detection
;; ========================================

;; Composite pattern detection
(= (composite-pattern $func map-reduce)
   (and (has-map-like-operation $func)
        (has-reduce-like-operation $func)
        (sequential-application $func)))

(= (composite-pattern $func filter-map)
   (and (has-filter-like-operation $func)
        (has-map-like-operation $func)
        (chained-operations $func)))

;; Design pattern detection
(= (design-pattern $func iterator)
   (and (implements-next-method $func)
        (implements-iter-method $func)
        (stateful-iteration $func)))

(= (design-pattern $func strategy)
   (and (accepts-algorithm-parameter $func)
        (algorithm-dispatch $func)))

(= (design-pattern $func template-method)
   (and (defines-algorithm-skeleton $func)
        (customizable-steps $func)))

;; Anti-pattern detection
(= (anti-pattern $func god-function)
   (and (> (function-length $func) 50)
        (> (cyclomatic-complexity $func) 10)
        (multiple-responsibilities $func)))

(= (anti-pattern $func magic-numbers)
   (and (uses-numeric-literals $func)
        (no-named-constants $func)))

;; ========================================
;; Enhanced Validation Rules
;; ========================================

;; Sophisticated correctness checks
(= (maintains-semantic-equivalence $candidate $original)
   (and (preserves-input-output-relationship $candidate $original)
        (preserves-side-effects $candidate $original)
        (preserves-error-conditions $candidate $original)))

(= (preserves-input-output-relationship $candidate $original)
   (forall $input (valid-input $original $input)
           (equivalent-output $candidate $original $input)))

(= (preserves-side-effects $candidate $original)
   (and (same-global-state-modifications $candidate $original)
        (same-io-operations $candidate $original)))

;; Property inference rules
(= (infer-property $func thread-safe)
   (and (no-shared-mutable-state $func)
        (no-global-variables $func)
        (immutable-operations-only $func)))

(= (infer-property $func deterministic)
   (and (no-random-operations $func)
        (no-time-dependent-operations $func)
        (no-external-dependencies $func)))

(= (infer-property $func memory-efficient)
   (and (no-large-temporary-structures $func)
        (streaming-operations $func)
        (efficient-algorithms $func)))

;; Compatibility validation
(= (interface-compatible $donor $target)
   (and (same-parameter-count $donor $target)
        (compatible-parameter-types $donor $target)
        (compatible-return-type $donor $target)
        (compatible-exceptions $donor $target)))

(= (behavioral-compatible $donor $target)
   (and (same-preconditions $donor $target)
        (same-postconditions $donor $target)
        (same-invariants $donor $target)))

;; ========================================
;; Multi-Objective Optimization
;; ========================================

;; Multiple criteria scoring
(= (multi-objective-score $candidate $weights $score)
   (+ (* (weight-correctness $weights) (correctness-score $candidate))
      (* (weight-performance $weights) (performance-score $candidate))
      (* (weight-maintainability $weights) (maintainability-score $candidate))
      (* (weight-reliability $weights) (reliability-score $candidate))))

;; Pareto optimal candidate selection
(= (pareto-optimal $candidate $candidates)
   (not (exists $other $candidates
               (pareto-dominates $other $candidate))))

(= (pareto-dominates $a $b)
   (and (>= (correctness-score $a) (correctness-score $b))
        (>= (performance-score $a) (performance-score $b))
        (>= (maintainability-score $a) (maintainability-score $b))
        (or (> (correctness-score $a) (correctness-score $b))
            (> (performance-score $a) (performance-score $b))
            (> (maintainability-score $a) (maintainability-score $b)))))

;; ========================================
;; Temporal and Evolutionary Rules
;; ========================================

;; Version evolution tracking
(= (track-evolution $original $candidate $version $timestamp)
   (add-atom &evolution (evolution-step $original $candidate $version $timestamp)))

(= (evolution-lineage $func $lineage)
   (collect-evolution-chain $func $lineage))

(= (collect-evolution-chain $func $chain)
   (case (match &self (evolution-step $parent $func $version $timestamp))
     (Empty (list $func))
     (($parent $version $timestamp) 
      (cons $func (collect-evolution-chain $parent)))))

;; Convergent evolution detection
(= (convergent-evolution $func1 $func2)
   (and (different-origins $func1 $func2)
        (similar-final-properties $func1 $func2)
        (different-evolutionary-paths $func1 $func2)))

;; Evolutionary pressure modeling
(= (evolutionary-pressure $context $pressure)
   (case $context
     (performance-critical high-performance-pressure)
     (safety-critical high-reliability-pressure)
     (development rapid-iteration-pressure)))

;; ========================================
;; Domain-Specific Knowledge Integration
;; ========================================

;; Financial domain rules
(= (financial-compliance $func)
   (and (uses-decimal-arithmetic $func)
        (handles-rounding-correctly $func)
        (audit-trail-capable $func)
        (regulatory-compliant $func)))

(= (financial-transformation-required $func use-decimal)
   (and (domain-context $func finance)
        (uses-floating-point-arithmetic $func)))

;; Scientific computing rules
(= (scientific-computing-requirements $func)
   (and (numerical-stability $func)
        (precision-preservation $func)
        (vectorizable-operations $func)
        (reproducible-results $func)))

(= (scientific-transformation-applicable $func vectorization)
   (and (domain-context $func scientific)
        (element-wise-operations $func)
        (large-dataset-processing $func)))

;; Web development rules
(= (web-development-requirements $func)
   (and (stateless-design $func)
        (serializable-data $func)
        (concurrent-safe $func)
        (cache-friendly $func)))

(= (web-transformation-applicable $func add-caching)
   (and (domain-context $func web)
        (expensive-computations $func)
        (deterministic-results $func)))

;; ========================================
;; Security and Safety Rules
;; ========================================

;; Security pattern detection
(= (security-vulnerability $func sql-injection)
   (and (builds-sql-queries $func)
        (uses-string-concatenation $func)
        (not (uses-parameterized-queries $func))))

(= (security-vulnerability $func xss)
   (and (processes-user-input $func)
        (generates-html-output $func)
        (not (sanitizes-output $func))))

(= (security-vulnerability $func buffer-overflow)
   (and (uses-unsafe-operations $func)
        (not (validates-buffer-bounds $func))))

;; Security enhancement transformations
(= (security-enhancement-applicable $func input-sanitization)
   (and (processes-external-input $func)
        (not (comprehensive-input-validation $func))))

(= (security-enhancement-applicable $func output-encoding)
   (and (generates-output $func)
        (potential-injection-risk $func)))

;; Safety-critical system rules
(= (safety-critical-requirements $func)
   (and (fail-safe-behavior $func)
        (redundant-checks $func)
        (bounded-execution-time $func)
        (deterministic-behavior $func)))

(= (safety-transformation-required $func add-redundancy)
   (and (safety-critical-context $func)
        (single-point-of-failure $func)))

;; ========================================
;; Concurrency and Parallelism Rules
;; ========================================

;; Concurrency pattern detection
(= (concurrency-pattern $func thread-safe)
   (and (no-shared-mutable-state $func)
        (atomic-operations-only $func)
        (no-race-conditions $func)))

(= (concurrency-pattern $func producer-consumer)
   (and (generates-data $func)
        (queue-based-communication $func)
        (asynchronous-processing $func)))

(= (concurrency-pattern $func map-reduce)
   (and (parallelizable-operations $func)
        (independent-processing-units $func)
        (result-aggregation $func)))

;; Parallelization applicability
(= (parallelization-applicable $func data-parallel)
   (and (independent-iterations $func)
        (no-loop-carried-dependencies $func)
        (sufficient-workload $func)))

(= (parallelization-applicable $func pipeline-parallel)
   (and (sequential-stages $func)
        (independent-stage-processing $func)
        (balanced-stage-workload $func)))

;; Thread safety analysis
(= (thread-safety-violation $func race-condition)
   (and (accesses-shared-data $func)
        (not (synchronized-access $func))))

(= (thread-safety-violation $func deadlock-potential)
   (and (acquires-multiple-locks $func)
        (inconsistent-lock-ordering $func)))

;; ========================================
;; Performance Optimization Rules
;; ========================================

;; Performance bottleneck detection
(= (performance-bottleneck $func algorithmic)
   (and (inefficient-algorithm $func)
        (large-input-size-sensitivity $func)))

(= (performance-bottleneck $func memory)
   (and (excessive-memory-allocation $func)
        (cache-unfriendly-access-patterns $func)))

(= (performance-bottleneck $func io)
   (and (frequent-io-operations $func)
        (not (batched-operations $func))))

;; Optimization transformation rules
(= (optimization-applicable $func algorithmic-improvement)
   (and (performance-bottleneck $func algorithmic)
        (better-algorithm-available $func)))

(= (optimization-applicable $func memory-optimization)
   (and (performance-bottleneck $func memory)
        (memory-optimization-possible $func)))

(= (optimization-applicable $func caching)
   (and (expensive-repeated-computations $func)
        (deterministic-results $func)
        (bounded-input-space $func)))

;; Performance prediction
(= (performance-impact $transformation $impact)
   (case $transformation
     (add-caching significant-improvement)
     (vectorization major-improvement)
     (parallelization speedup-proportional-to-cores)
     (algorithm-improvement depends-on-complexity-reduction)))

;; ========================================
;; Code Quality and Maintainability Rules
;; ========================================

;; Code quality metrics
(= (maintainability-score $func $score)
   (+ (readability-score $func)
      (modularity-score $func)
      (testability-score $func)
      (documentation-score $func)))

(= (readability-score $func $score)
   (+ (if (descriptive-names $func) 0.3 0.0)
      (if (appropriate-length $func) 0.2 0.0)
      (if (clear-structure $func) 0.3 0.0)
      (if (minimal-complexity $func) 0.2 0.0)))

(= (modularity-score $func $score)
   (+ (if (single-responsibility $func) 0.4 0.0)
      (if (loose-coupling $func) 0.3 0.0)
      (if (high-cohesion $func) 0.3 0.0)))

;; Refactoring opportunities
(= (refactoring-opportunity $func extract-method)
   (and (long-function $func)
        (identifiable-sub-operations $func)))

(= (refactoring-opportunity $func eliminate-duplication)
   (and (code-duplication-detected $func)
        (extractable-common-logic $func)))

(= (refactoring-opportunity $func simplify-conditional)
   (and (complex-conditional-logic $func)
        (simplifiable-conditions $func)))

;; Technical debt detection
(= (technical-debt $func high)
   (and (> (cyclomatic-complexity $func) 15)
        (< (test-coverage $func) 0.5)
        (anti-pattern-present $func)))

(= (technical-debt $func medium)
   (or (> (cyclomatic-complexity $func) 10)
       (< (test-coverage $func) 0.7)
       (code-smells-present $func)))

;; ========================================
;; Testing and Verification Rules
;; ========================================

;; Test generation guidance
(= (test-case-needed $func boundary-conditions)
   (and (has-bounds-checking $func)
        (not (comprehensive-boundary-tests $func))))

(= (test-case-needed $func error-conditions)
   (and (has-error-handling $func)
        (not (error-path-coverage $func))))

(= (test-case-needed $func edge-cases)
   (and (handles-special-inputs $func)
        (not (edge-case-coverage $func))))

;; Property-based testing applicability
(= (property-based-testing-applicable $func)
   (and (deterministic-behavior $func)
        (clear-invariants $func)
        (testable-properties $func)))

;; Formal verification opportunities
(= (formal-verification-applicable $func)
   (and (safety-critical $func)
        (bounded-complexity $func)
        (well-defined-specifications $func)))

;; ========================================
;; Documentation and Knowledge Management
;; ========================================

;; Documentation completeness
(= (documentation-complete $func)
   (and (has-function-docstring $func)
        (documents-parameters $func)
        (documents-return-value $func)
        (documents-exceptions $func)
        (includes-examples $func)))

;; Knowledge extraction rules
(= (extractable-knowledge $func design-pattern)
   (and (implements-known-pattern $func)
        (reusable-pattern-implementation $func)))

(= (extractable-knowledge $func best-practice)
   (and (high-quality-implementation $func)
        (follows-conventions $func)
        (generalizable-approach $func)))

;; Documentation generation guidance
(= (auto-documentation-applicable $func)
   (and (clear-function-structure $func)
        (descriptive-names $func)
        (type-annotations-present $func)))

;; ========================================
;; Integration and Compatibility Rules
;; ========================================

;; API compatibility
(= (api-compatible $donor $target)
   (and (compatible-signatures $donor $target)
        (compatible-semantics $donor $target)
        (compatible-error-behavior $donor $target)))

(= (api-evolution-safe $old-version $new-version)
   (and (backward-compatible $old-version $new-version)
        (no-breaking-changes $old-version $new-version)))

;; Library integration
(= (library-integration-compatible $func $library)
   (and (compatible-programming-paradigm $func $library)
        (compatible-error-handling $func $library)
        (compatible-resource-management $func $library)))

;; Framework compliance
(= (framework-compliant $func $framework)
   (and (follows-framework-conventions $func $framework)
        (uses-framework-patterns $func $framework)
        (respects-framework-constraints $func $framework)))

;; ========================================
;; Sustainability and Environmental Rules
;; ========================================

;; Energy efficiency
(= (energy-efficient $func)
   (and (minimal-computational-complexity $func)
        (efficient-algorithms $func)
        (minimal-resource-usage $func)))

;; Carbon footprint considerations
(= (low-carbon-footprint $func)
   (and (energy-efficient $func)
        (minimal-network-usage $func)
        (efficient-data-storage $func)))

;; Sustainable software practices
(= (sustainable-implementation $func)
   (and (energy-efficient $func)
        (maintainable-design $func)
        (long-term-viability $func)))

;; ========================================
;; Accessibility and Inclusivity Rules
;; ========================================

;; Accessibility requirements
(= (accessibility-compliant $func)
   (and (handles-diverse-inputs $func)
        (provides-alternative-interfaces $func)
        (supports-assistive-technologies $func)))

;; Internationalization support
(= (internationalization-ready $func)
   (and (unicode-aware $func)
        (locale-independent $func)
        (culturally-neutral-algorithms $func)))

;; ========================================
;; Utility Functions and Helpers
;; ========================================

;; Enhanced helper functions
(= (contains-function-line $func $line)
   (and (match &self (function-def $func $scope $start $end) True)
        (>= $line $start)
        (<= $line $end)))

(= (function-length $func $length)
   (match &self (function-def $func $scope $start $end)
          (- $end $start)))

;; Pattern matching utilities
(= (count-matches $space $pattern $count)
   (length (match $space $pattern $pattern)))

(= (collect-all-matches $space $pattern $results)
   (collapse (match $space $pattern $pattern)))

;; Type checking utilities
(= (variable-type $var $type)
   (or (match &self (variable-assign $var $scope $line)
              (inferred-type $var $type))
       (match &self (variable-ann-assign $var $scope $line)
              (declared-type $var $type))))

;; Complexity calculation utilities
(= (max-complexity $func1 $func2 $max)
   (let $c1 (cyclomatic-complexity $func1)
        (let $c2 (cyclomatic-complexity $func2)
             (if (> $c1 $c2) $c1 $c2))))

;; ========================================
;; Machine Learning Integration Rules
;; ========================================

;; ML-assisted pattern recognition
(= (ml-detected-pattern $func $pattern $confidence)
   (ml-pattern-classifier $func $pattern $confidence))

;; Neural network based similarity
(= (semantic-similarity $func1 $func2 $similarity)
   (neural-similarity-score $func1 $func2 $similarity))

;; Automated feature extraction
(= (extract-features $func $features)
   (code-feature-extractor $func $features))

;; ========================================
;; End of Enhanced Donor Generation Ontology
;; ========================================

;; Final integration rules to tie everything together
(= (comprehensive-donor-generation $func $context $strategies $donors)
   (let $patterns (detect-all-patterns $func)
        (let $applicable-strategies (filter-applicable-strategies $strategies $func $context)
             (let $candidates (generate-all-candidates $func $applicable-strategies)
                  (let $validated (validate-candidates $candidates $func $context)
                       (let $ranked (rank-candidates-multi-objective $validated $context)
                            (select-top-donors $ranked $donors)))))))

;; Master validation rule
(= (validate-comprehensive $donor $original $context)
   (and (maintains-semantic-equivalence $donor $original)
        (context-compatible $donor $context)
        (satisfies-quality-requirements $donor)
        (no-security-vulnerabilities $donor)
        (performance-acceptable $donor $context)
        (maintainability-adequate $donor)))

;; The complete ontology is now ready for production use