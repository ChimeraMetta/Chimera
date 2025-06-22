; MeTTa Rules for Semantic Evolution
; These rules guide the semantic evolution process with logical reasoning

; =============================================================================
; Core Semantic Evolution Rules
; =============================================================================

; Function purpose classification
(= (function-purpose maximize $func)
   (or (contains-name $func "max")
       (contains-name $func "maximum")
       (contains-name $func "largest")
       (contains-name $func "highest")))

(= (function-purpose minimize $func)
   (or (contains-name $func "min")
       (contains-name $func "minimum")
       (contains-name $func "smallest")
       (contains-name $func "lowest")))

(= (function-purpose search $func)
   (or (contains-name $func "find")
       (contains-name $func "search")
       (contains-name $func "locate")
       (contains-name $func "get")))

(= (function-purpose aggregate $func)
   (or (contains-name $func "sum")
       (contains-name $func "total")
       (contains-name $func "average")
       (contains-name $func "count")))

; =============================================================================
; Gene Selection Rules
; =============================================================================

; Gene selection based on purpose
(= (select-gene maximize maximize-condition) True)
(= (select-gene minimize minimize-condition) True)
(= (select-gene search linear-scan) True)
(= (select-gene aggregate accumulate-values) True)

; Gene compatibility rules
(= (genes-compatible $gene1 $gene2)
   (and (postcondition-match $gene1 $gene2)
        (no-semantic-conflict $gene1 $gene2)))

(= (postcondition-match $gene1 $gene2)
   (member-of (postcondition $gene1) (preconditions $gene2)))

(= (no-semantic-conflict maximize-condition minimize-condition) False)
(= (no-semantic-conflict minimize-condition maximize-condition) False)
(= (no-semantic-conflict forward-scan backward-scan) False)
(= (no-semantic-conflict $gene1 $gene2) True)

; =============================================================================
; Crossover and Mutation Rules
; =============================================================================

; Crossover beneficial conditions
(= (crossover-beneficial $parent1 $parent2)
   (and (complementary-genes $parent1 $parent2)
        (semantic-coverage-improved $parent1 $parent2)))

(= (complementary-genes $parent1 $parent2)
   (and (different-strengths $parent1 $parent2)
        (compatible-semantics $parent1 $parent2)))

; Mutation safety rules
(= (mutation-safe $gene $context)
   (and (semantic-role-preserved $gene)
        (preconditions-maintained $gene $context)))

(= (semantic-role-preserved $gene)
   (maintains-purpose $gene))

; =============================================================================
; Fitness Evaluation Rules
; =============================================================================

; Fitness component weights
(= (fitness-weight correctness) 0.6)
(= (fitness-weight efficiency) 0.3)
(= (fitness-weight maintainability) 0.1)

; Correctness evaluation rules
(= (correctness-bonus edge-case-pass) 0.1)
(= (correctness-bonus error-handling) 0.05)
(= (correctness-bonus type-safety) 0.05)

; Efficiency evaluation rules
(= (efficiency-penalty nested-loops) 0.3)
(= (efficiency-penalty excessive-conditions) 0.2)
(= (efficiency-bonus early-termination) 0.1)
(= (efficiency-bonus single-pass) 0.15)

; Maintainability evaluation rules
(= (maintainability-bonus semantic-clarity) 0.1)
(= (maintainability-bonus proper-documentation) 0.1)
(= (maintainability-bonus meaningful-names) 0.05)

; =============================================================================
; Semantic Pattern Recognition
; =============================================================================

; Pattern family classification
(= (pattern-family search linear-search)
   (and (has-iteration forward)
        (has-condition comparison)
        (has-early-termination)))

(= (pattern-family search binary-search)
   (and (has-iteration logarithmic)
        (has-condition range-split)
        (requires-sorted-input)))

(= (pattern-family aggregate reduction)
   (and (has-iteration forward)
        (has-accumulator)
        (has-combine-operation)))

(= (pattern-family transform map-like)
   (and (has-iteration forward)
        (has-element-transformation)
        (preserves-collection-size)))

; =============================================================================
; Code Quality Assessment
; =============================================================================

; Semantic consistency rules
(= (semantic-consistency-check $genome)
   (and (all-preconditions-satisfied $genome)
        (no-conflicting-genes $genome)
        (proper-gene-sequence $genome)))

(= (all-preconditions-satisfied $genome)
   (forall $gene (member-of $gene $genome)
          (preconditions-met $gene $genome)))

(= (proper-gene-sequence $genome)
   (and (starts-with-initialization $genome)
        (ends-with-termination $genome)
        (logical-flow $genome)))

; =============================================================================
; Evolution Strategy Rules
; =============================================================================

; Population diversity maintenance
(= (maintain-diversity $population)
   (and (diverse-gene-combinations $population)
        (diverse-semantic-roles $population)
        (diverse-fitness-landscapes $population)))

; Elitism strategy
(= (elite-selection $population $elite-count)
   (top-fitness $population $elite-count))

; Tournament selection rules
(= (tournament-selection $population $tournament-size)
   (best-from-random-subset $population $tournament-size))

; =============================================================================
; Learning and Adaptation Rules
; =============================================================================

; Gene success tracking
(= (update-gene-success $gene $fitness)
   (if (> $fitness 0.7)
       (increment-success-count $gene)
       (increment-failure-count $gene)))

; Pattern learning rules
(= (learn-successful-pattern $pattern $context)
   (increase-pattern-weight $pattern $context))

(= (learn-failed-pattern $pattern $context)
   (decrease-pattern-weight $pattern $context))

; Adaptive mutation rates
(= (adaptive-mutation-rate $generation $diversity)
   (if (< $diversity 0.3)
       (increase-mutation-rate)
       (standard-mutation-rate)))

; =============================================================================
; Termination Conditions
; =============================================================================

; Evolution termination criteria
(= (evolution-complete $generation $best-fitness $improvement)
   (or (>= $generation max-generations)
       (>= $best-fitness target-fitness)
       (< $improvement convergence-threshold)))

; Stagnation detection
(= (evolution-stagnant $history $window)
   (< (fitness-variance $history $window) stagnation-threshold))

; =============================================================================
; Problem-Specific Rules
; =============================================================================

; Range-based function rules
(= (range-function-requirements $func)
   (and (validates-bounds $func)
        (handles-empty-range $func)
        (processes-subrange $func)))

; Collection processing rules
(= (collection-processing-requirements $func)
   (and (handles-empty-collection $func)
        (type-safe-operations $func)
        (proper-iteration $func)))

; =============================================================================
; Debugging and Tracing Rules
; =============================================================================

; Evolution trace rules
(= (trace-evolution-step $generation $population $operation)
   (log-evolution-event $generation $population $operation))

; Gene trace rules
(= (trace-gene-usage $gene $genome $result)
   (log-gene-performance $gene $genome $result))

; =============================================================================
; Utility Rules
; =============================================================================

; String matching utilities
(= (contains-name $func $substring)
   (contains (function-name $func) $substring))

; List utilities
(= (member-of $element $list)
   (or (= $element (head $list))
       (member-of $element (tail $list))))

; Fitness utilities
(= (fitness-variance $history $window)
   (variance (last-n $history $window)))

; =============================================================================
; Default Rules and Fallbacks
; =============================================================================

; Default gene selection when no specific rule applies
(= (select-gene $purpose $gene-type)
   (random-selection $gene-type))

; Default compatibility when no conflicts detected
(= (genes-compatible $gene1 $gene2) True)

; Default mutation safety
(= (mutation-safe $gene $context) True)

; Default fitness calculation
(= (calculate-fitness $genome)
   (weighted-sum (correctness $genome)
                 (efficiency $genome)
                 (maintainability $genome)))

;; Gene selection preference rules
(= (prefer-gene-for-purpose maximize maximize-condition) True)
(= (prefer-gene-for-purpose minimize minimize-condition) True)
(= (prefer-gene-for-purpose search forward-scan) True)
(= (prefer-gene-for-purpose aggregate update-accumulator) True)

;; Gene compatibility rules
(= (compatible-gene-sequence initialization iteration) True)
(= (compatible-gene-sequence iteration condition) True)
(= (compatible-gene-sequence condition operation) True)
(= (compatible-gene-sequence operation termination) True)

;; Gene success prediction rules
(= (predict-gene-success $gene-role $success-rate)
   (and (gene-success-rate $gene-role $rate)
        (> $rate 0.5)))

;; LLM fallback necessity rules
(= (need-llm-fallback $gene-type $role-hint)
   (and (not (suitable-gene-available $gene-type $role-hint))
        (critical-gene-type $gene-type)))

(= (critical-gene-type initialization) True)
(= (critical-gene-type condition) True)
(= (critical-gene-type termination) True)

;; Learning rules for LLM-generated genes
(= (incorporate-llm-gene $gene-role $template $success-rate)
   (and (llm-generated-gene $gene-role)
        (> $success-rate 0.7)))
