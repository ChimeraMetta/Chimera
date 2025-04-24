;; Core Proof Verification Rules for MeTTa
;; These rules are designed to work with the JSON IR -> MeTTa conversion

;; =====================
;; Type Definitions
;; =====================

;; Define property types
(: bound-check Type)
(: ordering-check Type)
(: null-check Type)
(: termination-guarantee Type)
(: error-handling Type)

;; Define specific pattern types for bounds checking
(: index-less-than-length Pattern)       ;; e.g., "index < length"
(: index-less-equal-length Pattern)      ;; e.g., "index <= length"
(: index-greater-equal-zero Pattern)     ;; e.g., "index >= 0"
(: index-within-bounds Pattern)          ;; e.g., "0 <= index < length"

;; Define specific pattern types for ordering
(: array-is-sorted Pattern)              ;; e.g., "array is sorted"
(: ascending-order Pattern)              ;; e.g., "elements in ascending order"
(: descending-order Pattern)             ;; e.g., "elements in descending order"
(: preserves-element-order Pattern)      ;; e.g., "preserves the order of elements"

;; Define specific pattern types for null checking
(: value-not-null Pattern)               ;; e.g., "value != null"
(: checks-for-null Pattern)              ;; e.g., "if value is null"
(: handles-empty-collection Pattern)     ;; e.g., "if collection is empty"

;; Define specific pattern types for termination
(: decreasing-loop-variable Pattern)     ;; e.g., "i decreases in each iteration"
(: increasing-towards-bound Pattern)     ;; e.g., "i increases towards bound"
(: loop-invariant-progress Pattern)      ;; e.g., "invariant ensures progress"
(: finite-iteration-count Pattern)       ;; e.g., "loop runs a fixed number of times"

;; Define specific pattern types for error handling
(: checks-for-not-found Pattern)         ;; e.g., "returns -1 if not found"
(: validates-input Pattern)              ;; e.g., "validates input before processing"
(: handles-edge-cases Pattern)           ;; e.g., "handles edge cases"
(: error-code-return Pattern)            ;; e.g., "returns error code"

;; =====================
;; Pattern to Property Mapping
;; =====================

;; Map bounds checking patterns to bound-check property
(= (pattern-property index-less-than-length bound-check))
(= (pattern-property index-less-equal-length bound-check))
(= (pattern-property index-greater-equal-zero bound-check))
(= (pattern-property index-within-bounds bound-check))

;; Map ordering patterns to ordering-check property
(= (pattern-property array-is-sorted ordering-check))
(= (pattern-property ascending-order ordering-check))
(= (pattern-property descending-order ordering-check))
(= (pattern-property preserves-element-order ordering-check))

;; Map null checking patterns to null-check property
(= (pattern-property value-not-null null-check))
(= (pattern-property checks-for-null null-check))
(= (pattern-property handles-empty-collection null-check))

;; Map termination patterns to termination-guarantee property
(= (pattern-property decreasing-loop-variable termination-guarantee))
(= (pattern-property increasing-towards-bound termination-guarantee))
(= (pattern-property loop-invariant-progress termination-guarantee))
(= (pattern-property finite-iteration-count termination-guarantee))

;; Map error handling patterns to error-handling property
(= (pattern-property checks-for-not-found error-handling))
(= (pattern-property validates-input error-handling))
(= (pattern-property handles-edge-cases error-handling))
(= (pattern-property error-code-return error-handling))

;; =====================
;; Property Verification
;; =====================

;; Relationship between expressions and properties
(: Expression-Property (-> Expression Type Bool))

;; Verify that an expression has a specific property
(= (verify-property $expr $property-type)
   (match &self (Expression-Property $expr $property-type)))

;; Verify that a function has the required property
(= (function-has-required-property $func $property-atom)
   (match &self (function-has-property $func $property-atom)))

;; Verify function has a specific property through its expressions
(= (function-has-property-via-expr $func $property-type)
   (match &self (function-invariant $func $expr)
          (Expression-Property $expr $property-type)))

;; Define property atom equivalence classes
(= (property-equivalent (bound-checking) bound-check) True)
(= (property-equivalent (ordering-preservation) ordering-check) True)
(= (property-equivalent (null-check) null-check) True)
(= (property-equivalent (termination) termination-guarantee) True)
(= (property-equivalent (error-handling) error-handling) True)

;; Verify a function satisfies a property
(= (function-satisfies $func $property-atom)
   (or (function-has-required-property $func $property-atom)
       (match &self (property-equivalent $property-atom $property-type)
              (function-has-property-via-expr $func $property-type))))

;; =====================
;; Expression Equivalence
;; =====================

;; Expression equivalence and implication (for proofs)
(: Equivalent-Expressions (-> Expression Expression Bool))
(: Implies-Expression (-> Expression Expression Bool))

;; =====================
;; Loop Invariant Verification
;; =====================

;; Check if a loop invariant is properly maintained
(= (loop-invariant-valid $func $location $inv)
   (and (invariant-initialization-valid $func $inv)
        (invariant-preservation-valid $func $inv)
        (invariant-ensures-postcondition $func $inv)))

;; Check if the invariant is established properly
(= (invariant-initialization-valid $func $inv)
   (match &self (function-precondition $func $pre)
          (Implies-Expression $pre $inv)))

;; Check if the invariant is preserved by loop body
(= (invariant-preservation-valid $func $inv)
   (match &self (Assertion $location $assert)
          (function-def $func $scope $start $end)
          (and (>= $location $start) 
               (<= $location $end)
               (Implies-Expression $inv $assert))))

;; Check if the invariant ensures postcondition
(= (invariant-ensures-postcondition $func $inv)
   (match &self (function-postcondition $func $post)
          (Implies-Expression $inv $post)))

;; =====================
;; Proof Verification
;; =====================

;; Verify a complete proof
(= (verify-proof $func)
   (and (preconditions-handled $func)
        (invariants-maintained $func)
        (postconditions-ensured $func)))

;; Verify preconditions are properly handled
(= (preconditions-handled $func)
   (match &self (function-precondition $func $pre)
          (Precondition $expr)
          (Equivalent-Expressions $pre $expr)))

;; Verify invariants are maintained
(= (invariants-maintained $func)
   (match &self (LoopInvariant $location $inv)
          (function-def $func $scope $start $end)
          (and (>= $location $start) 
               (<= $location $end)
               (loop-invariant-valid $func $location $inv))))

;; Verify postconditions are ensured
(= (postconditions-ensured $func)
   (match &self (function-postcondition $func $post)
          (or (match &self (LoopInvariant $location $inv)
                     (Implies-Expression $inv $post))
              (match &self (Assertion $location $assert)
                     (Equivalent-Expressions $assert $post)))))

;; =====================
;; Donor Compatibility
;; =====================

;; Verify if a function is a suitable donor based on properties
(= (suitable-donor-candidate $func $required-properties)
   (forall $prop $required-properties
           (function-satisfies $func $prop)))

;; =====================
;; Pattern Classification Rules
;; =====================

;; Relate patterns to property types
;; Instead of using string literals, use atomic pattern definitions

;; Bounds checking patterns
(= (pattern-property bounds-pattern-1 bound-check))
(= (pattern-property bounds-pattern-2 bound-check))
(= (pattern-property bounds-pattern-3 bound-check))
(= (pattern-property bounds-pattern-4 bound-check))

;; Ordering patterns
(= (pattern-property ordering-pattern-1 ordering-check))
(= (pattern-property ordering-pattern-2 ordering-check))
(= (pattern-property ordering-pattern-3 ordering-check))
(= (pattern-property ordering-pattern-4 ordering-check))

;; Null checking patterns
(= (pattern-property null-pattern-1 null-check))
(= (pattern-property null-pattern-2 null-check))
(= (pattern-property null-pattern-3 null-check))

;; Termination patterns
(= (pattern-property termination-pattern-1 termination-guarantee))
(= (pattern-property termination-pattern-2 termination-guarantee))
(= (pattern-property termination-pattern-3 termination-guarantee))
(= (pattern-property termination-pattern-4 termination-guarantee))

;; Error handling patterns
(= (pattern-property error-pattern-1 error-handling))
(= (pattern-property error-pattern-2 error-handling))
(= (pattern-property error-pattern-3 error-handling))
(= (pattern-property error-pattern-4 error-handling))

;; =====================
;; Execution Evidence
;; =====================

;; Define execution record and state types
(: execution-record (-> Number Atom Atom Bool))
(: execution-state (-> Number Atom Atom Bool))

;; Check if execution evidence supports a property
(= (execution-evidence-supports $func $property)
   (match &self (execution-record $id $input $output)
          (execution-property-satisfied $id $property)))

;; Check if a specific execution supports a property
(= (execution-property-satisfied $id $property)
   (match &self (execution-state $id $var $val)
          (state-has-property $var $val $property)))

;; Map state properties based on variable values and types
(= (state-has-property index-var $val bound-check)
   (>= $val 0))

(= (state-has-property result-var $val error-handling)
   (or (= $val negative-one-value) (= $val not-found-value)))

(= (state-has-property array-var $val ordering-check)
   (is-ordered $val))

;; Detect if a collection is ordered
(= (is-ordered $val)
   (match &self (ordered-collection $val)))

;; =====================
;; Property Requirements for Donor Selection
;; =====================

;; Map requirement atoms to property types

;; Requirements for bounds checking
(= (donor-requirement-maps bounds-requirement bound-check))
(= (donor-requirement-maps index-requirement bound-check))

;; Requirements for ordering
(= (donor-requirement-maps order-requirement ordering-check))
(= (donor-requirement-maps sorted-requirement ordering-check))

;; Requirements for null checking
(= (donor-requirement-maps null-requirement null-check))
(= (donor-requirement-maps empty-requirement null-check))

;; Requirements for termination
(= (donor-requirement-maps termination-requirement termination-guarantee))
(= (donor-requirement-maps loop-requirement termination-guarantee))

;; Requirements for error handling
(= (donor-requirement-maps error-requirement error-handling))
(= (donor-requirement-maps not-found-requirement error-handling))

;; Define property requirement types
(: bounds-requirement Property-Requirement)
(: index-requirement Property-Requirement)
(: order-requirement Property-Requirement)
(: sorted-requirement Property-Requirement)
(: null-requirement Property-Requirement)
(: empty-requirement Property-Requirement)
(: termination-requirement Property-Requirement)
(: loop-requirement Property-Requirement)
(: error-requirement Property-Requirement)
(: not-found-requirement Property-Requirement)