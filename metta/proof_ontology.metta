;; Proof Ontology Rules for MeTTa
;; These rules enable reasoning about proofs and verifying donor candidate suitability
;; Updated to avoid string matching operations

;; ======================
;; Core Proof Concepts
;; ======================

;; A valid proof must satisfy its specifications
(= (valid-proof $func)
   (and (all-preconditions-handled $func)
        (all-postconditions-proved $func)
        (all-invariants-maintained $func)))

;; Check if all preconditions are handled
(= (all-preconditions-handled $func)
   (case (match &self (specification-precondition $pre)
                (has-precondition-handler $func $pre))
     (Empty True)  ;; No preconditions to handle
     (False False) ;; Some precondition not handled
     ($_ True)))   ;; All preconditions handled

;; Check if a function handles a precondition
(= (has-precondition-handler $func $pre)
   (match &self (Precondition $expr)
          (equivalent-expressions $expr $pre)))

;; Check if all postconditions are proved
(= (all-postconditions-proved $func)
   (case (match &self (specification-postcondition $post)
                (proves-postcondition $func $post))
     (Empty True)  ;; No postconditions to prove
     (False False) ;; Some postcondition not proved
     ($_ True)))   ;; All postconditions proved

;; Check if a function proves a postcondition
(= (proves-postcondition $func $post)
   (or (match &self (function-invariant $func $inv)
              (implies-expression $inv $post))
       (match &self (Assertion $loc $expr)
              (equivalent-expressions $expr $post))))

;; Check if all invariants are maintained
(= (all-invariants-maintained $func)
   (case (match &self (LoopInvariant $loc $inv)
                (invariant-maintained $func $loc $inv))
     (Empty True)  ;; No invariants to maintain
     (False False) ;; Some invariant not maintained
     ($_ True)))   ;; All invariants maintained

;; Check if an invariant is maintained throughout execution
(= (invariant-maintained $func $loc $inv)
   (and (invariant-initially-true $func $inv)
        (invariant-preserved $func $inv)))

;; Check if invariant is initially true (from preconditions)
(= (invariant-initially-true $func $inv)
   (match &self (specification-precondition $pre)
          (implies-expression $pre $inv)))

;; Check if invariant is preserved by loop body
(= (invariant-preserved $func $inv)
   (match &self (Assertion $loc $expr)
          (implies-expression $inv $expr)))

;; ======================
;; Donor Compatibility
;; ======================

;; Check if a function satisfies required properties for transplantation
(= (suitable-donor $func $required-properties)
   (forall $prop $required-properties
           (satisfies-property $func $prop)))

;; Check if a function satisfies a specific property
(= (satisfies-property $func $prop)
   (or (match &self (function-invariant $func $inv)
              (has-property $inv $prop))
       (match &self (Assertion $loc $expr)
              (has-property $expr $prop))
       (match &self (Precondition $expr)
              (has-property $expr $prop))))

;; Check if an expression has a property
;; Instead of string matching, use property atoms
(= (has-property $expr (bound-checking))
   (has-bound-check $expr))

(= (has-property $expr (ordering-preservation))
   (has-ordering-check $expr))

(= (has-property $expr (null-check))
   (has-null-check $expr))

(= (has-property $expr (termination))
   (has-termination-guarantee $expr))

(= (has-property $expr (error-handling))
   (has-error-check $expr))

;; Detect specific property types rather than searching for strings
(= (has-bound-check $expr)
   (match &self (Expression-Property $expr bound-check)))

(= (has-ordering-check $expr)
   (match &self (Expression-Property $expr ordering-check)))

(= (has-null-check $expr)
   (match &self (Expression-Property $expr null-check)))

(= (has-termination-guarantee $expr)
   (match &self (Expression-Property $expr termination-guarantee)))

(= (has-error-check $expr)
   (match &self (Expression-Property $expr error-handling)))

;; Equivalence and implication for expressions
(= (equivalent-expressions $expr1 $expr2)
   (match &self (Equivalent-Expressions $expr1 $expr2)))

(= (implies-expression $expr1 $expr2)
   (or (equivalent-expressions $expr1 $expr2)
       (match &self (Implies-Expression $expr1 $expr2))))

;; ======================
;; Property Categories
;; ======================

;; Define common property categories for donor selection

;; Functions that maintain numeric bounds
(= (maintains-bounds $func)
   (match &self (function-invariant $func $inv)
          (has-bound-check $inv)))

;; Functions that preserve list/array ordering
(= (preserves-ordering $func)
   (match &self (function-invariant $func $inv)
          (has-ordering-check $inv)))

;; Functions that maintain non-null invariants
(= (maintains-non-null $func)
   (match &self (function-invariant $func $inv)
          (has-null-check $inv)))

;; Functions with pure operations (no side effects)
(= (pure-function $func)
   (and (not (has-side-effect $func))
        (deterministic $func)))

;; Check if function has side effects
(= (has-side-effect $func)
   (match &self (function-modifies-external $func $var)))

;; Check if function is deterministic
(= (deterministic $func)
   (not (match &self (function-uses-random $func))))

;; ======================
;; Execution-Based Properties
;; ======================

;; Properties derived from execution traces

;; Check if a function behaves consistently across executions
(= (consistent-behavior $func)
   (forall $i $j (and (execution-record $i $input1 $output1)
                      (execution-record $j $input2 $output2)
                      (similar-inputs $input1 $input2))
          (similar-outputs $output1 $output2)))

;; Check if inputs are similar (simplified)
(= (similar-inputs $input1 $input2)
   (match &self (Similar-Inputs $input1 $input2)))

;; Check if outputs are similar (simplified)
(= (similar-outputs $output1 $output2)
   (match &self (Similar-Outputs $output1 $output2)))

;; Infer property from execution evidence
(= (execution-implies-property $func $prop)
   (forall $i (execution-record $i $input $output)
          (execution-state-satisfies $i $prop)))

;; Check if execution state satisfies property
(= (execution-state-satisfies $i $prop)
   (match &self (execution-state $i $var $val)
          (state-satisfies-property $var $val $prop)))

;; Check if variable state implies property (using property atoms instead of strings)
(= (state-satisfies-property $var $val (non-negative))
   (>= $val 0))

(= (state-satisfies-property $var $val (ordered))
   (ordered-value $val))

;; Check if a value represents an ordered collection
(= (ordered-value $val)
   (and (is-collection $val)
        (is-ordered $val)))

;; ======================
;; Adaptation Verification
;; ======================

;; Verify that adaptation preserves essential properties
(= (valid-adaptation $original $adapted $essential-properties)
   (forall $prop $essential-properties
           (property-preserved $original $adapted $prop)))

;; Check if property is preserved in adaptation
(= (property-preserved $original $adapted $prop)
   (and (satisfies-property $original $prop)
        (satisfies-property $adapted $prop)))

;; Identify which properties are preserved
(= (preserved-properties $original $adapted $all-properties)
   (filter (lambda ($p) (property-preserved $original $adapted $p))
           $all-properties))

;; Identify which properties are violated
(= (violated-properties $original $adapted $all-properties)
   (filter (lambda ($p) (not (property-preserved $original $adapted $p)))
           $all-properties))

;; ======================
;; Proof Complexity
;; ======================

;; Analyze proof complexity

;; Get total proof component count
(= (proof-component-count $func)
   (+ (count-invariants $func)
      (count-assertions $func)
      (count-preconditions $func)))

;; Count loop invariants for a function
(= (count-invariants $func)
   (case (match &self (function-invariant $func $inv))
     (Empty 0)
     ($inv (+ 1 (count-invariants $func)))))

;; Count assertions for a function
(= (count-assertions $func)
   (case (match &self (Assertion $loc $expr))
     (Empty 0)
     ($expr (+ 1 (count-assertions $func)))))

;; Count preconditions for a function
(= (count-preconditions $func)
   (case (match &self (Precondition $expr))
     (Empty 0)
     ($expr (+ 1 (count-preconditions $func)))))

;; Determine if a proof is complex
(= (complex-proof $func)
   (> (proof-component-count $func) 5))

;; ======================
;; Type-Specific Properties
;; ======================

;; Property atoms for different types of invariants
;; These replace string matching with atomic patterns

;; Define property types as atoms
(: bound-check Type)
(: ordering-check Type)
(: null-check Type)
(: termination-guarantee Type)
(: error-handling Type)
(: non-negative Type)
(: ordered Type)

;; Pattern atoms for different expressions
(: Expression-Property (-> Atom Type Bool))
(: Equivalent-Expressions (-> Atom Atom Bool))
(: Implies-Expression (-> Atom Atom Bool))
(: Similar-Inputs (-> Atom Atom Bool))
(: Similar-Outputs (-> Atom Atom Bool))

;; Collection and ordering type checks
(: is-collection (-> Atom Bool))
(: is-ordered (-> Atom Bool))