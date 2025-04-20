;; ======================
;; Core temporal atoms
;; ======================

(: commit-info (-> String String String String))  ; id, author, timestamp, message
(: function-state-at (-> String String Number String))  ; function-name, commit-id, version, state-hash
(: function-signature-at (-> String String String))  ; function-name, commit-id, signature
(: function-body-at (-> String String String))  ; function-name, commit-id, body-hash
(: function-complexity-at (-> String String Number))  ; function-name, commit-id, complexity
(: dependency-at (-> String String String Number))  ; source-func, target-func, commit-id, count

;; ======================
;; Function lifetime
;; ======================

(= (function-first-commit $func $first-commit)
   (sort-by-timestamp (match &self (function-signature-at $func $commit $_) $commit) "ascending" $results)
   (first $results $first-commit))

(= (function-last-commit $func $last-commit)
   (sort-by-timestamp (match &self (function-signature-at $func $commit $_) $commit) "descending" $results)
   (first $results $last-commit))

;; ======================
;; Change detection
;; ======================
(= (function-changed-between $func $commit1 $commit2)
   (function-body-at $func $commit1 $hash1)
   (function-body-at $func $commit2 $hash2)
   (not (= $hash1 $hash2)))

;; ======================
;; Growth tracking
;; ======================

(= (function-complexity-growth $func $start-commit $end-commit $change)
   (function-complexity-at $func $start-commit $start-complexity)
   (function-complexity-at $func $end-commit $end-complexity)
   (= $change (- $end-complexity $start-complexity)))

;; ======================
;; Co-evolution pattern
;; ======================

(= (functions-co-evolve $func1 $func2 $threshold)
   (> (co-change-frequency $func1 $func2) $threshold))

(= (co-change-frequency $func1 $func2)
   (count-results (match &self 
                         (function-changed-between $func1 $commit $_)
                         (function-changed-between $func2 $commit $_))))

;; ======================
;; Dependency evolution
;; ======================

(= (dependency-added-at $source $target $commit)
   (dependency-at $source $target $commit $_)
   (prev-commit $commit $prev-commit)
   (not (dependency-at $source $target $prev-commit $_)))

;; ======================
;; Hotspot detection
;; ======================

(= (function-hotspot $func $confidence)
   (function-change-frequency $func $frequency)
   (function-complexity-growth $func $_ $_ $growth)
   (> $frequency 10)
   (> $growth 5)
   (= $confidence "high"))