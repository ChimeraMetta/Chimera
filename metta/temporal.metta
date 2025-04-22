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
;; Atom operations
;; ======================

; Helper for extracting the first result
(= (first $results $first)
   (case $results
     (Empty $Empty)
     ((cons $head $tail) (= $first $head))))

; Helper for counting results - returns actual number
(= (count-results $list)
   (case $list
     (Empty 0)
     ((cons $head $tail) (+ 1 (count-results $tail)))))

; Helper to check if list is empty
(= (is-empty $list)
   (case $list
     (Empty True)
     ($_ False)))

; Timestamp comparison - returns True if ts1 is earlier than ts2
(= (timestamp-earlier $ts1 $ts2)
   (< $ts1 $ts2))

; Extract timestamp from commit-info
(= (get-commit-timestamp $commit-id $timestamp)
   (match &self (commit-info $commit-id $author $ts $msg) $result)
   (if (is-empty $result)
       (= $timestamp 0)  ; Default timestamp if not found
       (= $timestamp (first $result 0))))

; Sort commits by timestamp (earliest first)
(= (sort-by-timestamp $commits $sorted)
   (timestamp-sort $commits Empty $sorted))

; Helper for timestamp sorting (insertion sort)
(= (timestamp-sort $commits $sorted-so-far $sorted)
   (case $commits
     (Empty (= $sorted $sorted-so-far))
     ((cons $head $tail) 
      (let $inserted (insert-by-timestamp $head $sorted-so-far Empty)
           (timestamp-sort $tail $inserted $sorted)))))

; Insert a commit into the right position based on timestamp
(= (insert-by-timestamp $commit $sorted $result)
   (case $sorted
     (Empty (= $result (cons $commit Empty)))
     ((cons $head $tail)
      (let $ts1 (get-commit-timestamp $commit 0)
           $ts2 (get-commit-timestamp $head 0)
           (if (timestamp-earlier $ts1 $ts2)
               (= $result (cons $commit (cons $head $tail)))
               (let $inserted (insert-by-timestamp $commit $tail Empty)
                    (= $result (cons $head $inserted))))))))

;; ======================
;; Function lifetime
;; ======================

; Get all commits for a function
(= (function-commits $func $commits)
   (match &self (function-signature-at $func $commit $_) $commits))

; Get first commit (earliest) for a function
(= (function-first-commit $func $first-commit)
   (function-commits $func $all-commits)
   (if (is-empty $all-commits)
       (= $first-commit $Empty)
       (sort-by-timestamp $all-commits $sorted)
       (first $sorted $first-commit)))

; Get last commit (latest) for a function
(= (function-last-commit $func $last-commit)
   (function-commits $func $all-commits)
   (if (is-empty $all-commits)
       (= $last-commit $Empty)
       (sort-by-timestamp $all-commits $sorted)
       (last $sorted $last-commit)))

; Helper to get the last element of a list
(= (last $list $last-element)
   (case $list
     (Empty (= $last-element $Empty))
     ((cons $head Empty) (= $last-element $head))
     ((cons $head $tail) (last $tail $last-element))))

;; ======================
;; Change detection
;; ======================

; Get body hash for a function at a commit
(= (function-body-hash $func $commit $hash)
   (match &self (function-body-at $func $commit $h) $result)
   (if (is-empty $result)
       (= $hash $Empty)
       (= $hash (first $result $Empty))))

; Check if function changed between commits
(= (function-changed-between $func $commit1 $commit2)
   (function-body-hash $func $commit1 $hash1)
   (function-body-hash $func $commit2 $hash2)
   (and (not (= $hash1 $Empty))
        (not (= $hash2 $Empty))
        (not (= $hash1 $hash2))))

;; ======================
;; Change frequency tracking
;; ======================

; Count number of commits where function exists
(= (function-change-frequency $func $freq)
   (function-commits $func $all-commits)
   (= $freq (count-results $all-commits)))

;; ======================
;; Growth tracking
;; ======================

; Get complexity for a function at a commit
(= (get-complexity $func $commit $complexity)
   (match &self (function-complexity-at $func $commit $c) $result)
   (if (is-empty $result)
       (= $complexity 0)
       (= $complexity (first $result 0))))

; Calculate complexity growth between commits
(= (function-complexity-growth $func $start-commit $end-commit $change)
   (get-complexity $func $start-commit $start-complexity)
   (get-complexity $func $end-commit $end-complexity)
   (= $change (- $end-complexity $start-complexity)))

;; ======================
;; Co-evolution pattern
;; ======================

; Count commits where both functions changed
(= (co-change-count $func1 $func2 $count)
   (match &self 
          (function-body-at $func1 $commit $_)
          (function-body-at $func2 $commit $_)
          $common-commits)
   (= $count (count-results $common-commits)))

; Determine if functions co-evolve
(= (functions-co-evolve $func1 $func2 $threshold)
   (co-change-count $func1 $func2 $count)
   (> $count $threshold))

;; ======================
;; Dependency evolution
;; ======================

; Check if dependency exists
(= (has-dependency $source $target $commit)
   (match &self (dependency-at $source $target $commit $_) $result)
   (not (is-empty $result)))

; Find previous commit in chronological order
(= (find-previous-commit $commit $prev)
   (match &self (commit-info $c $_ $_ $_) $all-commits)
   (sort-by-timestamp $all-commits $sorted)
   (find-prev-in-sorted $commit $sorted $prev))

; Helper to find previous commit in sorted list
(= (find-prev-in-sorted $commit $sorted $prev)
   (case $sorted
     (Empty (= $prev $Empty))
     ((cons $c Empty) (= $prev $Empty))
     ((cons $c (cons $next $rest))
      (if (= $next $commit)
          (= $prev $c)
          (find-prev-in-sorted $commit (cons $next $rest) $prev)))))

; Check if dependency was added at commit
(= (dependency-added-at $source $target $commit)
   (has-dependency $source $target $commit)
   (find-previous-commit $commit $prev)
   (or (= $prev $Empty)
       (not (has-dependency $source $target $prev))))

;; ======================
;; Hotspot detection
;; ======================

; Detect function hotspots based on change frequency
(= (function-hotspot $func $confidence)
   (function-change-frequency $func $frequency)
   (if (> $frequency 5)
       (= $confidence High)
       (if (> $frequency 2)
           (= $confidence Medium)
           (= $confidence Low))))

; Detect hotspots based on complexity
(= (complexity-hotspot $func $confidence)
   (match &self (function-complexity-at $func $_ $complexity) $complexity-results)
   (if (is-empty $complexity-results)
       (= $confidence Low)
       (let $last-complexity (first $complexity-results 0)
            (if (> $last-complexity 10)
                (= $confidence High)
                (if (> $last-complexity 5)
                    (= $confidence Medium)
                    (= $confidence Low))))))