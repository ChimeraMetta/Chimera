;; ============================================
;; SQL Query Pattern Ontology for PostgreSQL Query Optimizer
;; ============================================
;; This ontology defines rules for detecting query anti-patterns
;; and generating optimization suggestions using MeTTa reasoning.

;; ============================================
;; Type Definitions
;; ============================================

(: PatternSeverity Type)
(: high PatternSeverity)
(: medium PatternSeverity)
(: low PatternSeverity)

(: PatternType Type)
(: full-table-scan PatternType)
(: missing-index PatternType)
(: select-star PatternType)
(: implicit-join PatternType)
(: cartesian-product-risk PatternType)
(: correlated-subquery PatternType)
(: n-plus-one PatternType)
(: filesort-required PatternType)
(: aggregation-full-scan PatternType)
(: missing-covering-index PatternType)

(: OptimizationType Type)
(: create-index OptimizationType)
(: rewrite-join OptimizationType)
(: subquery-to-join OptimizationType)
(: create-covering-index OptimizationType)
(: add-predicate-pushdown OptimizationType)

;; ============================================
;; Pattern Detection Rules
;; ============================================

;; Full Table Scan Detection
;; Detected when: large table without index usage in WHERE clause
(= (detect-pattern full-table-scan $query-id $atoms)
   (match $atoms
     (, (query-table $table)
        (where-condition $column $op)
        (table-row-count $table $rows)
        (table-missing-index $table $idx-col))
     (if (> $rows 10000)
         (pattern-detected $query-id full-table-scan high
           (concat "Table " $table " has " $rows " rows but missing index on WHERE column"))
         Empty)))

;; SELECT * Anti-pattern Detection
(= (detect-pattern select-star $query-id $atoms)
   (match $atoms
     (uses-select-star)
     (pattern-detected $query-id select-star medium
       "SELECT * fetches unnecessary columns, specify needed columns explicitly")))

;; Implicit JOIN Detection (comma syntax in FROM)
(= (detect-pattern implicit-join $query-id $atoms)
   (match $atoms
     (uses-implicit-join)
     (pattern-detected $query-id implicit-join medium
       "Implicit join using comma syntax can lead to cartesian products")))

;; Cartesian Product Risk
(= (detect-pattern cartesian-product $query-id $atoms)
   (match $atoms
     (, (uses-implicit-join)
        (query-join implicit $left $right))
     (pattern-detected $query-id cartesian-product-risk high
       (concat "Potential cartesian product between " $left " and " $right))))

;; Correlated Subquery Detection
(= (detect-pattern correlated-subquery $query-id $atoms)
   (match $atoms
     (has-correlated-subquery)
     (pattern-detected $query-id correlated-subquery high
       "Correlated subquery executes once per row - consider rewriting as JOIN")))

;; N+1 Query Pattern (subquery referencing outer table)
(= (detect-pattern n-plus-one $query-id $atoms)
   (match $atoms
     (, (has-correlated-subquery)
        (table-row-count $table $rows))
     (if (> $rows 1000)
         (pattern-detected $query-id n-plus-one high
           (concat "Subquery may execute " $rows " times - severe performance impact"))
         Empty)))

;; Aggregation without covering index
(= (detect-pattern aggregation-scan $query-id $atoms)
   (match $atoms
     (, (uses-aggregation $agg)
        (group-by-column $col)
        (table-missing-covering-index $table))
     (pattern-detected $query-id aggregation-full-scan high
       (concat "Aggregation on " $col " requires full table scan"))))

;; ORDER BY without index (filesort)
(= (detect-pattern filesort $query-id $atoms)
   (match $atoms
     (, (order-by-column $col)
        (uses-aggregation $agg))
     (pattern-detected $query-id filesort-required medium
       (concat "ORDER BY on aggregated column requires filesort"))))

;; Missing index on JOIN condition
(= (detect-pattern missing-join-index $query-id $atoms)
   (match $atoms
     (, (query-join $type $left $right)
        (table-row-count $right $rows))
     (if (> $rows 10000)
         (pattern-detected $query-id missing-join-index medium
           (concat "Consider index on join column for " $right))
         Empty)))

;; ============================================
;; Optimization Suggestion Rules
;; ============================================

;; Suggest index creation for full table scan
(= (suggest-optimization $query-id full-table-scan $table $column)
   (optimization-suggestion $query-id create-index high
     (concat "CREATE INDEX idx_" $table "_" $column " ON " $table "(" $column ");")))

;; Suggest JOIN rewrite for implicit join
(= (suggest-optimization $query-id implicit-join $left $right)
   (optimization-suggestion $query-id rewrite-join medium
     (concat "Rewrite as: SELECT ... FROM " $left " INNER JOIN " $right " ON ...")))

;; Suggest subquery to JOIN transformation
(= (suggest-optimization $query-id correlated-subquery)
   (optimization-suggestion $query-id subquery-to-join high
     "Rewrite correlated subquery as JOIN with GROUP BY for better performance"))

;; Suggest covering index for aggregation
(= (suggest-optimization $query-id aggregation-full-scan $table $group-col $agg-col)
   (optimization-suggestion $query-id create-covering-index high
     (concat "CREATE INDEX idx_" $table "_covering ON " $table "(" $group-col ", " $agg-col ");")))

;; ============================================
;; Optimization Quality Assessment
;; ============================================

;; High quality optimization (major performance improvement expected)
(= (optimization-quality $suggestion high)
   (match $suggestion
     (optimization-suggestion $id $type high $desc)
     True))

;; Medium quality optimization
(= (optimization-quality $suggestion medium)
   (match $suggestion
     (optimization-suggestion $id $type medium $desc)
     True))

;; ============================================
;; Severity Classification
;; ============================================

;; High severity patterns (>90% potential improvement)
(= (pattern-severity full-table-scan) high)
(= (pattern-severity correlated-subquery) high)
(= (pattern-severity n-plus-one) high)
(= (pattern-severity aggregation-full-scan) high)
(= (pattern-severity cartesian-product-risk) high)

;; Medium severity patterns (50-90% potential improvement)
(= (pattern-severity missing-index) medium)
(= (pattern-severity filesort-required) medium)
(= (pattern-severity implicit-join) medium)
(= (pattern-severity missing-join-index) medium)

;; Low severity patterns (<50% potential improvement)
(= (pattern-severity select-star) low)

;; ============================================
;; Pattern Family Classification
;; ============================================

(= (pattern-family full-table-scan) scan-patterns)
(= (pattern-family missing-index) scan-patterns)
(= (pattern-family aggregation-full-scan) scan-patterns)

(= (pattern-family implicit-join) join-patterns)
(= (pattern-family cartesian-product-risk) join-patterns)
(= (pattern-family missing-join-index) join-patterns)

(= (pattern-family correlated-subquery) subquery-patterns)
(= (pattern-family n-plus-one) subquery-patterns)

(= (pattern-family filesort-required) sort-patterns)
(= (pattern-family select-star) column-patterns)

;; ============================================
;; Expected Improvement Estimates
;; ============================================

;; Improvement percentages for each pattern type
(= (expected-improvement full-table-scan) 95)
(= (expected-improvement correlated-subquery) 98)
(= (expected-improvement n-plus-one) 98)
(= (expected-improvement aggregation-full-scan) 97)
(= (expected-improvement cartesian-product-risk) 99)
(= (expected-improvement implicit-join) 85)
(= (expected-improvement filesort-required) 70)
(= (expected-improvement missing-index) 90)
(= (expected-improvement select-star) 30)

;; ============================================
;; SQL Rewrite Rules - Dynamic Query Optimization
;; ============================================
;; These rules define transformations from anti-patterns to optimized SQL

;; --- Rewrite Rule: SELECT * to Explicit Columns ---
;; Input: (rewrite-select-star table-name columns)
;; Output: Optimized SELECT clause
(= (rewrite-select-star $table $columns)
   (rewrite-result select-star
     (build-select-clause $columns)
     "Replaced SELECT * with explicit columns to reduce I/O"))

;; Build SELECT clause from column list
(= (build-select-clause ()) "")
(= (build-select-clause ($col)) $col)
(= (build-select-clause ($col . $rest))
   (concat $col ", " (build-select-clause $rest)))

;; --- Rewrite Rule: Implicit JOIN to Explicit JOIN ---
;; Transform: FROM a, b WHERE a.id = b.a_id
;; To: FROM a INNER JOIN b ON a.id = b.a_id
(= (rewrite-implicit-join $left-table $left-alias $right-table $right-alias $join-condition)
   (rewrite-result implicit-join
     (build-explicit-join $left-table $left-alias $right-table $right-alias $join-condition)
     "Converted implicit join to explicit INNER JOIN for query optimizer"))

(= (build-explicit-join $left $left-alias $right $right-alias $condition)
   (concat "FROM " $left " " $left-alias
           " INNER JOIN " $right " " $right-alias
           " ON " $condition))

;; --- Rewrite Rule: Correlated Subquery to JOIN ---
;; Transform: WHERE (SELECT ... FROM inner WHERE inner.id = outer.id)
;; To: JOIN (SELECT ... GROUP BY ...) subq ON ...
(= (rewrite-correlated-subquery $outer-table $inner-table $correlation-col $agg-func $agg-col $condition)
   (rewrite-result correlated-subquery
     (build-subquery-join $outer-table $inner-table $correlation-col $agg-func $agg-col $condition)
     "Transformed correlated subquery to derived table JOIN"))

(= (build-subquery-join $outer $inner $corr-col $agg $agg-col $cond)
   (concat "FROM " $outer " "
           "INNER JOIN (SELECT " $corr-col ", " $agg "(" $agg-col ") as agg_result "
           "FROM " $inner " GROUP BY " $corr-col " HAVING " $agg "(" $agg-col ") " $cond
           ") subq ON " $outer "." $corr-col " = subq." $corr-col))

;; --- Rewrite Rule: Add Index Suggestion ---
(= (suggest-index $table $column $is-covering $extra-cols)
   (if $is-covering
       (index-suggestion
         (concat "CREATE INDEX idx_" $table "_" $column "_covering ON " $table "(" $column ", " $extra-cols ");")
         high
         "Covering index includes all columns needed by query")
       (index-suggestion
         (concat "CREATE INDEX idx_" $table "_" $column " ON " $table "(" $column ");")
         medium
         "Index on filter/join column")))

;; --- Rewrite Rule: ORDER BY Optimization ---
;; When ORDER BY uses an expression, use alias instead
(= (rewrite-order-by-expression $agg-func $column $alias)
   (rewrite-result order-by
     (concat "ORDER BY " $alias)
     "Using column alias instead of expression for ORDER BY"))

;; ============================================
;; Query Transformation Pipeline
;; ============================================
;; Orchestrates multiple rewrites based on detected patterns

;; Apply all applicable rewrites to a query
(= (optimize-query $query-id $patterns $query-info)
   (apply-rewrites $patterns $query-info ()))

;; Base case: no more patterns
(= (apply-rewrites () $query-info $rewrites)
   (optimization-complete $query-info $rewrites))

;; Recursive case: apply rewrite for each pattern
(= (apply-rewrites ($pattern . $rest) $query-info $rewrites)
   (let $new-rewrite (get-rewrite-for-pattern $pattern $query-info)
        (apply-rewrites $rest $query-info ($new-rewrite . $rewrites))))

;; Get appropriate rewrite based on pattern type
(= (get-rewrite-for-pattern (pattern-detected $qid select-star $sev $desc) $info)
   (match $info (query-columns $cols)
     (rewrite-select-star (get-primary-table $info) $cols)))

(= (get-rewrite-for-pattern (pattern-detected $qid implicit-join $sev $desc) $info)
   (match $info (, (left-table $lt $la) (right-table $rt $ra) (join-condition $cond))
     (rewrite-implicit-join $lt $la $rt $ra $cond)))

(= (get-rewrite-for-pattern (pattern-detected $qid correlated-subquery $sev $desc) $info)
   (match $info (, (outer-table $ot) (inner-table $it) (correlation $col) (aggregation $agg $acol) (condition $cond))
     (rewrite-correlated-subquery $ot $it $col $agg $acol $cond)))

;; ============================================
;; Optimization Priority Ordering
;; ============================================
;; Higher priority rewrites are applied first

(= (rewrite-priority correlated-subquery) 100)
(= (rewrite-priority n-plus-one) 95)
(= (rewrite-priority cartesian-product-risk) 90)
(= (rewrite-priority implicit-join) 80)
(= (rewrite-priority full-table-scan) 70)
(= (rewrite-priority aggregation-full-scan) 60)
(= (rewrite-priority select-star) 30)
(= (rewrite-priority filesort-required) 20)
