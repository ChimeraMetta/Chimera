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
