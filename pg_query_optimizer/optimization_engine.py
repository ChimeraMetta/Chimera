"""
Query Optimization Engine - Generates optimization suggestions based on detected patterns.

This module coordinates pattern detection and generates actionable optimization
suggestions including:
- Index creation recommendations
- Query rewrite suggestions
- Performance improvement estimates
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .demo_queries import DemoQuery, DEMO_QUERIES, get_demo_query
from .query_analyzer import QueryAnalyzer, QueryStructure, analyze_query_with_metadata
from .pattern_detector import MeTTaQueryPatternDetector, QueryPattern
from .query_rewriter import MeTTaQueryRewriter, OptimizedQuery


@dataclass
class OptimizationSuggestion:
    """Represents an optimization suggestion for a query."""

    query_id: str
    optimization_type: str  # "create-index", "rewrite-join", "subquery-to-join", etc.
    description: str
    suggested_action: str  # The actual SQL or recommendation
    expected_improvement_pct: float
    confidence: float
    patterns_addressed: List[str] = field(default_factory=list)
    metta_reasoning_trace: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query_id": self.query_id,
            "optimization_type": self.optimization_type,
            "description": self.description,
            "suggested_action": self.suggested_action,
            "expected_improvement_pct": self.expected_improvement_pct,
            "confidence": self.confidence,
            "patterns_addressed": self.patterns_addressed,
            "metta_reasoning_trace": self.metta_reasoning_trace
        }


@dataclass
class OptimizationResult:
    """Complete result of analyzing and optimizing a query."""

    query_id: str
    original_sql: str
    original_duration_ms: float
    original_cost: float

    # Analysis results
    query_structure: Optional[QueryStructure] = None
    detected_patterns: List[QueryPattern] = field(default_factory=list)
    suggestions: List[OptimizationSuggestion] = field(default_factory=list)

    # Optimized query info
    optimized_sql: str = ""
    optimized_duration_ms: float = 0.0
    optimized_cost: float = 0.0

    # Reasoning
    metta_atoms: List[str] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def improvement_percentage(self) -> float:
        """Calculate overall improvement percentage."""
        if self.original_duration_ms == 0:
            return 0.0
        return ((self.original_duration_ms - self.optimized_duration_ms)
                / self.original_duration_ms * 100)

    @property
    def cost_reduction(self) -> float:
        """Calculate cost reduction percentage."""
        if self.original_cost == 0:
            return 0.0
        return ((self.original_cost - self.optimized_cost)
                / self.original_cost * 100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query_id": self.query_id,
            "original_sql": self.original_sql,
            "original_duration_ms": self.original_duration_ms,
            "original_cost": self.original_cost,
            "detected_patterns": [p.to_dict() for p in self.detected_patterns],
            "suggestions": [s.to_dict() for s in self.suggestions],
            "optimized_sql": self.optimized_sql,
            "optimized_duration_ms": self.optimized_duration_ms,
            "optimized_cost": self.optimized_cost,
            "improvement_percentage": self.improvement_percentage,
            "cost_reduction": self.cost_reduction,
            "metta_atoms": self.metta_atoms,
            "reasoning_trace": self.reasoning_trace,
            "timestamp": self.timestamp.isoformat()
        }


class OptimizationEngine:
    """
    Main optimization engine that coordinates analysis and suggestion generation.

    This engine:
    1. Analyzes query structure using QueryAnalyzer
    2. Detects patterns using MeTTaQueryPatternDetector
    3. Generates optimization suggestions based on detected patterns
    """

    def __init__(self, use_dynamic_rewriting: bool = True):
        """Initialize the optimization engine.

        Args:
            use_dynamic_rewriting: If True, use MeTTa to dynamically generate
                                   optimized SQL. If False, use pre-defined demos.
        """
        self.query_analyzer = QueryAnalyzer()
        self.pattern_detector = MeTTaQueryPatternDetector()
        self.query_rewriter = MeTTaQueryRewriter()
        self.use_dynamic_rewriting = use_dynamic_rewriting
        self.optimization_history: List[OptimizationResult] = []

    def analyze_demo_query(self, query_id: str) -> Optional[OptimizationResult]:
        """
        Analyze a demo query by ID.

        Args:
            query_id: The demo query ID (e.g., "demo_1")

        Returns:
            OptimizationResult with full analysis
        """
        demo_query = get_demo_query(query_id)
        if not demo_query:
            return None

        return self.analyze_query(demo_query)

    def analyze_query(self, demo_query: DemoQuery) -> OptimizationResult:
        """
        Analyze a query and generate optimization suggestions.

        Args:
            demo_query: DemoQuery object containing query information

        Returns:
            OptimizationResult with patterns, suggestions, and metrics
        """
        # Create result object
        result = OptimizationResult(
            query_id=demo_query.id,
            original_sql=demo_query.original_sql,
            original_duration_ms=demo_query.original_duration_ms,
            original_cost=demo_query.original_cost,
            optimized_sql=demo_query.optimized_sql,
            optimized_duration_ms=demo_query.optimized_duration_ms,
            optimized_cost=demo_query.optimized_cost
        )

        # Step 1: Analyze query structure
        result.reasoning_trace.append("=== Step 1: Query Structure Analysis ===")
        structure, atoms = analyze_query_with_metadata(
            demo_query.original_sql,
            demo_query.table_metadata
        )
        result.query_structure = structure
        result.metta_atoms = atoms

        result.reasoning_trace.append(f"Query type: {structure.query_type}")
        result.reasoning_trace.append(f"Tables: {structure.tables}")
        result.reasoning_trace.append(f"Uses SELECT *: {structure.uses_select_star}")
        result.reasoning_trace.append(f"Uses implicit JOIN: {structure.uses_implicit_join}")
        result.reasoning_trace.append(f"Has subquery: {structure.has_subquery}")

        # Step 2: Detect patterns using MeTTa
        result.reasoning_trace.append("")
        result.reasoning_trace.append("=== Step 2: MeTTa Pattern Detection ===")

        patterns, pattern_trace = self.pattern_detector.detect_patterns(
            demo_query.id,
            atoms,
            anti_patterns=demo_query.anti_patterns
        )
        result.detected_patterns = patterns
        result.reasoning_trace.extend(pattern_trace)

        # Step 3: Generate optimized SQL using MeTTa rewriter
        result.reasoning_trace.append("")
        result.reasoning_trace.append("=== Step 3: MeTTa Query Rewriting ===")

        if self.use_dynamic_rewriting and patterns:
            # Use MeTTa to dynamically generate optimized SQL
            optimized = self.query_rewriter.rewrite_query(
                demo_query.original_sql,
                patterns,
                structure,
                demo_query.table_metadata
            )

            # Update result with dynamically generated SQL
            result.optimized_sql = optimized.optimized_sql
            result.reasoning_trace.extend(optimized.metta_reasoning_trace)

            # Add rewrite details to reasoning trace
            result.reasoning_trace.append("")
            result.reasoning_trace.append("=== Rewrites Applied ===")
            for rewrite in optimized.rewrites_applied:
                result.reasoning_trace.append(f"Pattern: {rewrite.pattern_type}")
                result.reasoning_trace.append(f"  Original: {rewrite.original_fragment}")
                result.reasoning_trace.append(f"  Rewritten: {rewrite.rewritten_fragment}")
                result.reasoning_trace.append(f"  Explanation: {rewrite.explanation}")

            # Add index suggestions
            if optimized.index_suggestions:
                result.reasoning_trace.append("")
                result.reasoning_trace.append("=== Index Suggestions ===")
                for idx in optimized.index_suggestions:
                    result.reasoning_trace.append(f"  {idx}")

        # Step 4: Generate optimization suggestions (for display)
        result.reasoning_trace.append("")
        result.reasoning_trace.append("=== Step 4: Optimization Summary ===")

        suggestions = self._generate_suggestions(demo_query, patterns)
        result.suggestions = suggestions

        for suggestion in suggestions:
            result.reasoning_trace.append(f"Suggestion: {suggestion.optimization_type}")
            result.reasoning_trace.append(f"  Action: {suggestion.suggested_action[:100]}...")
            result.reasoning_trace.append(f"  Expected improvement: {suggestion.expected_improvement_pct:.1f}%")

        # Store in history
        self.optimization_history.append(result)

        return result

    def optimize_raw_query(
        self,
        sql: str,
        table_metadata: Dict[str, Any] = None
    ) -> OptimizationResult:
        """
        Optimize a raw SQL query (not from demo set) using MeTTa reasoning.

        This is the main entry point for optimizing arbitrary queries.

        Args:
            sql: The SQL query to optimize
            table_metadata: Optional table schema information

        Returns:
            OptimizationResult with dynamically generated optimizations
        """
        import uuid

        query_id = f"raw_{uuid.uuid4().hex[:8]}"

        # Create result object
        result = OptimizationResult(
            query_id=query_id,
            original_sql=sql,
            original_duration_ms=0.0,  # Unknown for raw queries
            original_cost=0.0,
        )

        # Step 1: Analyze query structure
        result.reasoning_trace.append("=== Step 1: Query Structure Analysis ===")
        structure, atoms = analyze_query_with_metadata(sql, table_metadata or {})
        result.query_structure = structure
        result.metta_atoms = atoms

        result.reasoning_trace.append(f"Query type: {structure.query_type}")
        result.reasoning_trace.append(f"Tables: {structure.tables}")
        result.reasoning_trace.append(f"Uses SELECT *: {structure.uses_select_star}")
        result.reasoning_trace.append(f"Uses implicit JOIN: {structure.uses_implicit_join}")
        result.reasoning_trace.append(f"Has subquery: {structure.has_subquery}")

        # Step 2: Detect patterns using MeTTa
        result.reasoning_trace.append("")
        result.reasoning_trace.append("=== Step 2: MeTTa Pattern Detection ===")

        patterns, pattern_trace = self.pattern_detector.detect_patterns(
            query_id, atoms, anti_patterns=[]
        )
        result.detected_patterns = patterns
        result.reasoning_trace.extend(pattern_trace)

        # Step 3: Rewrite query using MeTTa
        result.reasoning_trace.append("")
        result.reasoning_trace.append("=== Step 3: MeTTa Query Rewriting ===")

        if patterns:
            optimized = self.query_rewriter.rewrite_query(
                sql, patterns, structure, table_metadata or {}
            )

            result.optimized_sql = optimized.optimized_sql
            result.reasoning_trace.extend(optimized.metta_reasoning_trace)

            # Create suggestions from rewrites
            for rewrite in optimized.rewrites_applied:
                suggestion = OptimizationSuggestion(
                    query_id=query_id,
                    optimization_type=rewrite.pattern_type,
                    description=rewrite.explanation,
                    suggested_action=rewrite.rewritten_fragment,
                    expected_improvement_pct=optimized.total_improvement_pct,
                    confidence=rewrite.confidence,
                    patterns_addressed=[rewrite.pattern_type],
                    metta_reasoning_trace=rewrite.metta_trace
                )
                result.suggestions.append(suggestion)

            # Add index suggestions
            for idx_sql in optimized.index_suggestions:
                suggestion = OptimizationSuggestion(
                    query_id=query_id,
                    optimization_type="create-index",
                    description="Index recommendation",
                    suggested_action=idx_sql,
                    expected_improvement_pct=50.0,
                    confidence=0.85,
                    patterns_addressed=["missing-index"],
                    metta_reasoning_trace=[]
                )
                result.suggestions.append(suggestion)
        else:
            result.optimized_sql = sql
            result.reasoning_trace.append("No anti-patterns detected - query appears optimized")

        # Store in history
        self.optimization_history.append(result)

        return result

    def _generate_suggestions(
        self,
        demo_query: DemoQuery,
        patterns: List[QueryPattern]
    ) -> List[OptimizationSuggestion]:
        """
        Generate optimization suggestions based on detected patterns.

        Args:
            demo_query: The query being analyzed
            patterns: Detected patterns from MeTTa reasoning

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Group patterns by type for suggestion generation
        pattern_types = {p.pattern_type for p in patterns}

        # Generate primary suggestion based on the demo query's optimization type
        primary_suggestion = OptimizationSuggestion(
            query_id=demo_query.id,
            optimization_type=demo_query.optimization_type,
            description=self._get_optimization_description(demo_query.optimization_type),
            suggested_action=self._format_suggestion_action(demo_query),
            expected_improvement_pct=demo_query.improvement_percentage,
            confidence=0.95,
            patterns_addressed=list(pattern_types),
            metta_reasoning_trace=[
                f"(suggest-optimization {demo_query.id} {demo_query.optimization_type})",
                f"(optimization-quality {demo_query.optimization_type} high)",
                f"(expected-improvement {demo_query.optimization_type}) -> {demo_query.improvement_percentage:.1f}%"
            ]
        )
        suggestions.append(primary_suggestion)

        # Add index suggestion if applicable
        if demo_query.index_suggestion:
            index_suggestion = OptimizationSuggestion(
                query_id=demo_query.id,
                optimization_type="create-index",
                description="Create index to improve query performance",
                suggested_action=demo_query.index_suggestion,
                expected_improvement_pct=min(demo_query.improvement_percentage, 95.0),
                confidence=0.9,
                patterns_addressed=["full-table-scan", "missing-index"],
                metta_reasoning_trace=[
                    f"(suggest-index {demo_query.id} ...)",
                    "(optimization-quality create-index high)"
                ]
            )
            suggestions.append(index_suggestion)

        return suggestions

    def _get_optimization_description(self, opt_type: str) -> str:
        """Get human-readable description for optimization type."""
        descriptions = {
            "create-index": "Create an index to eliminate full table scans",
            "rewrite-join": "Rewrite implicit join as explicit JOIN for better optimization",
            "subquery-to-join": "Transform correlated subquery to JOIN for single-pass execution",
            "create-covering-index": "Create covering index to include aggregation columns",
            "add-predicate-pushdown": "Push predicates closer to data source for early filtering"
        }
        return descriptions.get(opt_type, f"Optimization: {opt_type}")

    def _format_suggestion_action(self, demo_query: DemoQuery) -> str:
        """Format the suggestion action for display."""
        action_parts = []

        # Add optimized query
        action_parts.append("-- Optimized Query:")
        action_parts.append(demo_query.optimized_sql)

        # Add index if applicable
        if demo_query.index_suggestion:
            action_parts.append("")
            action_parts.append("-- Recommended Index:")
            action_parts.append(demo_query.index_suggestion)

        return "\n".join(action_parts)

    def analyze_all_demos(self) -> List[OptimizationResult]:
        """
        Analyze all demo queries and return results.

        Returns:
            List of OptimizationResult for all demo queries
        """
        results = []
        for demo_query in DEMO_QUERIES:
            result = self.analyze_query(demo_query)
            results.append(result)
        return results

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary metrics from optimization history.

        Returns:
            Dictionary with summary statistics
        """
        if not self.optimization_history:
            return {
                "total_queries_analyzed": 0,
                "total_patterns_detected": 0,
                "total_suggestions_generated": 0,
                "average_improvement_pct": 0.0
            }

        total_patterns = sum(len(r.detected_patterns) for r in self.optimization_history)
        total_suggestions = sum(len(r.suggestions) for r in self.optimization_history)
        improvements = [r.improvement_percentage for r in self.optimization_history if r.improvement_percentage > 0]

        return {
            "total_queries_analyzed": len(self.optimization_history),
            "total_patterns_detected": total_patterns,
            "total_suggestions_generated": total_suggestions,
            "average_improvement_pct": sum(improvements) / len(improvements) if improvements else 0.0,
            "patterns_by_severity": self._count_patterns_by_severity(),
            "optimizations_by_type": self._count_optimizations_by_type()
        }

    def _count_patterns_by_severity(self) -> Dict[str, int]:
        """Count patterns grouped by severity."""
        counts = {"high": 0, "medium": 0, "low": 0}
        for result in self.optimization_history:
            for pattern in result.detected_patterns:
                severity = pattern.severity
                counts[severity] = counts.get(severity, 0) + 1
        return counts

    def _count_optimizations_by_type(self) -> Dict[str, int]:
        """Count optimizations grouped by type."""
        counts: Dict[str, int] = {}
        for result in self.optimization_history:
            for suggestion in result.suggestions:
                opt_type = suggestion.optimization_type
                counts[opt_type] = counts.get(opt_type, 0) + 1
        return counts
