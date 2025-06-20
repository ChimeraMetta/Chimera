#!/usr/bin/env python3
"""
Comprehensive Fitness Evaluation System for Semantic Evolution
Evaluates correctness, efficiency, and maintainability of evolved solutions
"""

import ast
import time
import traceback
from typing import List, Dict, Any, Optional, Callable, Tuple
from metta_generator.genetics.semantic_genome import SemanticGenome

class FitnessEvaluator:
    """Comprehensive fitness evaluation for semantic genomes"""
    
    def __init__(self, reference_function: Optional[Callable] = None, 
                 test_cases: Optional[List[Tuple]] = None):
        self.reference_function = reference_function
        self.test_cases = test_cases or []
        self.evaluation_cache = {}  # Cache evaluation results
        
        # Weights for different fitness components
        self.correctness_weight = 0.6
        self.efficiency_weight = 0.3
        self.maintainability_weight = 0.1
        
        # Performance tracking
        self.evaluation_count = 0
        self.cache_hits = 0
    
    def evaluate_genome(self, genome: SemanticGenome, generated_code: str) -> Dict[str, float]:
        """Comprehensive evaluation of a semantic genome"""
        self.evaluation_count += 1
        
        # Check cache first
        cache_key = (genome.genome_id, hash(generated_code))
        if cache_key in self.evaluation_cache:
            self.cache_hits += 1
            return self.evaluation_cache[cache_key]
        
        # Evaluate all components
        results = {
            "correctness_score": self._evaluate_correctness(generated_code, genome),
            "efficiency_score": self._evaluate_efficiency(generated_code, genome),
            "maintainability_score": self._evaluate_maintainability(generated_code, genome),
            "semantic_consistency_score": self._evaluate_semantic_consistency(genome),
            "metta_reasoning_score": self._evaluate_metta_reasoning(genome)
        }
        
        # Calculate overall fitness
        results["overall_fitness"] = (
            self.correctness_weight * results["correctness_score"] +
            self.efficiency_weight * results["efficiency_score"] +
            self.maintainability_weight * results["maintainability_score"]
        )
        
        # Bonus for semantic consistency and MeTTa reasoning
        results["overall_fitness"] += 0.05 * results["semantic_consistency_score"]
        results["overall_fitness"] += 0.05 * results["metta_reasoning_score"]
        results["overall_fitness"] = min(1.0, results["overall_fitness"])
        
        # Cache results
        self.evaluation_cache[cache_key] = results
        
        return results
    
    def _evaluate_correctness(self, code: str, genome: SemanticGenome) -> float:
        """Evaluate correctness through comprehensive testing"""
        if not self.test_cases or not self.reference_function:
            return self._evaluate_syntactic_correctness(code)
        
        correctness_score = 0.0
        test_results = {
            "passed": 0,
            "failed": 0,
            "errors": [],
            "edge_case_performance": 0.0
        }
        
        try:
            # Execute the generated code
            exec_globals = {}
            exec(code, exec_globals)
            
            # Find the evolved function
            evolved_func = None
            for name, obj in exec_globals.items():
                if callable(obj) and name.startswith(('evolved_', 'semantic_')):
                    evolved_func = obj
                    break
            
            if not evolved_func:
                return 0.0
            
            # Test against all test cases
            for i, test_case in enumerate(self.test_cases):
                try:
                    expected = self.reference_function(*test_case)
                    actual = evolved_func(*test_case)
                    
                    if self._results_match(expected, actual):
                        test_results["passed"] += 1
                        # Bonus for edge cases (first and last tests)
                        if i in [0, len(self.test_cases) - 1]:
                            test_results["edge_case_performance"] += 0.1
                    else:
                        test_results["failed"] += 1
                        test_results["errors"].append(f"Test {i}: expected {expected}, got {actual}")
                
                except Exception as e:
                    test_results["failed"] += 1
                    test_results["errors"].append(f"Test {i}: {str(e)}")
            
            # Calculate correctness score
            base_correctness = test_results["passed"] / len(self.test_cases)
            edge_case_bonus = test_results["edge_case_performance"]
            correctness_score = min(1.0, base_correctness + edge_case_bonus)
            
            # Store test results in genome
            genome.test_results = test_results
            
        except Exception as e:
            test_results["errors"].append(f"Execution error: {str(e)}")
            genome.test_results = test_results
            return 0.0
        
        return correctness_score
    
    def _evaluate_syntactic_correctness(self, code: str) -> float:
        """Evaluate syntactic correctness when no tests available"""
        try:
            # Check if code parses
            ast.parse(code)
            
            # Basic structure checks
            score = 0.5  # Base score for valid syntax
            
            # Check for proper function definition
            if "def " in code and "return" in code:
                score += 0.3
            
            # Check for proper error handling
            if "if " in code and ("None" in code or "return" in code):
                score += 0.2
            
            return min(1.0, score)
        
        except SyntaxError:
            return 0.0
    
    def _evaluate_efficiency(self, code: str, genome: SemanticGenome) -> float:
        """Evaluate efficiency through static analysis and complexity metrics"""
        efficiency_score = 1.0
        
        try:
            # Parse code for analysis
            tree = ast.parse(code)
            
            # Count complexity indicators
            complexity_metrics = self._analyze_code_complexity(tree)
            
            # Penalize excessive complexity
            if complexity_metrics["nested_loops"] > 1:
                efficiency_score -= 0.3
            elif complexity_metrics["nested_loops"] == 1:
                efficiency_score -= 0.1
            
            if complexity_metrics["condition_count"] > 3:
                efficiency_score -= 0.2
            
            if complexity_metrics["function_calls"] > 5:
                efficiency_score -= 0.1
            
            # Reward efficient patterns
            if complexity_metrics["early_returns"] > 0:
                efficiency_score += 0.1
            
            # Analyze semantic efficiency
            semantic_efficiency = self._analyze_semantic_efficiency(genome)
            efficiency_score = (efficiency_score + semantic_efficiency) / 2
            
        except Exception as e:
            print(f"Efficiency evaluation error: {e}")
            efficiency_score = 0.5
        
        return max(0.0, min(1.0, efficiency_score))
    
    def _analyze_code_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """Analyze code complexity metrics"""
        metrics = {
            "nested_loops": 0,
            "condition_count": 0,
            "function_calls": 0,
            "early_returns": 0,
            "variable_assignments": 0
        }
        
        class ComplexityAnalyzer(ast.NodeVisitor):
            def __init__(self, metrics):
                self.metrics = metrics
                self.loop_depth = 0
            
            def visit_For(self, node):
                self.loop_depth += 1
                if self.loop_depth > 1:
                    self.metrics["nested_loops"] += 1
                self.generic_visit(node)
                self.loop_depth -= 1
            
            def visit_While(self, node):
                self.loop_depth += 1
                if self.loop_depth > 1:
                    self.metrics["nested_loops"] += 1
                self.generic_visit(node)
                self.loop_depth -= 1
            
            def visit_If(self, node):
                self.metrics["condition_count"] += 1
                self.generic_visit(node)
            
            def visit_Call(self, node):
                self.metrics["function_calls"] += 1
                self.generic_visit(node)
            
            def visit_Return(self, node):
                self.metrics["early_returns"] += 1
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                self.metrics["variable_assignments"] += 1
                self.generic_visit(node)
        
        analyzer = ComplexityAnalyzer(metrics)
        analyzer.visit(tree)
        
        return metrics
    
    def _analyze_semantic_efficiency(self, genome: SemanticGenome) -> float:
        """Analyze efficiency based on semantic structure"""
        efficiency = 0.8  # Base efficiency
        
        # Check for efficient semantic patterns
        gene_types = [gene.gene_type for gene in genome.genes]
        
        # Reward minimal but complete gene sets
        unique_types = len(set(gene_types))
        if 4 <= unique_types <= 6:  # Sweet spot for completeness without redundancy
            efficiency += 0.1
        elif unique_types > 6:
            efficiency -= 0.1
        
        # Check for semantic optimization genes
        optimization_genes = genome.get_genes_by_type("optimization")
        if optimization_genes:
            efficiency += 0.1
        
        # Penalize semantic conflicts (they indicate inefficient gene combinations)
        valid, errors = genome.validate_semantics()
        if not valid:
            efficiency -= 0.1 * len(errors)
        
        return max(0.0, min(1.0, efficiency))
    
    def _evaluate_maintainability(self, code: str, genome: SemanticGenome) -> float:
        """Evaluate maintainability through code quality metrics"""
        maintainability = 0.7  # Base score
        
        try:
            # Code length and readability
            lines = [line.strip() for line in code.split('\n') if line.strip()]
            
            # Reward reasonable length (not too short, not too long)
            if 10 <= len(lines) <= 25:
                maintainability += 0.1
            elif len(lines) > 30:
                maintainability -= 0.1
            
            # Check for documentation
            if '"""' in code or "'''" in code:
                maintainability += 0.1
            
            # Check for meaningful variable names
            if self._has_meaningful_names(code):
                maintainability += 0.1
            
            # Semantic maintainability
            semantic_maintainability = self._evaluate_semantic_maintainability(genome)
            maintainability = (maintainability + semantic_maintainability) / 2
            
        except Exception as e:
            print(f"Maintainability evaluation error: {e}")
            maintainability = 0.5
        
        return max(0.0, min(1.0, maintainability))
    
    def _evaluate_semantic_maintainability(self, genome: SemanticGenome) -> float:
        """Evaluate maintainability based on semantic structure"""
        maintainability = 0.8
        
        # Check semantic consistency
        valid, errors = genome.validate_semantics()
        if valid:
            maintainability += 0.2
        else:
            maintainability -= 0.05 * len(errors)
        
        # Reward clear semantic roles
        clear_roles = sum(1 for gene in genome.genes if gene.semantic_role and len(gene.semantic_role) > 5)
        if clear_roles == len(genome.genes):
            maintainability += 0.1
        
        return max(0.0, min(1.0, maintainability))
    
    def _evaluate_semantic_consistency(self, genome: SemanticGenome) -> float:
        """Evaluate semantic consistency of the genome"""
        valid, errors = genome.validate_semantics()
        
        if valid:
            return 1.0
        else:
            # Partial credit based on error severity
            return max(0.0, 1.0 - (len(errors) * 0.2))
    
    def _evaluate_metta_reasoning(self, genome: SemanticGenome) -> float:
        """Evaluate quality of MeTTa reasoning in genome"""
        reasoning_score = 0.5  # Base score
        
        # Check for MeTTa derivations in genes
        genes_with_metta = sum(1 for gene in genome.genes if gene.metta_derivation)
        if genes_with_metta > 0:
            reasoning_score += 0.3 * (genes_with_metta / len(genome.genes))
        
        # Check reasoning confidence
        avg_confidence = sum(gene.reasoning_confidence for gene in genome.genes) / len(genome.genes)
        reasoning_score += 0.2 * avg_confidence
        
        return min(1.0, reasoning_score)
    
    def _results_match(self, expected, actual) -> bool:
        """Check if results match, handling different types"""
        if expected is None and actual is None:
            return True
        
        if type(expected) != type(actual):
            return False
        
        if isinstance(expected, float):
            return abs(expected - actual) < 1e-9
        
        return expected == actual
    
    def _has_meaningful_names(self, code: str) -> bool:
        """Check if code has meaningful variable names"""
        meaningful_patterns = ['result', 'max_val', 'min_val', 'index', 'value', 'data']
        return any(pattern in code for pattern in meaningful_patterns)
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation performance statistics"""
        cache_hit_rate = self.cache_hits / max(1, self.evaluation_count)
        return {
            "total_evaluations": self.evaluation_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "cached_results": len(self.evaluation_cache)
        }
    
    def clear_cache(self):
        """Clear evaluation cache"""
        self.evaluation_cache.clear()
        self.cache_hits = 0
        self.evaluation_count = 0