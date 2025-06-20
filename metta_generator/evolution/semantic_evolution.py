#!/usr/bin/env python3
"""
Main Semantic Evolution Engine
Advanced evolution with semantic understanding and MeTTa reasoning integration
"""

import random
import os
from typing import List, Dict, Any, Optional, Callable
from metta_generator.genetics.semantic_genome import (
    SemanticGenome, SemanticGene, SemanticGeneType, SemanticGenePool
)
from metta_generator.evolution.fitness_evaluator import FitnessEvaluator

SEMANTIC_EVOLUTION_ONTOLOGY = "semantic_evolution.metta"

class SemanticEvolutionEngine:
    """Advanced evolution engine with semantic awareness and MeTTa reasoning"""
    
    def __init__(self, metta_space=None, reasoning_engine=None, 
                 population_size=20, max_generations=10):
        self.metta_space = metta_space
        self.reasoning_engine = reasoning_engine
        self.population_size = population_size
        self.max_generations = max_generations
        
        # Enhanced components
        self.semantic_gene_pool = SemanticGenePool()
        self.fitness_evaluator = FitnessEvaluator()
        
        # Evolution state
        self.population: List[SemanticGenome] = []
        self.generation = 0
        self.evolution_history = []
        self.best_genome_history = []
        
        # Initialize gene pool
        self._initialize_semantic_gene_pool()
        
        # MeTTa reasoning integration
        if self.reasoning_engine:
            rules_file = os.path.join(os.path.dirname(__file__), "..", "..", "metta", SEMANTIC_EVOLUTION_ONTOLOGY)
            rules_loaded = load_semantic_evolution_rules(self.reasoning_engine, rules_file)

            if rules_loaded:
                print("  SemanticEvolutionEngine: MeTTa reasoning rules loaded")
            else:
                print("  SemanticEvolutionEngine: Failed to load MeTTa rules, using defaults")
    
    def _initialize_semantic_gene_pool(self):
        """Initialize semantic gene pool with meaningful program components"""
        
        # Initialization genes
        init_genes = [
            SemanticGene(
                gene_type=SemanticGeneType.INITIALIZATION,
                semantic_role="validate_input_bounds",
                preconditions=["input_received"],
                postconditions=["bounds_validated", "input_safe"],
                code_template="if start_idx < 0 or end_idx > len({data}) or start_idx >= end_idx:\n        return None",
                parameter_slots={"data": "input_collection"},
                metta_derivation=["(validation-pattern input-bounds)"]
            ),
            SemanticGene(
                gene_type=SemanticGeneType.INITIALIZATION,
                semantic_role="initialize_accumulator",
                preconditions=["input_safe"],
                postconditions=["accumulator_ready"],
                code_template="    {result} = {data}[{start_idx}]",
                parameter_slots={"result": "accumulator", "data": "input_collection", "start_idx": "start_index"},
                metta_derivation=["(initialization-pattern accumulator)"]
            ),
            SemanticGene(
                gene_type=SemanticGeneType.INITIALIZATION,
                semantic_role="initialize_with_first_valid",
                preconditions=["input_safe"],
                postconditions=["accumulator_ready"],
                code_template="    {result} = None\n    if {start_idx} < len({data}):\n        {result} = {data}[{start_idx}]",
                parameter_slots={"result": "accumulator", "data": "input_collection", "start_idx": "start_index"},
                metta_derivation=["(safe-initialization-pattern accumulator)"]
            )
        ]
        
        for gene in init_genes:
            self.semantic_gene_pool.add_gene(gene)
        
        # Iteration genes
        iteration_genes = [
            SemanticGene(
                gene_type=SemanticGeneType.ITERATION,
                semantic_role="forward_linear_scan",
                preconditions=["accumulator_ready"],
                postconditions=["elements_processed"],
                code_template="    for i in range({start_idx} + 1, {end_idx}):",
                parameter_slots={"start_idx": "start_index", "end_idx": "end_index"},
                metta_derivation=["(iteration-pattern forward-scan)"]
            ),
            SemanticGene(
                gene_type=SemanticGeneType.ITERATION,
                semantic_role="indexed_iteration",
                preconditions=["accumulator_ready"],
                postconditions=["elements_processed"],
                code_template="    for idx in range({start_idx}, {end_idx}):\n        i = idx",
                parameter_slots={"start_idx": "start_index", "end_idx": "end_index"},
                metta_derivation=["(iteration-pattern indexed)"]
            ),
            SemanticGene(
                gene_type=SemanticGeneType.ITERATION,
                semantic_role="enumerated_iteration",
                preconditions=["accumulator_ready"],
                postconditions=["elements_processed"],
                code_template="    for i, value in enumerate({data}[{start_idx}:{end_idx}], {start_idx}):",
                parameter_slots={"data": "input_collection", "start_idx": "start_index", "end_idx": "end_index"},
                metta_derivation=["(iteration-pattern enumerated)"]
            )
        ]
        
        for gene in iteration_genes:
            self.semantic_gene_pool.add_gene(gene)
        
        # Condition genes
        condition_genes = [
            SemanticGene(
                gene_type=SemanticGeneType.CONDITION,
                semantic_role="maximize_condition",
                preconditions=["elements_processed"],
                postconditions=["optimal_element_found"],
                code_template="        if {data}[i] > {result}:",
                parameter_slots={"data": "input_collection", "result": "accumulator"},
                metta_derivation=["(condition-pattern maximize)"]
            ),
            SemanticGene(
                gene_type=SemanticGeneType.CONDITION,
                semantic_role="minimize_condition",
                preconditions=["elements_processed"],
                postconditions=["optimal_element_found"],
                code_template="        if {data}[i] < {result}:",
                parameter_slots={"data": "input_collection", "result": "accumulator"},
                metta_derivation=["(condition-pattern minimize)"]
            ),
            SemanticGene(
                gene_type=SemanticGeneType.CONDITION,
                semantic_role="safe_comparison",
                preconditions=["elements_processed"],
                postconditions=["optimal_element_found"],
                code_template="        if {result} is None or {data}[i] > {result}:",
                parameter_slots={"data": "input_collection", "result": "accumulator"},
                metta_derivation=["(condition-pattern safe-maximize)"]
            )
        ]
        
        for gene in condition_genes:
            self.semantic_gene_pool.add_gene(gene)
        
        # Operation genes
        operation_genes = [
            SemanticGene(
                gene_type=SemanticGeneType.OPERATION,
                semantic_role="update_accumulator",
                preconditions=["optimal_element_found"],
                postconditions=["accumulator_updated"],
                code_template="            {result} = {data}[i]",
                parameter_slots={"result": "accumulator", "data": "input_collection"},
                metta_derivation=["(operation-pattern update-accumulator)"]
            ),
            SemanticGene(
                gene_type=SemanticGeneType.OPERATION,
                semantic_role="track_position",
                preconditions=["optimal_element_found"],
                postconditions=["position_tracked"],
                code_template="            {result_pos} = i",
                parameter_slots={"result_pos": "best_position"},
                metta_derivation=["(operation-pattern track-position)"]
            ),
            SemanticGene(
                gene_type=SemanticGeneType.OPERATION,
                semantic_role="update_with_value",
                preconditions=["optimal_element_found"],
                postconditions=["accumulator_updated"],
                code_template="            {result} = value",
                parameter_slots={"result": "accumulator"},
                metta_derivation=["(operation-pattern update-with-value)"]
            )
        ]
        
        for gene in operation_genes:
            self.semantic_gene_pool.add_gene(gene)
        
        # Termination genes
        termination_genes = [
            SemanticGene(
                gene_type=SemanticGeneType.TERMINATION,
                semantic_role="return_result",
                preconditions=["accumulator_updated"],
                postconditions=["result_returned"],
                code_template="    return {result}",
                parameter_slots={"result": "accumulator"},
                metta_derivation=["(termination-pattern return-result)"]
            ),
            SemanticGene(
                gene_type=SemanticGeneType.TERMINATION,
                semantic_role="return_with_position",
                preconditions=["position_tracked"],
                postconditions=["result_returned"],
                code_template="    return {result}, {result_pos}",
                parameter_slots={"result": "accumulator", "result_pos": "best_position"},
                metta_derivation=["(termination-pattern return-with-position)"]
            )
        ]
        
        for gene in termination_genes:
            self.semantic_gene_pool.add_gene(gene)
        
        # Error handling genes
        error_genes = [
            SemanticGene(
                gene_type=SemanticGeneType.ERROR_HANDLING,
                semantic_role="null_check",
                preconditions=["input_received"],
                postconditions=["null_handled"],
                code_template="    if {data} is None or len({data}) == 0:\n        return None",
                parameter_slots={"data": "input_collection"},
                metta_derivation=["(error-handling-pattern null-check)"]
            )
        ]
        
        for gene in error_genes:
            self.semantic_gene_pool.add_gene(gene)
    
    def setup_for_function(self, reference_function: Callable):
        """Setup evolution for a specific reference function"""
        
        # Setup fitness evaluator
        self.fitness_evaluator = FitnessEvaluator(
            reference_function=reference_function,
        )
        
        print(f"  Setup complete: {len(self.test_framework.test_cases)} test cases generated")
    
    def evolve_solutions(self, target_semantics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Main evolution loop with semantic awareness"""
        print(f"\n=== Starting Semantic Evolution ===")
        print(f"Population: {self.population_size}, Generations: {self.max_generations}")
        print(f"Target semantics: {target_semantics}")
        
        # Initialize population
        self.population = self._create_semantic_initial_population(target_semantics)
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.generation = generation
            print(f"\n--- Generation {generation} ---")
            
            # Evaluate fitness
            self._evaluate_population_fitness()
            
            # Track best genome
            best_genome = max(self.population, key=lambda g: g.overall_fitness)
            self.best_genome_history.append(best_genome.clone())
            
            # Show statistics
            self._show_generation_stats()
            
            # Create next generation
            if generation < self.max_generations - 1:
                self.population = self._create_next_generation_semantic()
        
        # Convert to compatible format and return results
        return self._format_results_for_compatibility()
    
    def _create_semantic_initial_population(self, target_semantics: Dict[str, Any]) -> List[SemanticGenome]:
        """Create initial population with semantic diversity"""
        population = []
        
        for i in range(self.population_size):
            genome = SemanticGenome(
                generation=0,
                program_purpose=target_semantics.get("purpose", "optimization"),
                input_constraints=target_semantics.get("input_constraints", ["input_received"]),
                output_specification=target_semantics.get("output_spec", "optimal_result")
            )
            
            # Build semantically valid gene sequence
            self._build_semantic_gene_sequence(genome, target_semantics)
            population.append(genome)
        
        return population
    
    def _build_semantic_gene_sequence(self, genome: SemanticGenome, target_semantics: Dict[str, Any]):
        """Build semantically coherent gene sequence"""
        purpose = target_semantics.get("purpose", "generic")
        
        # Required gene types in semantic order
        gene_sequence = [
            (SemanticGeneType.INITIALIZATION, "validate_input_bounds"),
            (SemanticGeneType.INITIALIZATION, "initialize"),
            (SemanticGeneType.ITERATION, "scan"),
            (SemanticGeneType.CONDITION, purpose if purpose in ["maximize", "minimize"] else "maximize"),
            (SemanticGeneType.OPERATION, "update"),
            (SemanticGeneType.TERMINATION, "return")
        ]
        
        for gene_type, role_hint in gene_sequence:
            selected_gene = self._select_gene_with_reasoning(gene_type, role_hint, genome, target_semantics)
            if selected_gene:
                genome.add_gene(selected_gene)
    
    def _select_gene_with_reasoning(self, gene_type: SemanticGeneType, role_hint: str,
                                  genome: SemanticGenome, target_semantics: Dict[str, Any]) -> Optional[SemanticGene]:
        """Select gene using MeTTa reasoning when available"""
        available_genes = self.semantic_gene_pool.get_genes_by_type(gene_type)
        
        if not available_genes:
            return None
        
        if self.reasoning_engine and len(available_genes) > 1:
            # Try MeTTa reasoning for selection
            best_gene = self._metta_guided_gene_selection(available_genes, role_hint, target_semantics)
            if best_gene:
                return best_gene.clone()
        
        # Fallback to semantic selection
        return self._semantic_gene_selection(available_genes, role_hint, target_semantics)
    
    def _metta_guided_gene_selection(self, available_genes: List[SemanticGene], 
                                   role_hint: str, target_semantics: Dict[str, Any]) -> Optional[SemanticGene]:
        """Use MeTTa reasoning to select optimal gene"""
        if not self.reasoning_engine:
            return None
        
        try:
            purpose = target_semantics.get("purpose", "generic")
            
            # Query MeTTa for gene selection
            for gene in available_genes:
                if role_hint in gene.semantic_role or purpose in gene.semantic_role:
                    compatibility_query = f"""
                    (match &self
                      (select-gene {purpose} {gene.semantic_role.replace('_', '-')})
                      True)
                    """
                    
                    results = self.reasoning_engine._execute_metta_reasoning(compatibility_query, [])
                    if results:
                        gene.reasoning_confidence = 0.9
                        return gene
        except Exception as e:
            print(f"MeTTa gene selection failed: {e}")
        
        return None
    
    def _semantic_gene_selection(self, available_genes: List[SemanticGene], 
                               role_hint: str, target_semantics: Dict[str, Any]) -> SemanticGene:
        """Fallback semantic gene selection"""
        purpose = target_semantics.get("purpose", "generic")
        
        # Look for genes matching the role hint
        matching_genes = [g for g in available_genes if role_hint in g.semantic_role]
        if matching_genes:
            return random.choice(matching_genes).clone()
        
        # Look for genes matching the purpose
        if purpose in ["maximize", "minimize"]:
            purpose_genes = [g for g in available_genes if purpose in g.semantic_role]
            if purpose_genes:
                return random.choice(purpose_genes).clone()
        
        # Random selection with preference for higher success rates
        if available_genes:
            weights = [max(0.1, gene.success_rate) for gene in available_genes]
            return random.choices(available_genes, weights=weights)[0].clone()
        
        return available_genes[0].clone()
    
    def _evaluate_population_fitness(self):
        """Comprehensive fitness evaluation for all genomes"""
        for genome in self.population:
            # Generate code from genome
            code = self._generate_code_from_genome(genome)
            
            # Evaluate using fitness evaluator
            fitness_results = self.fitness_evaluator.evaluate_genome(genome, code)
            
            # Update genome fitness scores
            genome.correctness_score = fitness_results["correctness_score"]
            genome.efficiency_score = fitness_results["efficiency_score"]
            genome.maintainability_score = fitness_results["maintainability_score"]
            genome.overall_fitness = fitness_results["overall_fitness"]
            genome.fitness_scores = fitness_results
            
            # Update gene success rates
            for gene in genome.genes:
                gene.add_fitness_record(genome.overall_fitness)
                self.semantic_gene_pool.mark_gene_successful(gene, genome.overall_fitness)
    
    def _generate_code_from_genome(self, genome: SemanticGenome) -> str:
        """Generate executable code from semantic genome"""
        code_lines = []
        
        # Function signature
        code_lines.append("def semantic_evolved_solution(numbers, start_idx, end_idx):")
        code_lines.append('    """Generated by semantic evolution with MeTTa reasoning"""')
        
        # Variable context for template instantiation
        context_vars = {
            "input_collection": "numbers",
            "start_index": "start_idx", 
            "end_index": "end_idx",
            "accumulator": "result",
            "best_position": "best_pos"
        }
        
        # Generate code from genes in sequence
        for gene in genome.genes:
            gene_code = self._instantiate_gene_template(gene, context_vars)
            if gene_code:
                code_lines.append(gene_code)
        
        return "\n".join(code_lines)
    
    def _instantiate_gene_template(self, gene: SemanticGene, context_vars: Dict[str, str]) -> str:
        """Instantiate gene template with context variables"""
        template = gene.code_template
        
        # Replace parameter slots with actual variables
        for slot, var_type in gene.parameter_slots.items():
            if var_type in context_vars:
                template = template.replace(f"{{{slot}}}", context_vars[var_type])
            elif slot in context_vars:
                template = template.replace(f"{{{slot}}}", context_vars[slot])
        
        return template
    
    def _show_generation_stats(self):
        """Show generation statistics"""
        fitnesses = [g.overall_fitness for g in self.population]
        correctness_scores = [g.correctness_score for g in self.population]
        efficiency_scores = [g.efficiency_score for g in self.population]
        
        best_genome = max(self.population, key=lambda g: g.overall_fitness)
        
        print(f"  Best fitness: {best_genome.overall_fitness:.3f} (C:{best_genome.correctness_score:.3f}, E:{best_genome.efficiency_score:.3f})")
        print(f"  Average fitness: {sum(fitnesses) / len(fitnesses):.3f}")
        print(f"  Best correctness: {max(correctness_scores):.3f}")
        print(f"  Average correctness: {sum(correctness_scores) / len(correctness_scores):.3f}")
        print(f"  Best genome: {len(best_genome.genes)} genes, ID: {best_genome.genome_id}")
        print(f"  Semantic roles: {[g.semantic_role for g in best_genome.genes]}")
        
        # Track evolution progress
        generation_stats = {
            "generation": self.generation,
            "best_fitness": best_genome.overall_fitness,
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "best_correctness": max(correctness_scores),
            "avg_correctness": sum(correctness_scores) / len(correctness_scores),
            "gene_pool_stats": self.semantic_gene_pool.get_stats()
        }
        self.evolution_history.append(generation_stats)
    
    def _create_next_generation_semantic(self) -> List[SemanticGenome]:
        """Create next generation using semantic-aware operations"""
        next_generation = []
        
        # Elitism - keep best 20%
        elite_count = max(1, self.population_size // 5)
        elites = sorted(self.population, key=lambda g: g.overall_fitness, reverse=True)[:elite_count]
        next_generation.extend([elite.clone() for elite in elites])
        
        # Generate offspring
        while len(next_generation) < self.population_size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Semantic crossover
            offspring = self._semantic_crossover(parent1, parent2)
            
            # Semantic mutation
            if random.random() < 0.7:  # 70% mutation rate
                offspring = self._semantic_mutation(offspring)
            
            next_generation.append(offspring)
        
        return next_generation[:self.population_size]
    
    def _tournament_selection(self, tournament_size: int = 3) -> SemanticGenome:
        """Tournament selection for parent selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda g: g.overall_fitness)
    
    def _semantic_crossover(self, parent1: SemanticGenome, parent2: SemanticGenome) -> SemanticGenome:
        """Semantic-aware crossover preserving program semantics"""
        offspring = SemanticGenome(
            generation=self.generation + 1,
            parent_ids=[parent1.genome_id, parent2.genome_id],
            program_purpose=parent1.program_purpose,
            input_constraints=parent1.input_constraints,
            output_specification=parent1.output_specification
        )
        
        # Combine genes while maintaining semantic consistency
        for gene_type in SemanticGeneType:
            p1_genes = parent1.get_genes_by_type(gene_type)
            p2_genes = parent2.get_genes_by_type(gene_type)
            
            if p1_genes and p2_genes:
                # Choose gene from better performing parent for this type
                p1_avg_fitness = sum(g.success_rate for g in p1_genes) / len(p1_genes)
                p2_avg_fitness = sum(g.success_rate for g in p2_genes) / len(p2_genes)
                
                if p1_avg_fitness > p2_avg_fitness:
                    chosen_gene = random.choice(p1_genes).clone()
                else:
                    chosen_gene = random.choice(p2_genes).clone()
                    
                offspring.add_gene(chosen_gene)
            elif p1_genes:
                offspring.add_gene(random.choice(p1_genes).clone())
            elif p2_genes:
                offspring.add_gene(random.choice(p2_genes).clone())
        
        return offspring
    
    def _semantic_mutation(self, genome: SemanticGenome) -> SemanticGenome:
        """Semantic-preserving mutation"""
        mutated = genome.clone()
        
        # Randomly mutate one gene while preserving semantics
        if mutated.genes and random.random() < 0.5:  # 50% chance to mutate a gene
            gene_to_mutate = random.choice(mutated.genes)
            gene_type = gene_to_mutate.gene_type
            
            # Replace with alternative gene of same type
            alternatives = self.semantic_gene_pool.get_genes_by_type(gene_type)
            suitable_alternatives = [
                g for g in alternatives 
                if g.semantic_role != gene_to_mutate.semantic_role
            ]
            
            if suitable_alternatives:
                new_gene = random.choice(suitable_alternatives).clone()
                new_gene.generation_born = self.generation + 1
                
                gene_index = mutated.genes.index(gene_to_mutate)
                mutated.genes[gene_index] = new_gene
        
        # Sometimes add beneficial genes
        if random.random() < 0.2:  # 20% chance to add error handling
            error_genes = self.semantic_gene_pool.get_genes_by_type(SemanticGeneType.ERROR_HANDLING)
            if error_genes and not mutated.get_genes_by_type(SemanticGeneType.ERROR_HANDLING):
                error_gene = random.choice(error_genes).clone()
                error_gene.generation_born = self.generation + 1
                mutated.genes.insert(1, error_gene)  # Add after function definition
        
        return mutated
    
    def _format_results_for_compatibility(self) -> List[Dict[str, Any]]:
        """Format results to be compatible with existing donor candidate format"""
        results = []
        
        # Sort by fitness
        sorted_genomes = sorted(self.population, key=lambda g: g.overall_fitness, reverse=True)
        
        for i, genome in enumerate(sorted_genomes[:10]):  # Top 10 results
            # Generate code
            code = self._generate_code_from_genome(genome)
            
            # Create compatible result format
            result = {
                "name": f"semantic_evolved_{genome.genome_id}",
                "description": f"Semantically evolved solution (Gen {genome.generation})",
                "code": code,
                "strategy": "semantic_evolution",
                "pattern_family": "evolved",
                "data_structures_used": ["list"],
                "operations_used": [gene.semantic_role for gene in genome.genes],
                "metta_derivation": [
                    item for gene in genome.genes 
                    for item in gene.metta_derivation
                ],
                "confidence": genome.overall_fitness,
                "final_score": genome.overall_fitness,
                "properties": ["semantically-evolved", "metta-guided", "tested"],
                "complexity_estimate": self._estimate_complexity(genome),
                "applicability_scope": "broad" if genome.overall_fitness > 0.8 else "medium",
                "generator_used": "SemanticEvolutionEngine",
                
                # Additional semantic evolution specific data
                "semantic_evolution_metadata": {
                    "genome_id": genome.genome_id,
                    "generation": genome.generation,
                    "parent_ids": genome.parent_ids,
                    "gene_count": len(genome.genes),
                    "semantic_roles": [gene.semantic_role for gene in genome.genes],
                    "correctness_score": genome.correctness_score,
                    "efficiency_score": genome.efficiency_score,
                    "maintainability_score": genome.maintainability_score,
                    "test_results": genome.test_results,
                    "semantic_validation": genome.validate_semantics()[0]
                },
                "metta_reasoning_trace": [
                    f"semantic-evolution-{genome.genome_id}",
                    f"generation-{genome.generation}",
                    f"fitness-{genome.overall_fitness:.3f}"
                ]
            }
            
            results.append(result)
        
        return results
    
    def _estimate_complexity(self, genome: SemanticGenome) -> str:
        """Estimate complexity based on genome structure"""
        complexity_metrics = genome.calculate_complexity()
        
        if complexity_metrics["loop_count"] > 1:
            return "higher"
        elif complexity_metrics["condition_count"] > 2:
            return "moderate"
        else:
            return "same"
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get comprehensive evolution summary"""
        if not self.best_genome_history:
            return {}
        
        best_ever = max(self.best_genome_history, key=lambda g: g.overall_fitness)
        final_best = self.best_genome_history[-1] if self.best_genome_history else None
        
        return {
            "total_generations": len(self.evolution_history),
            "population_size": self.population_size,
            "best_ever_fitness": best_ever.overall_fitness,
            "final_best_fitness": final_best.overall_fitness if final_best else 0,
            "fitness_improvement": (
                final_best.overall_fitness - self.best_genome_history[0].overall_fitness
                if len(self.best_genome_history) > 0 else 0
            ),
            "gene_pool_stats": self.semantic_gene_pool.get_stats(),
            "evaluator_stats": self.fitness_evaluator.get_evaluation_stats(),
            "test_summary": self.test_framework.get_test_summary(),
            "evolution_history": self.evolution_history,
            "best_genome_summary": best_ever.get_semantic_summary() if best_ever else {}
        }

def load_semantic_evolution_rules(reasoning_engine, rules_file_path: str) -> bool:
    """Load semantic evolution rules from file"""
    if not os.path.exists(rules_file_path):
        print(f"MeTTa rules file not found: {rules_file_path}")
        return False
    
    try:
        with open(rules_file_path, 'r') as f:
            content = f.read()
        
        loaded_count = 0
        failed_count = 0
        
        # Parse and load rules
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith(';'):
                continue
            
            # Load rule expressions
            if line.startswith('(=') and line.endswith(')'):
                try:
                    reasoning_engine._add_rule_safely(line)
                    loaded_count += 1
                except Exception as e:
                    print(f"Failed to load rule at line {line_num}: {e}")
                    failed_count += 1
        
        print(f"Loaded {loaded_count} semantic evolution rules ({failed_count} failed)")
        return loaded_count > 0
        
    except Exception as e:
        print(f"Error loading semantic evolution rules: {e}")
        return False

def demonstrate_semantic_evolution():
    """Demonstrate the semantic evolution system"""
    print("=== Semantic Evolution Demonstration ===")
    
    # Reference function
    def find_max_in_range(numbers, start_idx, end_idx):
        if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
            return None
        
        max_val = numbers[start_idx]
        for i in range(start_idx + 1, end_idx):
            if numbers[i] > max_val:
                max_val = numbers[i]
        
        return max_val
    
    # Create evolution engine
    engine = SemanticEvolutionEngine(
        population_size=15,
        max_generations=8
    )
    
    # Setup for the reference function
    engine.setup_for_function(find_max_in_range)
    
    # Define target semantics
    target_semantics = {
        "purpose": "maximize",
        "input_constraints": ["input_received", "valid_indices"],
        "output_spec": "maximum_element_in_range"
    }
    
    # Run evolution
    results = engine.evolve_solutions(target_semantics)
    
    print(f"\n=== Evolution Results ===")
    print(f"Generated {len(results)} solutions")
    
    # Show top 3 results
    for i, result in enumerate(results[:3], 1):
        metadata = result["semantic_evolution_metadata"]
        print(f"\n{i}. {result['name']}")
        print(f"   Overall Fitness: {result['final_score']:.3f}")
        print(f"   Correctness: {metadata['correctness_score']:.3f}")
        print(f"   Efficiency: {metadata['efficiency_score']:.3f}")
        print(f"   Generation: {metadata['generation']}")
        print(f"   Semantic Roles: {metadata['semantic_roles']}")
        print(f"   Test Results: {metadata['test_results'].get('passed', 0)}/{len(engine.test_framework.test_cases)} passed")
        
        # Show code preview
        code_lines = result['code'].split('\n')
        print(f"   Generated Code:")
        for line_num, line in enumerate(code_lines[:12], 1):
            print(f"     {line_num:2d}: {line}")
        if len(code_lines) > 12:
            print(f"     ... (truncated)")
    
    # Show evolution summary
    summary = engine.get_evolution_summary()
    print(f"\n=== Evolution Summary ===")
    print(f"Generations: {summary['total_generations']}")
    print(f"Best fitness achieved: {summary['best_ever_fitness']:.3f}")
    print(f"Fitness improvement: {summary['fitness_improvement']:.3f}")
    print(f"Gene pool: {summary['gene_pool_stats']['total_genes']} genes")
    print(f"Evaluations: {summary['evaluator_stats']['total_evaluations']}")
    
    return results

if __name__ == "__main__":
    demonstrate_semantic_evolution()