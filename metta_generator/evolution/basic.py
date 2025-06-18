#!/usr/bin/env python3
"""
Basic evolutionary operations for MeTTa donor generation
Minimal implementation to test the evolutionary approach
"""

import random
import ast
from typing import List, Dict, Any, Optional, Tuple
from metta_generator.genetics.genome import SimpleCodeGenome, MeTTaGene, BasicGenePool

class BasicEvolutionEngine:
    """Simplified evolutionary engine to test the approach"""
    
    def __init__(self, metta_space=None, reasoning_engine=None, population_size=10, max_generations=5):
        self.metta_space = metta_space
        self.reasoning_engine = reasoning_engine  # Add this line
        self.population_size = population_size
        self.max_generations = max_generations
        self.gene_pool = BasicGenePool()
        self.population: List[SimpleCodeGenome] = []
        self.generation = 0
        self.evolution_history = []
        
        # Initialize with basic patterns
        self._initialize_basic_gene_pool()
        
        # Add MeTTa-guided enhancements if reasoning engine available
        if self.reasoning_engine:
            print("MeTTa reasoning engine available - enabling MeTTa-guided evolution")
            self._load_metta_evolution_rules()
            self._mine_additional_genes_from_metta()
        else:
            print("No MeTTa reasoning engine - using basic evolution only")
    
    def _load_metta_evolution_rules(self):
        """Load MeTTa rules for genetic operations"""
        metta_rules = [
            # Gene compatibility rules
            "(= (gene-compatible loop conditional) True)",
            "(= (gene-compatible operation structure) True)", 
            "(= (gene-compatible loop loop) False)",
            
            # Fitness contribution rules
            "(= (fitness-weight loop) 0.3)",
            "(= (fitness-weight conditional) 0.2)", 
            "(= (fitness-weight operation) 0.2)",
            "(= (fitness-weight structure) 0.3)",
            
            # Selection pressure rules
            "(= (selection-pressure high) 0.8)",
            "(= (selection-pressure medium) 0.6)",
            "(= (selection-pressure low) 0.4)"
        ]
        
        for rule in metta_rules:
            try:
                self.reasoning_engine._add_rule_safely(rule)
            except Exception as e:
                print(f"Failed to load MeTTa rule: {e}")
        
        print(f"Loaded {len(metta_rules)} MeTTa evolution rules")
    
    def _initialize_basic_gene_pool(self):
        """Initialize gene pool with basic code patterns"""
        basic_genes = [
            # Loop patterns
            MeTTaGene('loop', 'for i in range(len(data)):', '(loop-pattern for range)', 0.6),
            MeTTaGene('loop', 'for item in data:', '(loop-pattern for item)', 0.6),
            MeTTaGene('loop', 'while condition:', '(loop-pattern while)', 0.5),
            
            # Conditional patterns
            MeTTaGene('conditional', 'if condition:', '(conditional-pattern if)', 0.6),
            MeTTaGene('conditional', 'if x > y:', '(conditional-pattern comparison)', 0.6),
            MeTTaGene('conditional', 'if data is None:', '(conditional-pattern null-check)', 0.7),
            
            # Operation patterns
            MeTTaGene('operation', 'max(data)', '(operation-pattern max)', 0.6),
            MeTTaGene('operation', 'min(data)', '(operation-pattern min)', 0.6),
            MeTTaGene('operation', 'sum(data)', '(operation-pattern sum)', 0.6),
            MeTTaGene('operation', 'len(data)', '(operation-pattern len)', 0.7),
            
            # Structure patterns
            MeTTaGene('structure', 'return result', '(structure-pattern return)', 0.8),
            MeTTaGene('structure', 'result = None', '(structure-pattern init-none)', 0.6),
            MeTTaGene('structure', 'result = []', '(structure-pattern init-list)', 0.6),
        ]
        
        for gene in basic_genes:
            self.gene_pool.add_gene(gene)
    
    def _mine_additional_genes_from_metta(self):
        """Mine genes from existing MeTTa atoms"""
        print("Mining additional genes from MeTTa atoms...")
    
        # Try to extract patterns from MeTTa space
        if hasattr(self.metta_space, 'get_atoms'):
            try:
                atoms = self.metta_space.get_atoms()
                mined_count = 0
                
                for atom in atoms[:20]:  # Limit to first 20 for testing
                    atom_str = str(atom)
                    
                    if 'loop-pattern' in atom_str:
                        gene = MeTTaGene(
                            'loop',
                            'for i in metta_range:',
                            f'(metta-mined {atom_str})',
                            0.7,  # Higher fitness for MeTTa-mined genes
                            generation_born=0
                        )
                        self.gene_pool.add_gene(gene)
                        mined_count += 1
                    
                    elif 'conditional-statement' in atom_str:
                        gene = MeTTaGene(
                            'conditional',
                            'if metta_condition:',
                            f'(metta-mined {atom_str})',
                            0.7,
                            generation_born=0
                        )
                        self.gene_pool.add_gene(gene)
                        mined_count += 1
                
                print(f"Mined {mined_count} additional genes from MeTTa atoms")
                
            except Exception as e:
                print(f"MeTTa atom mining failed: {e}")
    
    def metta_guided_crossover(self, parent1: SimpleCodeGenome, parent2: SimpleCodeGenome) -> SimpleCodeGenome:
        """Enhanced crossover using MeTTa reasoning if available"""
        
        if self.reasoning_engine:
            print(f"Using MeTTa-guided crossover")
            
            # Query MeTTa for gene compatibility
            offspring_genes = []
            
            for gene1 in parent1.genes:
                compatible_found = False
                
                for gene2 in parent2.genes:
                    # Check MeTTa compatibility
                    compatibility_query = f"(gene-compatible {gene1.pattern_type} {gene2.pattern_type})"
                    
                    try:
                        # Simple MeTTa query
                        query_result = f"(match &self {compatibility_query} True)"
                        results = self.reasoning_engine._execute_metta_reasoning(query_result, [])
                        
                        if results:  # Compatible according to MeTTa
                            # Choose better gene
                            chosen_gene = gene1 if gene1.fitness_score > gene2.fitness_score else gene2
                            offspring_genes.append(chosen_gene)
                            print(f"MeTTa compatible: {gene1.pattern_type} + {gene2.pattern_type} → {chosen_gene.pattern_type}")
                            compatible_found = True
                            break
                            
                    except Exception as e:
                        print(f"MeTTa query failed: {e}")
                
                if not compatible_found:
                    offspring_genes.append(gene1)  # Keep original if no compatible match
            
            if offspring_genes:
                return SimpleCodeGenome(
                    genes=offspring_genes,
                    generation=max(parent1.generation, parent2.generation) + 1,
                    parent_ids=[parent1.genome_id, parent2.genome_id]
                )
        
        # Fallback to original crossover if MeTTa reasoning fails
        print(f"Falling back to basic crossover")
        return self.simple_crossover(parent1, parent2)
    
    def metta_enhanced_fitness(self, genome: SimpleCodeGenome, original_function: str) -> float:
        """Enhanced fitness using MeTTa reasoning if available"""
        
        base_fitness = self.basic_fitness_function(genome, original_function)
        
        if self.reasoning_engine:
            print(f"     🧠 Enhancing fitness with MeTTa reasoning")
            
            metta_bonus = 0.0
            
            # Query MeTTa for pattern weights
            for gene in genome.genes:
                weight_query = f"(fitness-weight {gene.pattern_type})"
                
                try:
                    query_result = f"(match &self {weight_query} $weight)"
                    results = self.reasoning_engine._execute_metta_reasoning(query_result, [])
                    
                    if results:
                        # Add MeTTa-derived bonus
                        metta_bonus += 0.1 * gene.fitness_score
                        print(f"       + MeTTa bonus for {gene.pattern_type}: 0.1")
                        
                except Exception as e:
                    print(f"       ⚠ MeTTa fitness query failed: {e}")
            
            enhanced_fitness = min(1.0, base_fitness + metta_bonus)
            print(f"       📊 Enhanced fitness: {base_fitness:.3f} + {metta_bonus:.3f} = {enhanced_fitness:.3f}")
            
            return enhanced_fitness
        
        return base_fitness
    
    def extract_genes_from_function(self, function_code: str, function_name: str) -> List[MeTTaGene]:
        """Extract genetic material from a function using basic AST analysis"""
        genes = []
        
        try:
            tree = ast.parse(function_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    # Extract for loop pattern
                    if hasattr(node.iter, 'func') and hasattr(node.iter.func, 'id'):
                        if node.iter.func.id == 'range':
                            gene = MeTTaGene(
                                'loop', 
                                'for i in range(n):',
                                f'(loop-pattern for range {function_name})',
                                0.5,
                                generation_born=0
                            )
                            genes.append(gene)
                    else:
                        gene = MeTTaGene(
                            'loop',
                            'for item in data:',
                            f'(loop-pattern for item {function_name})',
                            0.5,
                            generation_born=0
                        )
                        genes.append(gene)
                
                elif isinstance(node, ast.If):
                    # Extract conditional pattern
                    gene = MeTTaGene(
                        'conditional',
                        'if condition:',
                        f'(conditional-pattern if {function_name})',
                        0.5,
                        generation_born=0
                    )
                    genes.append(gene)
                
                elif isinstance(node, ast.Call):
                    # Extract function call patterns
                    if hasattr(node.func, 'id'):
                        func_name = node.func.id
                        if func_name in ['max', 'min', 'sum', 'len']:
                            gene = MeTTaGene(
                                'operation',
                                f'{func_name}(data)',
                                f'(operation-pattern {func_name} {function_name})',
                                0.5,
                                generation_born=0
                            )
                            genes.append(gene)
                
                elif isinstance(node, ast.Return):
                    # Extract return pattern
                    gene = MeTTaGene(
                        'structure',
                        'return result',
                        f'(structure-pattern return {function_name})',
                        0.6,
                        generation_born=0
                    )
                    genes.append(gene)
        
        except Exception as e:
            print(f"Error extracting genes from {function_name}: {e}")
            # Add fallback gene
            genes.append(MeTTaGene(
                'structure',
                '# Basic function structure',
                f'(generic-pattern {function_name})',
                0.3,
                generation_born=0
            ))
        
        return genes
    
    def create_initial_genome(self, function_code: str, function_name: str) -> SimpleCodeGenome:
        """Create initial genome from function code"""
        genes = self.extract_genes_from_function(function_code, function_name)
        
        # Add some random genes from gene pool for variation
        for pattern_type in ['loop', 'conditional', 'operation']:
            if random.random() < 0.3:  # 30% chance to add random gene
                random_gene = self.gene_pool.get_random_gene(pattern_type)
                if random_gene:
                    genes.append(random_gene)
        
        genome = SimpleCodeGenome(
            genes=genes,
            generation=0,
            parent_ids=[]
        )
        
        return genome
    
    def simple_crossover(self, parent1: SimpleCodeGenome, parent2: SimpleCodeGenome) -> SimpleCodeGenome:
        """Basic crossover operation"""
        # Simple gene swapping
        all_genes = parent1.genes + parent2.genes
        
        # Randomly select genes, preferring those with higher fitness
        offspring_genes = []
        target_size = min(len(parent1.genes), len(parent2.genes))
        
        for _ in range(target_size):
            if all_genes:
                # Weighted selection based on fitness
                weights = [gene.fitness_score for gene in all_genes]
                if sum(weights) > 0:
                    selected_gene = random.choices(all_genes, weights=weights)[0]
                else:
                    selected_gene = random.choice(all_genes)
                
                offspring_genes.append(selected_gene)
                all_genes.remove(selected_gene)
        
        offspring = SimpleCodeGenome(
            genes=offspring_genes,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.genome_id, parent2.genome_id]
        )
        
        return offspring
    
    def simple_mutation(self, genome: SimpleCodeGenome) -> SimpleCodeGenome:
        """Basic mutation operation"""
        mutation_rate = 0.2  # 20% chance to mutate each gene
        
        mutated_genes = []
        for gene in genome.genes:
            if random.random() < mutation_rate:
                # Replace with similar gene from pool
                similar_gene = self.gene_pool.get_random_gene(gene.pattern_type)
                if similar_gene and similar_gene != gene:
                    new_gene = MeTTaGene(
                        similar_gene.pattern_type,
                        similar_gene.code_fragment,
                        f'mutated-{similar_gene.metta_atom}',
                        similar_gene.fitness_score,
                        generation_born=genome.generation + 1
                    )
                    mutated_genes.append(new_gene)
                else:
                    mutated_genes.append(gene)
            else:
                mutated_genes.append(gene)
        
        mutated_genome = SimpleCodeGenome(
            genes=mutated_genes,
            generation=genome.generation + 1,
            parent_ids=[genome.genome_id]
        )
        
        return mutated_genome
    
    def basic_fitness_function(self, genome: SimpleCodeGenome, original_function: str) -> float:
        """Basic fitness evaluation"""
        fitness = 0.0
        
        # 1. Gene quality (40% weight)
        if genome.genes:
            avg_gene_fitness = sum(gene.fitness_score for gene in genome.genes) / len(genome.genes)
            fitness += 0.4 * avg_gene_fitness
        
        # 2. Diversity bonus (20% weight)
        unique_types = len(set(gene.pattern_type for gene in genome.genes))
        diversity_score = min(1.0, unique_types / 4)  # Normalize to max 4 types
        fitness += 0.2 * diversity_score
        
        # 3. Size appropriateness (20% weight)
        ideal_size = 5  # Ideal number of genes
        size_score = max(0.0, 1.0 - abs(len(genome.genes) - ideal_size) / ideal_size)
        fitness += 0.2 * size_score
        
        # 4. Pattern completeness (20% weight)
        required_patterns = ['structure', 'loop', 'conditional']
        present_patterns = set(gene.pattern_type for gene in genome.genes)
        completeness = len(present_patterns.intersection(required_patterns)) / len(required_patterns)
        fitness += 0.2 * completeness
        
        return min(1.0, fitness)
    
    def tournament_selection(self, population: List[SimpleCodeGenome], tournament_size: int = 3) -> SimpleCodeGenome:
        """Tournament selection for choosing parents"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda g: g.fitness_score)
    
    def evolve_population(self, original_function: str, function_name: str) -> List[SimpleCodeGenome]:
        """Main evolution loop"""
        print(f"\nStarting basic evolution for {function_name}")
        
        # Initialize population
        print(f"Creating initial population of {self.population_size} individuals...")
        self.population = []
        for i in range(self.population_size):
            genome = self.create_initial_genome(original_function, function_name)
            genome.fitness_score = self.basic_fitness_function(genome, original_function)
            self.population.append(genome)
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.generation = generation
            print(f"\nGeneration {generation}")
            
            # Show current best
            best_genome = max(self.population, key=lambda g: g.fitness_score)
            avg_fitness = sum(g.fitness_score for g in self.population) / len(self.population)
            print(f"   Best fitness: {best_genome.fitness_score:.3f}")
            print(f"   Average fitness: {avg_fitness:.3f}")
            print(f"   Best genome genes: {len(best_genome.genes)} ({[g.pattern_type for g in best_genome.genes]})")
            
            # Create new generation
            new_population = []
            
            # Keep best individual (elitism)
            new_population.append(best_genome)
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(self.population)
                parent2 = self.tournament_selection(self.population)
                
                # Crossover
                offspring = self.metta_guided_crossover(parent1, parent2)
                
                # Mutation
                if random.random() < 0.8:  # 80% chance of mutation
                    offspring = self.simple_mutation(offspring)
                
                # Evaluate fitness
                offspring.fitness_score = self.metta_enhanced_fitness(offspring, original_function)
                
                new_population.append(offspring)
            
            self.population = new_population[:self.population_size]
            
            # Track evolution
            generation_stats = {
                'generation': generation,
                'best_fitness': best_genome.fitness_score,
                'avg_fitness': avg_fitness,
                'gene_pool_stats': self.gene_pool.get_stats()
            }
            self.evolution_history.append(generation_stats)
        
        # Final results
        final_best = max(self.population, key=lambda g: g.fitness_score)
        print(f"\nEvolution complete!")
        print(f"   Final best fitness: {final_best.fitness_score:.3f}")
        print(f"   Generation: {final_best.generation}")
        print(f"   Genes: {[(g.pattern_type, g.code_fragment) for g in final_best.genes]}")
        
        return sorted(self.population, key=lambda g: g.fitness_score, reverse=True)
    
    def genome_to_code(self, genome: SimpleCodeGenome, function_name: str) -> str:
        """Convert genome back to executable code (basic implementation)"""
        code_parts = []
        
        # Function definition
        code_parts.append(f"def {function_name}_evolved(data, start_idx=None, end_idx=None):")
        code_parts.append('    """Evolved function using basic MeTTa genetic algorithm"""')
        
        # Add error checking
        code_parts.append('    if not data:')
        code_parts.append('        return None')
        
        # Process genes by type
        structure_genes = genome.get_genes_by_type('structure')
        loop_genes = genome.get_genes_by_type('loop')
        conditional_genes = genome.get_genes_by_type('conditional')
        operation_genes = genome.get_genes_by_type('operation')
        
        # Add initialization
        if any('init' in gene.code_fragment for gene in structure_genes):
            code_parts.append('    result = None')
        else:
            code_parts.append('    result = data[0] if data else None')
        
        # Add main logic
        if loop_genes:
            loop_gene = loop_genes[0]
            if 'range' in loop_gene.code_fragment:
                code_parts.append('    for i in range(len(data)):')
                if operation_genes:
                    op_gene = operation_genes[0]
                    if 'max' in op_gene.code_fragment:
                        code_parts.append('        if data[i] > result:')
                        code_parts.append('            result = data[i]')
                    elif 'min' in op_gene.code_fragment:
                        code_parts.append('        if result is None or data[i] < result:')
                        code_parts.append('            result = data[i]')
                else:
                    code_parts.append('        result = data[i]')
            else:
                code_parts.append('    for item in data:')
                code_parts.append('        result = item')
        
        # Add return
        code_parts.append('    return result')
        
        return '\n'.join(code_parts)