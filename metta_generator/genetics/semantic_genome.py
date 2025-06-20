#!/usr/bin/env python3
"""
Semantic Genome and Gene Classes for Enhanced Evolution
Provides semantic understanding of program components for better evolution
"""

import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

class SemanticGeneType(Enum):
    """Semantic gene types based on program semantics"""
    INITIALIZATION = "initialization"
    ITERATION = "iteration" 
    CONDITION = "condition"
    OPERATION = "operation"
    AGGREGATION = "aggregation"
    TERMINATION = "termination"
    ERROR_HANDLING = "error_handling"
    OPTIMIZATION = "optimization"

@dataclass
class SemanticGene:
    """Enhanced gene with semantic understanding"""
    gene_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    gene_type: SemanticGeneType = SemanticGeneType.OPERATION
    
    # Semantic properties
    semantic_role: str = ""  # What this gene does semantically
    preconditions: List[str] = field(default_factory=list)  # What must be true before
    postconditions: List[str] = field(default_factory=list)  # What becomes true after
    side_effects: List[str] = field(default_factory=list)   # What else changes
    
    # Code representation
    ast_pattern: str = ""           # Abstract syntax pattern
    code_template: str = ""         # Concrete code template
    parameter_slots: Dict[str, Any] = field(default_factory=dict)  # Fillable parameters
    
    # Evolution metrics
    fitness_history: List[float] = field(default_factory=list)
    usage_count: int = 0
    generation_born: int = 0
    success_rate: float = 0.5
    
    # MeTTa reasoning
    metta_derivation: List[str] = field(default_factory=list)
    reasoning_confidence: float = 0.5
    
    def add_fitness_record(self, fitness: float):
        """Add a fitness record for this gene"""
        self.fitness_history.append(fitness)
        # Update success rate as moving average
        if len(self.fitness_history) > 10:
            self.fitness_history = self.fitness_history[-10:]  # Keep last 10
        self.success_rate = sum(self.fitness_history) / len(self.fitness_history)
    
    def clone(self) -> 'SemanticGene':
        """Create a clone of this gene"""
        return SemanticGene(
            gene_type=self.gene_type,
            semantic_role=self.semantic_role,
            preconditions=self.preconditions.copy(),
            postconditions=self.postconditions.copy(),
            side_effects=self.side_effects.copy(),
            ast_pattern=self.ast_pattern,
            code_template=self.code_template,
            parameter_slots=self.parameter_slots.copy(),
            generation_born=self.generation_born,
            success_rate=self.success_rate,
            metta_derivation=self.metta_derivation.copy(),
            reasoning_confidence=self.reasoning_confidence
        )

@dataclass
class SemanticGenome:
    """Enhanced genome with semantic program representation"""
    genome_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    genes: List[SemanticGene] = field(default_factory=list)
    
    # Program semantics
    program_purpose: str = ""
    input_constraints: List[str] = field(default_factory=list)
    output_specification: str = ""
    algorithmic_invariants: List[str] = field(default_factory=list)
    
    # Evolution tracking
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    test_results: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    correctness_score: float = 0.0
    efficiency_score: float = 0.0
    maintainability_score: float = 0.0
    overall_fitness: float = 0.0
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    
    def add_gene(self, gene: SemanticGene, position: Optional[int] = None):
        """Add gene at specific position or append"""
        if position is None:
            self.genes.append(gene)
        else:
            self.genes.insert(position, gene)
        gene.usage_count += 1
    
    def remove_gene(self, gene: SemanticGene) -> bool:
        """Remove gene from genome"""
        try:
            self.genes.remove(gene)
            return True
        except ValueError:
            return False
    
    def get_genes_by_type(self, gene_type: SemanticGeneType) -> List[SemanticGene]:
        """Get genes of specific semantic type"""
        return [gene for gene in self.genes if gene.gene_type == gene_type]
    
    def get_genes_by_role(self, role: str) -> List[SemanticGene]:
        """Get genes with specific semantic role"""
        return [gene for gene in self.genes if role in gene.semantic_role]
    
    def validate_semantics(self) -> Tuple[bool, List[str]]:
        """Validate genome semantic consistency"""
        errors = []
        
        # Check for required gene types
        required_types = [SemanticGeneType.INITIALIZATION, SemanticGeneType.TERMINATION]
        for req_type in required_types:
            if not self.get_genes_by_type(req_type):
                errors.append(f"Missing required gene type: {req_type}")
        
        # Check semantic dependencies
        for i, gene in enumerate(self.genes):
            for precondition in gene.preconditions:
                if not self._check_precondition_satisfied(precondition, i):
                    errors.append(f"Precondition '{precondition}' not satisfied for gene {gene.gene_id}")
        
        # Check for semantic conflicts
        conflicts = self._check_semantic_conflicts()
        errors.extend(conflicts)
        
        return len(errors) == 0, errors
    
    def _check_precondition_satisfied(self, precondition: str, gene_index: int) -> bool:
        """Check if precondition is satisfied by earlier genes"""
        # Check if any previous gene provides this postcondition
        for i in range(gene_index):
            if precondition in self.genes[i].postconditions:
                return True
        
        # Check if it's an input constraint
        return precondition in self.input_constraints
    
    def _check_semantic_conflicts(self) -> List[str]:
        """Check for semantic conflicts between genes"""
        conflicts = []
        
        # Check for conflicting operations
        operation_genes = self.get_genes_by_type(SemanticGeneType.OPERATION)
        for i, gene1 in enumerate(operation_genes):
            for gene2 in operation_genes[i+1:]:
                if self._genes_conflict(gene1, gene2):
                    conflicts.append(f"Conflict between {gene1.semantic_role} and {gene2.semantic_role}")
        
        return conflicts
    
    def _genes_conflict(self, gene1: SemanticGene, gene2: SemanticGene) -> bool:
        """Check if two genes have conflicting semantics"""
        # Simple conflict detection
        conflicting_pairs = [
            ("maximize", "minimize"),
            ("forward_scan", "backward_scan"),
            ("increment", "decrement")
        ]
        
        for conflict1, conflict2 in conflicting_pairs:
            if (conflict1 in gene1.semantic_role and conflict2 in gene2.semantic_role) or \
               (conflict2 in gene1.semantic_role and conflict1 in gene2.semantic_role):
                return True
        
        return False
    
    def calculate_complexity(self) -> Dict[str, int]:
        """Calculate complexity metrics"""
        return {
            "gene_count": len(self.genes),
            "unique_types": len(set(gene.gene_type for gene in self.genes)),
            "condition_count": len(self.get_genes_by_type(SemanticGeneType.CONDITION)),
            "loop_count": len(self.get_genes_by_type(SemanticGeneType.ITERATION)),
            "operation_count": len(self.get_genes_by_type(SemanticGeneType.OPERATION))
        }
    
    def get_semantic_summary(self) -> Dict[str, Any]:
        """Get semantic summary of genome"""
        return {
            "genome_id": self.genome_id,
            "purpose": self.program_purpose,
            "gene_count": len(self.genes),
            "gene_types": [gene.gene_type.value for gene in self.genes],
            "semantic_roles": [gene.semantic_role for gene in self.genes],
            "fitness_scores": self.fitness_scores,
            "overall_fitness": self.overall_fitness,
            "generation": self.generation,
            "complexity": self.calculate_complexity()
        }
    
    def clone(self) -> 'SemanticGenome':
        """Create a clone of this genome"""
        return SemanticGenome(
            genes=[gene.clone() for gene in self.genes],
            program_purpose=self.program_purpose,
            input_constraints=self.input_constraints.copy(),
            output_specification=self.output_specification,
            algorithmic_invariants=self.algorithmic_invariants.copy(),
            generation=self.generation,
            parent_ids=self.parent_ids.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary for serialization"""
        return {
            "genome_id": self.genome_id,
            "genes": [
                {
                    "gene_id": gene.gene_id,
                    "gene_type": gene.gene_type.value,
                    "semantic_role": gene.semantic_role,
                    "preconditions": gene.preconditions,
                    "postconditions": gene.postconditions,
                    "code_template": gene.code_template,
                    "parameter_slots": gene.parameter_slots,
                    "success_rate": gene.success_rate,
                    "generation_born": gene.generation_born,
                    "metta_derivation": gene.metta_derivation
                }
                for gene in self.genes
            ],
            "program_purpose": self.program_purpose,
            "input_constraints": self.input_constraints,
            "output_specification": self.output_specification,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "fitness_scores": self.fitness_scores,
            "correctness_score": self.correctness_score,
            "efficiency_score": self.efficiency_score,
            "maintainability_score": self.maintainability_score,
            "overall_fitness": self.overall_fitness,
            "created_at": self.created_at
        }

class SemanticGenePool:
    """Pool of semantic genes for evolution"""
    
    def __init__(self):
        self.genes_by_type: Dict[SemanticGeneType, List[SemanticGene]] = {
            gene_type: [] for gene_type in SemanticGeneType
        }
        self.successful_genes: List[SemanticGene] = []
    
    def add_gene(self, gene: SemanticGene):
        """Add gene to appropriate pool"""
        self.genes_by_type[gene.gene_type].append(gene)
    
    def get_genes_by_type(self, gene_type: SemanticGeneType) -> List[SemanticGene]:
        """Get all genes of specific type"""
        return self.genes_by_type[gene_type]
    
    def get_best_genes_by_type(self, gene_type: SemanticGeneType, count: int = 5) -> List[SemanticGene]:
        """Get best performing genes of specific type"""
        genes = self.genes_by_type[gene_type]
        return sorted(genes, key=lambda g: g.success_rate, reverse=True)[:count]
    
    def mark_gene_successful(self, gene: SemanticGene, fitness: float):
        """Mark gene as successful with fitness score"""
        gene.add_fitness_record(fitness)
        if gene not in self.successful_genes and gene.success_rate > 0.7:
            self.successful_genes.append(gene)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gene pool statistics"""
        total_genes = sum(len(genes) for genes in self.genes_by_type.values())
        return {
            "total_genes": total_genes,
            "successful_genes": len(self.successful_genes),
            "genes_by_type": {
                gene_type.value: len(genes) 
                for gene_type, genes in self.genes_by_type.items()
            },
            "avg_success_rate": sum(
                gene.success_rate 
                for genes in self.genes_by_type.values() 
                for gene in genes
            ) / max(1, total_genes)
        }