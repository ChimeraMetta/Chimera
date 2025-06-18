#!/usr/bin/env python3
"""
Basic genetic structures for MeTTa evolutionary donor generation
Start with minimal viable implementation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid
import time

@dataclass
class MeTTaGene:
    """Basic genetic component derived from MeTTa patterns"""
    pattern_type: str  # 'loop', 'conditional', 'operation', etc.
    code_fragment: str  # Actual code pattern
    metta_atom: str  # Original MeTTa atom representation
    fitness_score: float = 0.5  # How successful this gene has been
    generation_born: int = 0
    usage_count: int = 0
    
    def __post_init__(self):
        self.gene_id = str(uuid.uuid4())[:8]

@dataclass 
class SimpleCodeGenome:
    """Simplified genetic representation of a function"""
    genes: List[MeTTaGene] = field(default_factory=list)
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.genome_id = str(uuid.uuid4())[:8]
        self.created_at = time.time()
    
    def add_gene(self, gene: MeTTaGene):
        """Add a gene to this genome"""
        self.genes.append(gene)
        gene.usage_count += 1
    
    def get_genes_by_type(self, pattern_type: str) -> List[MeTTaGene]:
        """Get all genes of a specific pattern type"""
        return [gene for gene in self.genes if gene.pattern_type == pattern_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'genome_id': self.genome_id,
            'fitness_score': self.fitness_score,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'created_at': self.created_at,
            'genes': [
                {
                    'gene_id': gene.gene_id,
                    'pattern_type': gene.pattern_type,
                    'code_fragment': gene.code_fragment,
                    'metta_atom': gene.metta_atom,
                    'fitness_score': gene.fitness_score,
                    'generation_born': gene.generation_born,
                    'usage_count': gene.usage_count
                }
                for gene in self.genes
            ]
        }

class BasicGenePool:
    """Basic gene pool for storing and managing genetic material"""
    
    def __init__(self):
        self.genes_by_type: Dict[str, List[MeTTaGene]] = {
            'loop': [],
            'conditional': [],
            'operation': [],
            'structure': [],
            'function_call': []
        }
        self.successful_genes: List[MeTTaGene] = []
    
    def add_gene(self, gene: MeTTaGene):
        """Add a gene to the appropriate category"""
        if gene.pattern_type in self.genes_by_type:
            self.genes_by_type[gene.pattern_type].append(gene)
        else:
            # Create new category if needed
            self.genes_by_type[gene.pattern_type] = [gene]
    
    def get_genes_by_type(self, pattern_type: str) -> List[MeTTaGene]:
        """Get all genes of a specific type"""
        return self.genes_by_type.get(pattern_type, [])
    
    def get_random_gene(self, pattern_type: str) -> Optional[MeTTaGene]:
        """Get a random gene of specified type"""
        import random
        genes = self.get_genes_by_type(pattern_type)
        return random.choice(genes) if genes else None
    
    def mark_gene_successful(self, gene: MeTTaGene):
        """Mark a gene as successful to increase its selection probability"""
        gene.fitness_score = min(1.0, gene.fitness_score + 0.1)
        if gene not in self.successful_genes:
            self.successful_genes.append(gene)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gene pool statistics"""
        total_genes = sum(len(genes) for genes in self.genes_by_type.values())
        return {
            'total_genes': total_genes,
            'successful_genes': len(self.successful_genes),
            'categories': {cat: len(genes) for cat, genes in self.genes_by_type.items()},
            'avg_fitness': sum(gene.fitness_score for genes in self.genes_by_type.values() 
                              for gene in genes) / max(1, total_genes)
        }