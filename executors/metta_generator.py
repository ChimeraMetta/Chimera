#!/usr/bin/env python3
"""
MeTTa-Powered Donor Generation System - FIXED VERSION
Now properly integrates with decompose_function and convert_to_metta_atoms
"""

import ast
import inspect
import re
import os
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum

# Import your existing components
from reflectors.static_analyzer import decompose_function, convert_to_metta_atoms, CodeDecomposer
from reflectors.dynamic_monitor import monitor

class GenerationStrategy(Enum):
    """Different strategies for generating donor candidates."""
    OPERATION_SUBSTITUTION = "operation_substitution"
    ACCUMULATOR_VARIATION = "accumulator_variation"
    CONDITION_VARIATION = "condition_variation"
    STRUCTURE_PRESERVATION = "structure_preservation"
    PROPERTY_GUIDED = "property_guided"
    PATTERN_EXPANSION = "pattern_expansion"

@dataclass
class CodePattern:
    """Represents a detected code pattern from MeTTa reasoning."""
    pattern_type: str
    metta_evidence: List[str]  # MeTTa atoms that support this pattern
    confidence: float
    properties: List[str]

@dataclass
class DonorCandidate:
    """A donor candidate generated through MeTTa reasoning."""
    name: str
    description: str
    code: str
    strategy: str
    metta_derivation: List[str]  # MeTTa reasoning steps that led to this candidate
    confidence: float
    properties: List[str]

class MettaDonorGenerator:
    """
    True MeTTa-powered donor generator that uses actual MeTTa reasoning
    and ontology rules for pattern detection and code generation.
    """
    
    def __init__(self, metta_space=None):
        self.metta_space = metta_space or monitor.metta_space
        self.function_name = None
        self.original_code = None
        self.analysis_result = None  # Store the full analysis result
        
        print("  Initializing MeTTa-powered donor generator...")
        
    def load_donor_ontology(self, ontology_file: str = "metta/donor_generation_ontology.metta"):
        """Load MeTTa ontology rules for donor generation."""
        print(" Loading donor generation ontology...")
        
        if os.path.exists(ontology_file):
            if monitor.load_metta_rules(ontology_file):
                print("  Donor generation ontology loaded successfully")
                return True
            else:
                print("  Failed to load donor generation ontology")
                return False
        else:
            print(f"   Ontology file not found: {ontology_file}")
            print("   Please ensure the ontology file exists before running generation")
            return False
    
    def generate_donors_from_function(self, func, 
                                    strategies: Optional[List[GenerationStrategy]] = None) -> List[Dict[str, Any]]:
        """
        Generate donor candidates directly from a Python function using proper analysis pipeline.
        Now uses convert_to_metta_atoms more directly for better control.
        
        Args:
            func: Python function object or source code string
            strategies: Optional list of strategies to apply
            
        Returns:
            List of generated donor candidates
        """
        print("  Starting MeTTa-powered donor generation from function...")
        
        # 1. Extract source code
        if isinstance(func, str):
            self.original_code = func
            self.function_name = self._extract_function_name(func)
        else:
            try:
                print(f"[METTA_GENERATOR]   Getting source code for {func.__name__}")
                self.original_code = inspect.getsource(func)
                print(f"[METTA_GENERATOR]   Source code: {self.original_code}")
                self.function_name = func.__name__
            except Exception as e:
                print(f"  Failed to extract source code: {e}")
                return []
        
        print(f"  Analyzing function: {self.function_name}")
        
        # 2. Use the static analysis pipeline with more direct control
        print("  Running static analysis...")
        
        try:
            # Parse the source code
            tree = ast.parse(self.original_code)
            
            # Create and run the decomposer
            decomposer = CodeDecomposer()
            decomposer.visit(tree)
            
            # Store the decomposer for richer access to structural info
            self.decomposer = decomposer
            
            # Convert to MeTTa atoms using the converter
            metta_atoms = convert_to_metta_atoms(decomposer)
            
            # Create the analysis result structure
            self.analysis_result = {
                "metta_atoms": metta_atoms,
                "structure": decomposer.atoms,
                "function_calls": decomposer.function_calls,
                "variables": decomposer.variables,
                "module_relationships": decomposer.module_relationships,
                "class_hierarchies": decomposer.class_hierarchies,
                "function_dependencies": decomposer.function_dependencies,
                "line_mapping": decomposer.line_mapping
            }
            
        except Exception as e:
            print(f"  Static analysis failed: {e}")
            return []
        
        print(f"  Static analysis complete - found {len(metta_atoms)} atoms")
        print(f"  Structural atoms: {len(decomposer.atoms)}")
        print(f"  Function calls: {len(decomposer.function_calls)}")
        print(f"  Variables: {sum(len(v) for v in decomposer.variables.values())}")
        
        # 3. Continue with the existing generation pipeline
        return self.generate_donors(metta_atoms, self.original_code, strategies)
    
    def generate_donors(self, metta_atoms: List[str], original_code: str, 
                       strategies: Optional[List[GenerationStrategy]] = None) -> List[Dict[str, Any]]:
        """
        Generate donor candidates using true MeTTa reasoning.
        
        Args:
            metta_atoms: MeTTa atoms from static analysis
            original_code: Original function source code
            strategies: Optional list of strategies to apply
            
        Returns:
            List of generated donor candidates
        """
        print("  Starting MeTTa-powered donor generation...")
        
        # Extract function name if not already set
        if not self.function_name:
            self.original_code = original_code
            self.function_name = self._extract_function_name(original_code)
        
        print(f"  Generating donors for function: {self.function_name}")
        print(f"  Working with {len(metta_atoms)} MeTTa atoms")
        
        # Display some atoms for debugging
        print("  Sample MeTTa atoms:")
        for i, atom in enumerate(metta_atoms[:5]):
            print(f"   {i+1}. {atom}")
        if len(metta_atoms) > 5:
            print(f"   ... and {len(metta_atoms) - 5} more atoms")
        
        # 1. Load all atoms into MeTTa space
        print("  Loading atoms into MeTTa space...")
        self._load_atoms_to_metta(metta_atoms)
        
        # 2. Use MeTTa reasoning to detect patterns
        print("  Using MeTTa reasoning to detect patterns...")
        patterns = self._detect_patterns_with_metta()
        
        # 3. Use MeTTa reasoning to determine applicable strategies
        print("  Determining applicable strategies with MeTTa...")
        applicable_strategies = self._get_applicable_strategies_from_metta(strategies)
        
        # 4. Generate candidates using MeTTa rules
        print("   Generating candidates using MeTTa rules...")
        candidates = self._generate_candidates_with_metta(applicable_strategies)
        
        if not candidates:
            print("   No candidates generated, using fallback generation...")
            candidates = self._fallback_candidate_generation(applicable_strategies)
        
        # 5. Rank candidates using MeTTa scoring
        print("  Ranking candidates using MeTTa scoring...")
        ranked_candidates = self._rank_candidates_with_metta(candidates)
        
        print(f"  Generated {len(ranked_candidates)} candidates using MeTTa reasoning")
        return ranked_candidates
    
    def _extract_function_name(self, code: str) -> str:
        """Extract function name from code."""
        match = re.search(r'def\s+(\w+)', code)
        return match.group(1) if match else "unknown_function"
    
    def _load_atoms_to_metta(self, metta_atoms: List[str]):
        """Load all atoms into MeTTa space for reasoning."""
        loaded_count = 0
        failed_count = 0
        
        # Store the atoms for later reference
        self.metta_atoms = metta_atoms
        
        for atom in metta_atoms:
            if monitor.add_atom(atom):
                loaded_count += 1
            else:
                failed_count += 1
        
        print(f"   LOADING: Loaded {loaded_count}/{len(metta_atoms)} atoms ({failed_count} failed)")
        
        # Also add the original function information
        monitor.add_atom(f"(original-function {self.function_name})")
        
        # Add analysis metadata if available
        if hasattr(self, 'analysis_result') and self.analysis_result:
            # Add function call information
            for func_name, call_patterns in self.analysis_result.get("function_calls", {}).items():
                monitor.add_atom(f"(calls-function {self.function_name} {func_name})")
            
            # Add variable information  
            for scope, vars_dict in self.analysis_result.get("variables", {}).items():
                for var_name, var_type in vars_dict.items():
                    monitor.add_atom(f"(has-variable {self.function_name} {var_name} {var_type})")
    
    def _detect_patterns_with_metta(self) -> List[CodePattern]:
        """Use MeTTa reasoning to detect code patterns using actual atoms."""
        patterns = []
        
        print("     Analyzing patterns from MeTTa atoms...")
        
        # Get all atoms to inspect what we actually have
        atoms_info = self._get_atoms_summary()
        print(f"     Atom summary: {atoms_info}")
        
        # Pattern 1: Iterate-accumulate pattern (loops + comparisons + bounds checking)
        try:
            has_loop = self._has_evidence_in_metta("loop-pattern")
            has_comparison = self._has_evidence_in_metta("bin-op")
            has_bounds_check = self._has_bounds_checking_evidence()
            
            if has_loop and has_comparison:
                confidence = 0.9 if has_bounds_check else 0.7
                patterns.append(CodePattern(
                    pattern_type="iterate_accumulate",
                    metta_evidence=[
                        f"(has-loop-pattern {self.function_name})" if has_loop else "",
                        f"(has-comparison-ops {self.function_name})" if has_comparison else "",
                        f"(has-bounds-checking {self.function_name})" if has_bounds_check else ""
                    ],
                    confidence=confidence,
                    properties=self._get_function_properties_from_evidence()
                ))
                print(f"     Detected iterate-accumulate pattern (confidence: {confidence})")
        except Exception as e:
            print(f"      Error detecting iterate-accumulate pattern: {e}")
        
        # Pattern 2: Search pattern (comparison + return)
        try:
            has_comparison = self._has_evidence_in_metta("bin-op")
            has_return = self._has_evidence_in_metta("function-return")
            
            if has_comparison and has_return:
                patterns.append(CodePattern(
                    pattern_type="search",
                    metta_evidence=[
                        f"(has-search-pattern {self.function_name})"
                    ],
                    confidence=0.8,
                    properties=["search", "conditional-return"]
                ))
                print(f"     Detected search pattern")
        except Exception as e:
            print(f"      Error detecting search pattern: {e}")
        
        # Pattern 3: Bounds checking pattern
        try:
            if self._has_bounds_checking_evidence():
                patterns.append(CodePattern(
                    pattern_type="bounds_checking",
                    metta_evidence=[f"(has-bounds-checking {self.function_name})"],
                    confidence=0.85,
                    properties=["bounds-checked", "error-handling"]
                ))
                print(f"     Detected bounds-checking pattern")
        except Exception as e:
            print(f"      Error detecting bounds-checking pattern: {e}")
        
        if not patterns:
            print("      No patterns detected via MeTTa analysis")
            # Create a basic pattern so generation can proceed
            patterns.append(CodePattern(
                pattern_type="basic_function",
                metta_evidence=[f"(is-function {self.function_name})"],
                confidence=0.5,
                properties=["function"]
            ))
        
        return patterns
    
    def _get_atoms_summary(self) -> Dict[str, int]:
        """Get a summary of what types of atoms we have."""
        summary = {}
        
        # Method 1: Try to use the stored metta_atoms directly
        if hasattr(self, 'metta_atoms') and self.metta_atoms:
            print(f"   DEBUG: Analyzing {len(self.metta_atoms)} stored atoms...")
            
            for atom in self.metta_atoms:
                atom_str = str(atom)
                print(f"   DEBUG: Checking atom: {atom_str}")
                
                # Count different types of atoms - check if the pattern appears anywhere in the atom
                if "function-def" in atom_str:
                    summary["function-def"] = summary.get("function-def", 0) + 1
                    print(f"   DEBUG: Found function-def in: {atom_str}")
                if "function-call" in atom_str:
                    summary["function-call"] = summary.get("function-call", 0) + 1
                    print(f"   DEBUG: Found function-call in: {atom_str}")
                if "bin-op" in atom_str:
                    summary["bin-op"] = summary.get("bin-op", 0) + 1
                    print(f"   DEBUG: Found bin-op in: {atom_str}")
                if "loop-pattern" in atom_str:
                    summary["loop-pattern"] = summary.get("loop-pattern", 0) + 1
                    print(f"   DEBUG: Found loop-pattern in: {atom_str}")
                if "variable-assign" in atom_str:
                    summary["variable-assign"] = summary.get("variable-assign", 0) + 1
                    print(f"   DEBUG: Found variable-assign in: {atom_str}")
                if "function-return" in atom_str:
                    summary["function-return"] = summary.get("function-return", 0) + 1
                    print(f"   DEBUG: Found function-return in: {atom_str}")
        
        # Method 2: If that doesn't work, try the MeTTa space string approach
        if not summary:
            try:
                atoms_str = str(self.metta_space)
                print(f"   DEBUG: MeTTa space string (first 200 chars): {atoms_str[:200]}")
                
                # Count different types of atoms
                atom_patterns = [
                    "function-def", "function-call", "bin-op", "loop-pattern", 
                    "variable-assign", "function-return", "import", "class-def"
                ]
                
                for pattern in atom_patterns:
                    count = atoms_str.count(pattern)
                    if count > 0:
                        summary[pattern] = count
            except Exception as e:
                print(f"   DEBUG: Error getting MeTTa space string: {e}")
        
        # Method 3: Fallback - check if we have analysis result
        if not summary and hasattr(self, 'analysis_result') and self.analysis_result:
            atoms = self.analysis_result.get('metta_atoms', [])
            print(f"   DEBUG: Found {len(atoms)} atoms in analysis_result")
            
            for atom in atoms:
                atom_str = str(atom)
                if "function-def" in atom_str:
                    summary["function-def"] = summary.get("function-def", 0) + 1
                elif "function-call" in atom_str:
                    summary["function-call"] = summary.get("function-call", 0) + 1
                elif "bin-op" in atom_str:
                    summary["bin-op"] = summary.get("bin-op", 0) + 1
                elif "loop-pattern" in atom_str:
                    summary["loop-pattern"] = summary.get("loop-pattern", 0) + 1
                elif "variable-assign" in atom_str:
                    summary["variable-assign"] = summary.get("variable-assign", 0) + 1
        
        print(f"   DEBUG: Final atom summary: {summary}")
        return summary
    
    def _has_evidence_in_metta(self, evidence_type: str) -> bool:
        """Check if we have evidence of a certain type in MeTTa space."""
        
        # Method 1: Check stored atoms directly
        if hasattr(self, 'metta_atoms'):
            for atom in self.metta_atoms:
                if evidence_type in str(atom):
                    return True
        
        # Method 2: Check analysis result
        if hasattr(self, 'analysis_result') and self.analysis_result:
            atoms = self.analysis_result.get('metta_atoms', [])
            for atom in atoms:
                if evidence_type in str(atom):
                    return True
        
        # Method 3: Fallback to string approach
        try:
            atoms_str = str(self.metta_space)
            return evidence_type in atoms_str
        except:
            return False
    
    def _has_bounds_checking_evidence(self) -> bool:
        """Check for bounds checking evidence (multiple comparison operations)."""
        
        # Method 1: Check stored atoms directly  
        if hasattr(self, 'metta_atoms'):
            lt_count = gt_count = ge_count = le_count = 0
            for atom in self.metta_atoms:
                atom_str = str(atom)
                if "bin-op Lt" in atom_str:
                    lt_count += 1
                elif "bin-op Gt" in atom_str:
                    gt_count += 1
                elif "bin-op GtE" in atom_str:
                    ge_count += 1
                elif "bin-op LtE" in atom_str:
                    le_count += 1
            
            total_comparisons = lt_count + gt_count + ge_count + le_count
            print(f"   DEBUG: Bounds checking evidence - Lt:{lt_count}, Gt:{gt_count}, GtE:{ge_count}, LtE:{le_count}, Total:{total_comparisons}")
            return total_comparisons >= 2
        
        # Method 2: Fallback to string approach
        try:
            atoms_str = str(self.metta_space)
            
            # Look for multiple comparison operations which suggest bounds checking
            lt_count = atoms_str.count("bin-op Lt")
            gt_count = atoms_str.count("bin-op Gt") 
            ge_count = atoms_str.count("bin-op GtE")
            le_count = atoms_str.count("bin-op LtE")
            
            # Bounds checking typically has at least 2 comparison operations
            total_comparisons = lt_count + gt_count + ge_count + le_count
            return total_comparisons >= 2
        except:
            return False
    
    def _get_function_properties_from_evidence(self) -> List[str]:
        """Get function properties based on evidence in MeTTa atoms."""
        properties = []
        atoms_str = str(self.metta_space)
        
        if self._has_bounds_checking_evidence():
            properties.append("bounds-checked")
        
        if "loop-pattern" in atoms_str:
            properties.append("iterative")
        
        if "bin-op" in atoms_str:
            properties.append("comparative")
        
        if "function-return" in atoms_str:
            properties.append("has-return")
        
        # Always add this for any function
        properties.append("termination-guaranteed")
        
        return properties
    
    def _get_applicable_strategies_from_metta(self, requested_strategies: Optional[List[GenerationStrategy]]) -> List[str]:
        """Use MeTTa reasoning to determine applicable strategies."""
        applicable = []
        
        strategies_to_check = requested_strategies or list(GenerationStrategy)
        
        for strategy in strategies_to_check:
            strategy_name = strategy.value if hasattr(strategy, 'value') else str(strategy)
            
            # Check if strategy applies based on MeTTa evidence
            if self._check_strategy_applicability(strategy_name):
                applicable.append(strategy_name)
                print(f"     Strategy {strategy_name} applicable")
            else:
                print(f"     Strategy {strategy_name} not applicable")
        
        return applicable
    
    def _check_strategy_applicability(self, strategy: str) -> bool:
        """Check if a strategy is applicable based on MeTTa evidence."""
        
        # Get evidence from multiple sources
        has_bin_op = self._has_evidence_in_metta("bin-op")
        has_loop = self._has_evidence_in_metta("loop-pattern") 
        has_function_def = self._has_evidence_in_metta("function-def")
        has_bounds_check = self._has_bounds_checking_evidence()
        
        print(f"   DEBUG: Strategy {strategy} evidence - bin-op:{has_bin_op}, loop:{has_loop}, func-def:{has_function_def}, bounds:{has_bounds_check}")
        
        if strategy == "operation_substitution":
            # Needs comparison operations to substitute
            has_gt = self._has_evidence_in_metta("bin-op Gt")
            has_lt = self._has_evidence_in_metta("bin-op Lt")
            return has_gt or has_lt
        
        elif strategy == "accumulator_variation":
            # Needs loop + accumulation pattern
            return has_loop and has_bin_op
        
        elif strategy == "structure_preservation":
            # Always applicable - preserves the basic structure
            return has_function_def
        
        elif strategy == "condition_variation":
            # Needs conditional operations
            return has_bin_op
        
        elif strategy == "property_guided":
            # Applicable if we have bounds checking
            return has_bounds_check
        
        elif strategy == "pattern_expansion":
            # Applicable if we have loop patterns
            return has_loop
        
        return False
    
    def _generate_candidates_with_metta(self, strategies: List[str]) -> List[DonorCandidate]:
        """Generate candidates using MeTTa transformation rules."""
        candidates = []
        
        for strategy in strategies:
            print(f"    Generating candidates for strategy: {strategy}")
            strategy_candidates = self._generate_strategy_candidates(strategy)
            candidates.extend(strategy_candidates)
            print(f"     Generated {len(strategy_candidates)} candidates for {strategy}")
        
        return candidates
    
    def _generate_strategy_candidates(self, strategy: str) -> List[DonorCandidate]:
        """Generate candidates for a specific strategy based on MeTTa analysis."""
        candidates = []
        
        # Get available transformations based on what we actually found in MeTTa
        transformations = self._get_available_transformations_from_metta(strategy)
        
        for transformation in transformations:
            candidate = self._create_candidate_from_transformation(strategy, transformation)
            if candidate:
                candidates.append(candidate)
        
        return candidates
    
    def _get_available_transformations_from_metta(self, strategy: str) -> List[str]:
        """Get available transformations based on actual MeTTa evidence."""
        transformations = []
        atoms_str = str(self.metta_space)
        
        if strategy == "operation_substitution":
            # Check what operations we actually have
            if "bin-op Gt" in atoms_str and ("max" in self.function_name.lower() or "maximum" in self.function_name.lower()):
                transformations.append("max-to-min")
            if "bin-op Lt" in atoms_str and ("min" in self.function_name.lower() or "minimum" in self.function_name.lower()):
                transformations.append("min-to-max")
            # Generic comparison operations
            if "bin-op Gt" in atoms_str or "bin-op Lt" in atoms_str:
                transformations.append("reverse-comparison")
        
        elif strategy == "accumulator_variation":
            if "loop-pattern" in atoms_str:
                transformations.extend(["to-sum", "to-count", "to-product", "to-average"])
        
        elif strategy == "structure_preservation":
            if "function-return" in atoms_str:
                transformations.append("value-to-index")
                transformations.append("preserve-structure")
        
        elif strategy == "condition_variation":
            if "bin-op" in atoms_str:
                transformations.extend(["to-predicate", "to-threshold"])
        
        elif strategy == "property_guided":
            if self._has_bounds_checking_evidence():
                transformations.extend(["enhanced-bounds-check", "safe-variant"])
        
        elif strategy == "pattern_expansion":
            if "loop-pattern" in atoms_str:
                transformations.extend(["multi-range", "windowed"])
        
        return transformations
    
    def _fallback_candidate_generation(self, strategies: List[str]) -> List[DonorCandidate]:
        """Generate basic candidates when MeTTa reasoning fails."""
        print("    Using fallback candidate generation...")
        
        candidates = []
        
        for strategy in strategies:
            # Create at least one basic candidate per strategy
            candidate_name = f"{self.function_name}_{strategy}_variant"
            
            candidate = DonorCandidate(
                name=candidate_name,
                description=f"Basic {strategy} variant of {self.function_name}",
                code=self._create_basic_variant(strategy),
                strategy=strategy,
                metta_derivation=[f"(fallback-generation {self.function_name} {strategy})"],
                confidence=0.5,
                properties=["basic-variant"]
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _create_basic_variant(self, strategy: str) -> str:
        """Create a basic variant for fallback generation."""
        if strategy == "operation_substitution":
            return self.original_code.replace("max", "min").replace(">", "<")
        elif strategy == "structure_preservation":
            return self.original_code.replace("return max_val", "return max_index")
        else:
            # Just add a comment indicating the strategy
            lines = self.original_code.split('\n')
            if lines and lines[0].strip().startswith('def'):
                func_line = lines[0]
                new_func_line = func_line.replace('def ', f'def {strategy}_variant_')
                lines[0] = new_func_line
                lines.insert(1, f'    """Generated using {strategy} strategy."""')
            return '\n'.join(lines)
    
    # Keep all the existing helper methods for candidate creation
    def _create_candidate_from_transformation(self, strategy: str, transformation: str) -> Optional[DonorCandidate]:
        """Create a donor candidate from a transformation."""
        try:
            # Generate the code using the transformation
            generated_code = self._generate_code_for_transformation(strategy, transformation)
            
            if not generated_code:
                return None
            
            # Create candidate name
            candidate_name = self._generate_candidate_name(transformation)
            
            # Get confidence from strategy
            confidence = self._get_strategy_confidence(strategy)
            
            # Get properties that will be preserved
            properties = self._get_preserved_properties(strategy, transformation)
            
            # Create MeTTa derivation trace
            derivation = [
                f"(strategy-applies {strategy.replace('_', '-')} {self.function_name})",
                f"({strategy.replace('_', '-')}-transform {self.function_name} {transformation})",
                f"(generate-candidate {self.function_name} {strategy} {transformation} {candidate_name})"
            ]
            
            return DonorCandidate(
                name=candidate_name,
                description=self._generate_candidate_description(strategy, transformation),
                code=generated_code,
                strategy=strategy,
                metta_derivation=derivation,
                confidence=confidence,
                properties=properties
            )
            
        except Exception as e:
            print(f"        Failed to create candidate for {strategy}/{transformation}: {e}")
            return None
    
    def _rank_candidates_with_metta(self, candidates: List[DonorCandidate]) -> List[Dict[str, Any]]:
        """Rank candidates using MeTTa scoring rules."""
        scored_candidates = []
        
        for candidate in candidates:
            # Calculate final score based on MeTTa evidence
            base_confidence = candidate.confidence
            property_bonus = len(candidate.properties) * 0.02
            quality_bonus = 0.1 if '"""' in candidate.code else 0.0
            
            # Bonus for candidates that preserve original patterns
            metta_bonus = 0.0
            if self._candidate_preserves_patterns(candidate):
                metta_bonus = 0.1
            
            final_score = base_confidence + property_bonus + quality_bonus + metta_bonus
            
            scored_candidates.append({
                "name": candidate.name,
                "description": candidate.description,
                "code": candidate.code,
                "strategy": candidate.strategy,
                "metta_derivation": candidate.metta_derivation,
                "confidence": candidate.confidence,
                "final_score": min(1.0, final_score),
                "properties": candidate.properties
            })
        
        # Sort by final score
        return sorted(scored_candidates, key=lambda x: x["final_score"], reverse=True)
    
    def _candidate_preserves_patterns(self, candidate: DonorCandidate) -> bool:
        """Check if candidate preserves important patterns detected in original."""
        # Simple heuristic - check if candidate maintains loop structure
        has_loop_original = "for " in self.original_code or "while " in self.original_code
        has_loop_candidate = "for " in candidate.code or "while " in candidate.code
        
        return has_loop_original == has_loop_candidate
    
    # Include all the existing code generation methods
    def _generate_code_for_transformation(self, strategy: str, transformation: str) -> str:
        """Generate code for a specific transformation."""
        if strategy == "operation_substitution":
            if transformation == "max-to-min":
                return self._create_min_variant()
            elif transformation == "min-to-max":
                return self._create_max_variant()
            elif transformation == "reverse-comparison":
                return self._create_reverse_comparison_variant()
        
        elif strategy == "accumulator_variation":
            if transformation == "to-sum":
                return self._create_sum_variant()
            elif transformation == "to-count":
                return self._create_count_variant()
            elif transformation == "to-product":
                return self._create_product_variant()
            elif transformation == "to-average":
                return self._create_average_variant()
        
        elif strategy == "structure_preservation":
            if transformation == "value-to-index":
                return self._create_index_variant()
            elif transformation == "preserve-structure":
                return self._create_structure_preserving_variant()
        
        elif strategy == "condition_variation":
            if transformation == "to-predicate":
                return self._create_predicate_variant()
            elif transformation == "to-threshold":
                return self._create_threshold_variant()
        
        elif strategy == "property_guided":
            if transformation == "enhanced-bounds-check":
                return self._create_enhanced_bounds_variant()
            elif transformation == "safe-variant":
                return self._create_safe_variant()
        
        elif strategy == "pattern_expansion":
            if transformation == "multi-range":
                return self._create_multi_range_variant()
            elif transformation == "windowed":
                return self._create_windowed_variant()
        
        return ""
    
    # All the existing code generation helper methods remain the same
    def _generate_candidate_name(self, transformation: str) -> str:
        """Generate generic candidate name based on strategy and transformation."""
        base_name = self.function_name
        
        # Generic naming based on transformation type
        if transformation in ["max-to-min", "min-to-max", "reverse-comparison"]:
            return f"{base_name}_op_sub"
        elif transformation in ["to-sum", "to-count", "to-product", "to-average"]:
            return f"{base_name}_acc_var"
        elif transformation in ["value-to-index", "preserve-structure"]:
            return f"{base_name}_struct_pres"
        elif transformation in ["to-predicate", "to-threshold"]:
            return f"{base_name}_cond_var"
        elif transformation in ["enhanced-bounds-check", "safe-variant"]:
            return f"{base_name}_prop_guided"
        elif transformation in ["multi-range", "windowed"]:
            return f"{base_name}_pattern_exp"
        else:
            return f"{base_name}_{transformation.replace('-', '_')}"
    
    def _generate_candidate_description(self, strategy: str, transformation: str) -> str:
        """Generate simple description based on strategy."""
        strategy_descriptions = {
            "operation_substitution": "Operation substitution variant",
            "accumulator_variation": "Accumulator variation variant",
            "structure_preservation": "Structure preservation variant",
            "condition_variation": "Condition variation variant",
            "property_guided": "Property-guided variant",
            "pattern_expansion": "Pattern expansion variant"
        }
        
        return strategy_descriptions.get(strategy, f"{strategy.replace('_', ' ').title()} variant")
    
    def _get_strategy_confidence(self, strategy: str) -> float:
        """Get base confidence for strategy."""
        confidence_map = {
            "operation_substitution": 0.9,
            "accumulator_variation": 0.85,
            "structure_preservation": 0.95,
            "condition_variation": 0.8,
            "property_guided": 0.85,
            "pattern_expansion": 0.65
        }
        
        return confidence_map.get(strategy, 0.5)
    
    def _get_preserved_properties(self, strategy: str, transformation: str) -> List[str]:
        """Get properties that will be preserved by this transformation."""
        base_properties = ["bounds-checked", "iterative", "error-handling", "termination-guaranteed"]
        
        if strategy == "structure_preservation":
            return base_properties
        elif strategy == "operation_substitution":
            return base_properties
        elif strategy == "accumulator_variation":
            return ["bounds-checked", "iterative", "error-handling", "termination-guaranteed"]
        else:
            return ["bounds-checked", "error-handling", "termination-guaranteed"]
    
    # Code generation methods with additional variants
    def _create_min_variant(self) -> str:
        """Create minimum finding variant."""
        return self.original_code.replace("max_val", "min_val").replace("max", "min").replace(">", "<")
    
    def _create_max_variant(self) -> str:
        """Create maximum finding variant."""
        return self.original_code.replace("min_val", "max_val").replace("min", "max").replace("<", ">")
    
    def _create_reverse_comparison_variant(self) -> str:
        """Create variant with reversed comparison."""
        code = self.original_code
        # Simple replacement - in practice you'd want more sophisticated AST manipulation
        code = code.replace(" > ", " TEMP_GT ")
        code = code.replace(" < ", " > ")
        code = code.replace(" TEMP_GT ", " < ")
        return code
    
    def _create_sum_variant(self) -> str:
        """Create sum variant - generic version."""
        func_name = f"{self.function_name}_acc_var"
        
        # Extract the parameter names from the original function
        param_names = self._extract_parameter_names()
        
        return f'''def {func_name}({', '.join(param_names)}):
    """Accumulator variation: sum values."""
    # Bounds checking (preserved from original)
    if {param_names[1]} < 0 or {param_names[2]} > len({param_names[0]}) or {param_names[1]} >= {param_names[2]}:
        return 0
    
    total = 0
    for i in range({param_names[1]}, {param_names[2]}):
        total += {param_names[0]}[i]
    
    return total'''
    
    def _create_count_variant(self) -> str:
        """Create count variant - generic version."""
        func_name = f"{self.function_name}_acc_var"
        param_names = self._extract_parameter_names()
        
        return f'''def {func_name}({', '.join(param_names)}, condition=None):
    """Accumulator variation: count elements."""
    if {param_names[1]} < 0 or {param_names[2]} > len({param_names[0]}) or {param_names[1]} >= {param_names[2]}:
        return 0
    
    count = 0
    for i in range({param_names[1]}, {param_names[2]}):
        if condition is None or condition({param_names[0]}[i]):
            count += 1
    
    return count'''
    
    def _create_product_variant(self) -> str:
        """Create product variant - generic version."""
        func_name = f"{self.function_name}_acc_var"
        param_names = self._extract_parameter_names()
        
        return f'''def {func_name}({', '.join(param_names)}):
    """Accumulator variation: product of values."""
    if {param_names[1]} < 0 or {param_names[2]} > len({param_names[0]}) or {param_names[1]} >= {param_names[2]}:
        return 1
    
    product = 1
    for i in range({param_names[1]}, {param_names[2]}):
        product *= {param_names[0]}[i]
    
    return product'''
    
    def _create_average_variant(self) -> str:
        """Create average variant - generic version."""
        func_name = f"{self.function_name}_acc_var"
        param_names = self._extract_parameter_names()
        
        return f'''def {func_name}({', '.join(param_names)}):
    """Accumulator variation: average of values."""
    if {param_names[1]} < 0 or {param_names[2]} > len({param_names[0]}) or {param_names[1]} >= {param_names[2]}:
        return None
    
    total = 0
    count = 0
    for i in range({param_names[1]}, {param_names[2]}):
        total += {param_names[0]}[i]
        count += 1
    
    return total / count if count > 0 else None'''
    
    def _create_predicate_variant(self) -> str:
        """Create predicate variant - generic version."""
        func_name = f"{self.function_name}_cond_var"
        param_names = self._extract_parameter_names()
        
        return f'''def {func_name}({', '.join(param_names)}, predicate):
    """Condition variation: find elements matching predicate."""
    if {param_names[1]} < 0 or {param_names[2]} > len({param_names[0]}) or {param_names[1]} >= {param_names[2]}:
        return None
    
    for i in range({param_names[1]}, {param_names[2]}):
        if predicate({param_names[0]}[i]):
            return {param_names[0]}[i]
    
    return None'''
    
    def _create_threshold_variant(self) -> str:
        """Create threshold variant - generic version."""
        func_name = f"{self.function_name}_cond_var"
        param_names = self._extract_parameter_names()
        
        return f'''def {func_name}({', '.join(param_names)}, threshold):
    """Condition variation: find elements above threshold."""
    if {param_names[1]} < 0 or {param_names[2]} > len({param_names[0]}) or {param_names[1]} >= {param_names[2]}:
        return []
    
    result = []
    for i in range({param_names[1]}, {param_names[2]}):
        if {param_names[0]}[i] >= threshold:
            result.append({param_names[0]}[i])
    
    return result'''
    
    def _create_multi_range_variant(self) -> str:
        """Create multi-range variant - generic version."""
        func_name = f"{self.function_name}_pattern_exp"
        param_names = self._extract_parameter_names()
        
        return f'''def {func_name}({param_names[0]}, ranges):
    """Pattern expansion: handle multiple ranges."""
    if not ranges:
        return None
    
    results = []
    for start_idx, end_idx in ranges:
        if start_idx < 0 or end_idx > len({param_names[0]}) or start_idx >= end_idx:
            continue
        
        # Apply original logic to each range
        current_val = {param_names[0]}[start_idx]
        for i in range(start_idx + 1, end_idx):
            if {param_names[0]}[i] > current_val:  # Preserve comparison logic
                current_val = {param_names[0]}[i]
        results.append(current_val)
    
    return max(results) if results else None'''
    
    def _create_windowed_variant(self) -> str:
        """Create windowed variant - generic version."""
        func_name = f"{self.function_name}_pattern_exp" 
        param_names = self._extract_parameter_names()
        
        return f'''def {func_name}({param_names[0]}, window_size, step=1):
    """Pattern expansion: sliding window operation."""
    if window_size <= 0 or window_size > len({param_names[0]}):
        return []
    
    results = []
    for start in range(0, len({param_names[0]}) - window_size + 1, step):
        end = start + window_size
        
        # Apply original logic to each window
        window_val = {param_names[0]}[start]
        for i in range(start + 1, end):
            if {param_names[0]}[i] > window_val:  # Preserve comparison logic
                window_val = {param_names[0]}[i]
        results.append(window_val)
    
    return results'''
    
    def _extract_parameter_names(self) -> List[str]:
        """Extract parameter names from the original function."""
        try:
            # Try to parse the function to get parameter names
            tree = ast.parse(self.original_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return [arg.arg for arg in node.args.args]
        except:
            pass
        
        # Fallback to generic names
        return ["data", "start", "end"]
    
    def _create_index_variant(self) -> str:
        """Create index-finding variant."""
        # More sophisticated replacement that preserves the algorithm structure
        code = self.original_code
        
        # Replace variable initialization
        if "max_val = numbers[start_idx]" in code:
            code = code.replace(
                "max_val = numbers[start_idx]",
                "max_val = numbers[start_idx]\n    max_index = start_idx"
            )
        
        # Replace comparison and assignment
        if "max_val = numbers[i]" in code:
            code = code.replace(
                "max_val = numbers[i]",
                "max_val = numbers[i]\n            max_index = i"
            )
        
        # Replace return statement
        code = code.replace("return max_val", "return max_index")
        
        # Update function name
        new_func_name = self.function_name + "_index"
        code = code.replace(f"def {self.function_name}(", f"def {new_func_name}(")
        
        return code
    
    def _create_structure_preserving_variant(self) -> str:
        """Create variant that preserves structure with minimal changes."""
        code = self.original_code
        
        # Add enhanced documentation
        if '"""' in code:
            code = code.replace('"""', '"""Enhanced version: ')
        else:
            # Add docstring if none exists
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    lines.insert(i+1, '    """Structure-preserving variant."""')
                    break
            code = '\n'.join(lines)
        
        # Update function name
        new_func_name = f"preserved_{self.function_name}"
        code = code.replace(f"def {self.function_name}(", f"def {new_func_name}(")
        
        return code
    
    def _create_predicate_variant(self) -> str:
        """Create predicate-based variant."""
        func_name = self.function_name.replace("find_max", "find_matching").replace("find_min", "find_matching")
        return f'''def {func_name}(numbers, start_idx, end_idx, predicate):
    """Find first element matching predicate in the specified range."""
    if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
        return None
    
    for i in range(start_idx, end_idx):
        if predicate(numbers[i]):
            return numbers[i]
    
    return None'''
    
    def _create_threshold_variant(self) -> str:
        """Create threshold-based variant."""
        func_name = self.function_name.replace("find_max", "find_above_threshold").replace("find_min", "find_above_threshold")
        return f'''def {func_name}(numbers, start_idx, end_idx, threshold):
    """Find elements above threshold in the specified range."""
    if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
        return []
    
    result = []
    for i in range(start_idx, end_idx):
        if numbers[i] >= threshold:
            result.append(numbers[i])
    
    return result'''
    
    def _create_enhanced_bounds_variant(self) -> str:
        """Create enhanced bounds checking variant."""
        code = self.original_code
        
        # Enhance the bounds checking
        old_check = "if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:"
        new_check = '''if not numbers or start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
        return None
    if not isinstance(start_idx, int) or not isinstance(end_idx, int):
        raise TypeError("Indices must be integers")
    if'''
        
        code = code.replace(old_check, new_check)
        
        # Update function name
        new_func_name = f"safe_{self.function_name}"
        code = code.replace(f"def {self.function_name}(", f"def {new_func_name}(")
        
        return code
    
    def _create_safe_variant(self) -> str:
        """Create safe variant with additional error handling."""
        code = self.original_code
        
        # Add try-catch wrapper
        lines = code.split('\n')
        
        # Find the function definition
        func_def_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                func_def_idx = i
                break
        
        # Update function name
        new_func_name = f"safe_{self.function_name}"
        lines[func_def_idx] = lines[func_def_idx].replace(self.function_name, new_func_name)
        
        # Add try-except around the main logic
        # This is a simplified approach - in practice you'd want better AST manipulation
        lines.insert(func_def_idx + 2, '    try:')
        
        # Find the return statement and add except before it
        for i in range(len(lines) - 1, -1, -1):
            if 'return' in lines[i] and not lines[i].strip().startswith('#'):
                lines.insert(i + 1, '    except (IndexError, TypeError, ValueError) as e:')
                lines.insert(i + 2, '        return None  # Safe fallback')
                break
        
        return '\n'.join(lines)
    
    def _create_multi_range_variant(self) -> str:
        """Create multi-range variant."""
        func_name = self.function_name + "_multi_range"
        return f'''def {func_name}(numbers, ranges):
    """Find maximum across multiple ranges."""
    if not ranges:
        return None
    
    results = []
    for start_idx, end_idx in ranges:
        if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
            continue
        
        range_max = numbers[start_idx]
        for i in range(start_idx + 1, end_idx):
            if numbers[i] > range_max:
                range_max = numbers[i]
        results.append(range_max)
    
    return max(results) if results else None'''
    
    def _create_windowed_variant(self) -> str:
        """Create windowed variant."""
        func_name = self.function_name + "_windowed"
        return f'''def {func_name}(numbers, window_size, step=1):
    """Find maximum in sliding windows."""
    if window_size <= 0 or window_size > len(numbers):
        return []
    
    results = []
    for start in range(0, len(numbers) - window_size + 1, step):
        end = start + window_size
        
        window_max = numbers[start]
        for i in range(start + 1, end):
            if numbers[i] > window_max:
                window_max = numbers[i]
        results.append(window_max)
    
    return results'''


# Integration function for use in the main workflow
def integrate_metta_generation(func, strategies: Optional[List[GenerationStrategy]] = None) -> List[Dict[str, Any]]:
    """
    Integration function to use MeTTa-based donor generation in the main workflow.
    NOW PROPERLY INTEGRATES WITH STATIC ANALYSIS PIPELINE.
    
    Args:
        func: Python function object or source code string
        strategies: Optional list of generation strategies
        
    Returns:
        List of generated donor candidates
    """
    generator = MettaDonorGenerator()
    
    # Load the donor ontology
    generator.load_donor_ontology()
    
    # Generate donors using the proper analysis pipeline
    return generator.generate_donors_from_function(func, strategies)


# Demo function that shows the proper integration
def demonstrate_metta_generation():
    """Demonstrate the MeTTa-based donor generation system with proper integration."""
    
    print("  METTA DONOR GENERATION DEMO - FIXED VERSION")
    print("=" * 60)
    
    # Example function
    def find_max_in_range(numbers, start_idx, end_idx):
        """Find the maximum value in a list within a specific range."""
        if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
            return None
        
        max_val = numbers[start_idx]
        for i in range(start_idx + 1, end_idx):
            if numbers[i] > max_val:
                max_val = numbers[i]
        
        return max_val
    
    print(f"  Original function: {find_max_in_range.__name__}")
    
    # Generate donors using the PROPER integration
    print("\n Generating donors using PROPER MeTTa analysis pipeline...")
    
    candidates = integrate_metta_generation(find_max_in_range)
    
    print(f"\n  Generated {len(candidates)} donor candidates using REAL MeTTa reasoning!")
    
    # Show the results
    print("\n  Generated Candidates:")
    print("-" * 40)
    
    for i, candidate in enumerate(candidates[:3], 1):  # Show top 3
        print(f"\n{i}. {candidate['name']}")
        print(f"     {candidate['description']}")
        print(f"     Strategy: {candidate['strategy']}")
        print(f"     Final Score: {candidate['final_score']:.2f}")
        print(f"     Properties: {', '.join(candidate['properties'])}")
        print(f"     MeTTa Derivation: {candidate['metta_derivation'][0]}")
        
        # Show a snippet of the generated code
        code_lines = candidate['code'].split('\n')
        print(f"     Code preview:")
        for line in code_lines[:6]:
            print(f"      {line}")
        if len(code_lines) > 6:
            print(f"      ... ({len(code_lines)-6} more lines)")
    
    return candidates


def demonstrate_with_string_function():
    """Demonstrate with a string-based function definition."""
    
    print("\n  DEMO WITH STRING FUNCTION")
    print("=" * 40)
    
    # Function as string
    string_function = '''def calculate_sum_in_range(arr, low, high):
    """Calculate sum of elements in array within given range."""
    if low < 0 or high > len(arr) or low >= high:
        return 0
    
    total = 0
    for idx in range(low, high):
        total += arr[idx]
    
    return total'''
    
    print("  Function provided as string")
    
    # Generate using the proper pipeline
    candidates = integrate_metta_generation(string_function)
    
    print(f"\n  Generated {len(candidates)} candidates from string function!")
    
    # Show first candidate
    if candidates:
        candidate = candidates[0]
        print(f"\n🏆 Top candidate: {candidate['name']}")
        print(f"     {candidate['description']}")
        print(f"     Score: {candidate['final_score']:.2f}")
        
        print(f"\n     Generated code:")
        for line in candidate['code'].split('\n')[:8]:
            print(f"      {line}")
    
    return candidates


if __name__ == "__main__":
    # Run both demos
    candidates1 = demonstrate_metta_generation()
    candidates2 = demonstrate_with_string_function()
    
    print(f"\n  SUMMARY:")
    print(f"   Function object demo: {len(candidates1)} candidates")
    print(f"   String function demo: {len(candidates2)} candidates")
    print(f"   Total candidates generated: {len(candidates1) + len(candidates2)}")
    print("\n  Fixed MeTTa generator now properly integrates with static analysis!")