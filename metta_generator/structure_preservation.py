#!/usr/bin/env python3
"""
Simple Structure Preservation Generator
Creates variants that preserve the original structure while making minor modifications.
"""

from typing import List, Dict, Optional
from metta_generator.base import BaseDonorGenerator, GenerationContext, DonorCandidate, GenerationStrategy

class StructurePreservationGenerator(BaseDonorGenerator):
    """Generator that preserves original structure while making safe modifications."""
    
    def __init__(self):
        super().__init__()
        
    def can_generate(self, context: GenerationContext, strategy) -> bool:
        """Check if this generator can handle the given context and strategy."""
        # Handle both enum and string strategy types
        if hasattr(strategy, 'value'):
            strategy_name = strategy.value
        elif hasattr(strategy, 'name'):
            strategy_name = strategy.name.lower()
        else:
            strategy_name = str(strategy).lower()
            
        return strategy_name in ["structure_preservation"]
    
    def generate_candidates(self, context: GenerationContext, strategy) -> List[DonorCandidate]:
        """Generate donor candidates for the given context and strategy."""
        candidates = self._generate_candidates_impl(context, strategy)
        
        # Ensure all candidates have proper generator attribution
        for candidate in candidates:
            if not hasattr(candidate, 'generator_used') or candidate.generator_used == "UnknownGenerator":
                candidate.generator_used = self.generator_name
        
        return candidates
    
    def _generate_candidates_impl(self, context: GenerationContext, strategy) -> List[DonorCandidate]:
        """Generate structure preservation candidates."""
        candidates = []
        
        # Create a simple variant with added documentation
        doc_variant = self._create_documentation_variant(context)
        if doc_variant:
            candidates.append(doc_variant)
        
        # Create a variant with additional type hints
        type_variant = self._create_type_hint_variant(context)
        if type_variant:
            candidates.append(type_variant)
        
        # Create a variant with improved variable names
        naming_variant = self._create_naming_variant(context)
        if naming_variant:
            candidates.append(naming_variant)
        
        return candidates
    
    def get_supported_strategies(self) -> List:
        """Get list of strategies this generator supports."""
        return [GenerationStrategy.STRUCTURE_PRESERVATION]
    
    def _create_documentation_variant(self, context: GenerationContext) -> Optional[DonorCandidate]:
        """Create a variant with enhanced documentation."""
        try:
            enhanced_code = context.original_code
            func_name = context.function_name
            
            # Add enhanced docstring
            lines = enhanced_code.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    if i + 1 < len(lines) and '"""' in lines[i + 1]:
                        # Enhance existing docstring
                        lines[i + 1] = lines[i + 1].replace('"""', '"""Enhanced with structure preservation. ')
                    else:
                        # Add new enhanced docstring
                        lines.insert(i + 1, '    """Structure-preserved variant with enhanced documentation."""')
                    break
            
            # Rename function
            new_func_name = f"{func_name}_documented"
            enhanced_code = '\n'.join(lines)
            enhanced_code = enhanced_code.replace(f"def {func_name}(", f"def {new_func_name}(")
            
            return DonorCandidate(
                name=new_func_name,
                description="Structure preservation: enhanced documentation",
                code=enhanced_code,
                strategy="structure_preservation",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=self._get_data_structures_from_context(context),
                operations_used=["documentation"],
                metta_derivation=[f"(structure-preservation {func_name} documentation-enhancement)"],
                confidence=0.95,
                properties=["structure-preserved", "well-documented"],
                complexity_estimate="same",
                applicability_scope="broad",
                generator_used=self.generator_name
            )
        except Exception as e:
            print(f"      Failed to create documentation variant: {e}")
            return None
    
    def _create_type_hint_variant(self, context: GenerationContext) -> Optional[DonorCandidate]:
        """Create a variant with type hints."""
        try:
            enhanced_code = context.original_code
            func_name = context.function_name
            
            # Simple type hint addition (basic implementation)
            if "def " + func_name + "(" in enhanced_code and "->" not in enhanced_code:
                enhanced_code = enhanced_code.replace(
                    f"def {func_name}(",
                    f"def {func_name}_typed("
                )
                
                # Add typing import if not present
                if "from typing import" not in enhanced_code:
                    enhanced_code = "from typing import Any, Optional, List\n\n" + enhanced_code
            
            new_func_name = f"{func_name}_typed"
            
            return DonorCandidate(
                name=new_func_name,
                description="Structure preservation: added type hints",
                code=enhanced_code,
                strategy="structure_preservation",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=self._get_data_structures_from_context(context),
                operations_used=["type-annotation"],
                metta_derivation=[f"(structure-preservation {func_name} type-enhancement)"],
                confidence=0.9,
                properties=["structure-preserved", "type-safe"],
                complexity_estimate="same",
                applicability_scope="broad",
                generator_used=self.generator_name
            )
        except Exception as e:
            print(f"      Failed to create type hint variant: {e}")
            return None
    
    def _create_naming_variant(self, context: GenerationContext) -> Optional[DonorCandidate]:
        """Create a variant with improved variable names."""
        try:
            enhanced_code = context.original_code
            func_name = context.function_name
            
            # Simple variable name improvements
            enhanced_code = enhanced_code.replace("i)", "index)")
            enhanced_code = enhanced_code.replace("i ", "index ")
            enhanced_code = enhanced_code.replace("i+", "index+")
            enhanced_code = enhanced_code.replace("i-", "index-")
            
            new_func_name = f"{func_name}_readable"
            enhanced_code = enhanced_code.replace(f"def {func_name}(", f"def {new_func_name}(")
            
            return DonorCandidate(
                name=new_func_name,
                description="Structure preservation: improved variable names",
                code=enhanced_code,
                strategy="structure_preservation",
                pattern_family=self._get_primary_pattern_family(context),
                data_structures_used=self._get_data_structures_from_context(context),
                operations_used=["naming-improvement"],
                metta_derivation=[f"(structure-preservation {func_name} naming-enhancement)"],
                confidence=0.85,
                properties=["structure-preserved", "readable"],
                complexity_estimate="same",
                applicability_scope="broad",
                generator_used=self.generator_name
            )
        except Exception as e:
            print(f"      Failed to create naming variant: {e}")
            return None
    
    def _get_primary_pattern_family(self, context: GenerationContext) -> str:
        """Get the primary pattern family from context."""
        if hasattr(context, 'detected_patterns') and context.detected_patterns:
            return context.detected_patterns[0].pattern_family
        return "generic"
    
    def _get_data_structures_from_context(self, context: GenerationContext) -> List[str]:
        """Get data structures from context."""
        if hasattr(context, 'detected_patterns') and context.detected_patterns:
            return context.detected_patterns[0].data_structures
        return ["generic"]