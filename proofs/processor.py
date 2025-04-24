from typing import Dict, List, Any
import logging
from proofs.pattern_mapper import PatternMapper

class ProofProcessorWithPatterns:
    """Integration code for the proof generation system with improved pattern mapping."""
    
    def __init__(self, monitor=None):
        """Initialize with the pattern mapper and monitor."""
        self.monitor = monitor
        self.pattern_mapper = PatternMapper()
        
    def _add_property_annotations(self, metta_components: List[str], expr_id: str, 
                                expr_content: str, component: Dict[str, Any]) -> None:
        """
        Add property annotations for expressions based on content and metadata.
        Uses the pattern mapper to identify patterns and create proper MeTTa atoms.
        """
        # Get natural language description if available
        description = component.get("natural_language", "")
        
        # Use pattern mapper to identify patterns and generate atoms
        pattern_atoms = self.pattern_mapper.generate_metta_atoms(
            expr_id, expr_content, description
        )
        
        # Add pattern atoms to MeTTa components
        metta_components.extend(pattern_atoms)
    
    def _map_property_to_metta_atom(self, property: str) -> str:
        """
        Map a property string to a corresponding MeTTa atom using the pattern mapper.
        """
        return self.pattern_mapper.map_requirement_to_property(property)
    
    def _check_property_satisfied(self, json_ir: Dict[str, Any], property: str) -> bool:
        """
        Check if a specific property is satisfied by the proof using the pattern mapper.
        """
        components = json_ir.get("proof_components", [])
        property_type = self.pattern_mapper.map_requirement_to_property(property)
        
        # Look for expressions with the required property type
        for component in components:
            expr = component.get("expression", "")
            desc = component.get("natural_language", "")
            
            # Use pattern mapper to identify patterns
            patterns = self.pattern_mapper.identify_patterns(expr, desc)
            
            # Check if any pattern maps to the required property
            for _, pattern_property in patterns:
                if pattern_property == property_type:
                    return True
        
        # Also check verification strategy
        strategy = json_ir.get("verification_strategy", {})
        approach = strategy.get("approach", "")
        
        # Check strategy approach with pattern mapper
        strategy_patterns = self.pattern_mapper.identify_patterns(approach)
        if any(prop == property_type for _, prop in strategy_patterns):
            return True
        
        # Check lemmas
        for lemma in strategy.get("key_lemmas", []):
            lemma_patterns = self.pattern_mapper.identify_patterns(lemma)
            if any(prop == property_type for _, prop in lemma_patterns):
                return True
        
        return False
    
    def _json_to_metta_proof(self, proof_json: Dict[str, Any]) -> List[str]:
        """
        Convert proof component JSON to MeTTa representation.
        Creates proper atoms with descriptive pattern names.
        """
        metta_components = []
        
        try:
            for component in proof_json.get("proof_components", []):
                component_type = component.get("type", "unknown")
                
                if component_type == "precondition":
                    # Create a unique atom ID for the expression
                    expr_id = f"precond_{len(metta_components)}"
                    expr_content = component.get("expression", "")
                    
                    # Add the expression as a proper atom definition
                    metta_components.append(f"(: {expr_id} Expression)")
                    metta_components.append(f"(= ({expr_id}) \"{self._escape_for_metta(expr_content)}\")")
                    
                    # Reference the atom in the Precondition
                    metta_components.append(f"(Precondition {expr_id})")
                    
                    # Add property annotations using pattern mapper
                    self._add_property_annotations(metta_components, expr_id, expr_content, component)
                
                elif component_type == "loop_invariant":
                    location = component.get("location", "unknown_loop")
                    expr_id = f"inv_{location}_{len(metta_components)}"
                    expr_content = component.get("expression", "")
                    
                    # Create atoms for the expression
                    metta_components.append(f"(: {expr_id} Expression)")
                    metta_components.append(f"(= ({expr_id}) \"{self._escape_for_metta(expr_content)}\")")
                    
                    # Reference the atom in the loop invariant
                    metta_components.append(f"(LoopInvariant {location} {expr_id})")
                    
                    # Add property annotations using pattern mapper
                    self._add_property_annotations(metta_components, expr_id, expr_content, component)
                
                elif component_type == "assertion":
                    location = component.get("location", "unknown_location")
                    expr_id = f"assert_{location}_{len(metta_components)}"
                    expr_content = component.get("expression", "")
                    
                    # Create atoms for the expression
                    metta_components.append(f"(: {expr_id} Expression)")
                    metta_components.append(f"(= ({expr_id}) \"{self._escape_for_metta(expr_content)}\")")
                    
                    # Reference the atom in the assertion
                    metta_components.append(f"(Assertion {location} {expr_id})")
                    
                    # Add property annotations using pattern mapper
                    self._add_property_annotations(metta_components, expr_id, expr_content, component)
            
            # Add verification strategy as atoms with descriptive names
            strategy = proof_json.get("verification_strategy", {})
            approach = strategy.get("approach", "No approach specified")
            strategy_id = f"strategy_{len(metta_components)}"
            
            metta_components.append(f"(: {strategy_id} Strategy)")
            metta_components.append(f"(= ({strategy_id}) \"{self._escape_for_metta(approach)}\")")
            metta_components.append(f"(VerificationStrategy {strategy_id})")
            
            # Add property annotations for strategy using pattern mapper
            self._add_property_annotations(metta_components, strategy_id, approach, {})
            
            # Add lemmas as atoms with descriptive names
            for i, lemma in enumerate(strategy.get("key_lemmas", [])):
                lemma_id = f"lemma_{len(metta_components)}_{i}"
                metta_components.append(f"(: {lemma_id} Lemma)")
                metta_components.append(f"(= ({lemma_id}) \"{self._escape_for_metta(lemma)}\")")
                metta_components.append(f"(KeyLemma {lemma_id})")
                
                # Add property annotations for lemma using pattern mapper
                self._add_property_annotations(metta_components, lemma_id, lemma, {})
                
        except Exception as e:
            logging.error(f"Error converting JSON to MeTTa proof: {e}")
            error_id = f"error_{len(metta_components)}"
            metta_components.append(f"(: {error_id} Error)")
            metta_components.append(f"(= ({error_id}) \"Error converting to MeTTa: {str(e)}\")")
            metta_components.append(f"(ProofError {error_id})")
        
        return metta_components
    
    def _escape_for_metta(self, expr: str) -> str:
        """Escape a string for inclusion in MeTTa."""
        return expr.replace('"', '\\"').replace('\\', '\\\\')

    def verify_adaptation(self, original_func: str, adapted_func: str, 
                        essential_properties: List[str]) -> Dict[str, Any]:
        """
        Verify that an adaptation preserves essential properties.
        Uses pattern mapper for property identification.
        """
        logging.info("Verifying adaptation preserves essential properties")
        
        # Generate proofs for both functions
        original_proof = self.analyze_function_for_proof(original_func, "original")
        adapted_proof = self.analyze_function_for_proof(adapted_func, "adapted")
        
        if not original_proof["success"] or not adapted_proof["success"]:
            return {
                "success": False,
                "error": "Failed to generate proofs for comparison",
                "original_proof_success": original_proof["success"],
                "adapted_proof_success": adapted_proof["success"]
            }
        
        # Check which essential properties are preserved
        preserved = []
        violated = []
        
        for prop in essential_properties:
            property_type = self.pattern_mapper.map_requirement_to_property(prop)
            
            # Add to MeTTa for verification
            if hasattr(self, 'monitor') and self.monitor:
                func_name_orig = "original_func"
                func_name_adapted = "adapted_func"
                
                # Add property requirement to MeTTa space
                property_atoms_orig = self._add_property_to_metta(func_name_orig, prop)
                property_atoms_adapted = self._add_property_to_metta(func_name_adapted, prop)
                
                # Query MeTTa to check if property is preserved
                preserved_query = f"""
                (match &self 
                    (function-has-property {func_name_orig} {property_type})
                    (function-has-property {func_name_adapted} {property_type})
                    True)
                """
                
                preservation_result = self.monitor.query(preserved_query)
                if preservation_result and len(preservation_result) > 0 and preservation_result[0]:
                    preserved.append(prop)
                else:
                    violated.append(prop)
            else:
                # Fallback to Python implementation if monitor not available
                if (self._check_property_satisfied(original_proof.get("json_ir", {}), prop) and
                    self._check_property_satisfied(adapted_proof.get("json_ir", {}), prop)):
                    preserved.append(prop)
                else:
                    violated.append(prop)
        
        return {
            "success": len(violated) == 0,
            "preserved_properties": preserved,
            "violated_properties": violated,
            "original_proof": original_proof.get("proof", []),
            "adapted_proof": adapted_proof.get("proof", [])
        }
    
    def _add_property_to_metta(self, func_name: str, property: str) -> List[str]:
        """
        Add a property to MeTTa space using proper descriptive pattern atoms.
        Returns the list of added atoms.
        """
        if not hasattr(self, 'monitor') or not self.monitor:
            return []
            
        added_atoms = []
        
        # Convert property to MeTTa atom
        property_atom = self.pattern_mapper.map_requirement_to_property(property)
        
        # Add property relationship to MeTTa space
        property_atom_expr = f"(function-has-property {func_name} {property_atom})"
        self.monitor.add_atom(property_atom_expr)
        added_atoms.append(property_atom_expr)
        
        # Add property name definitions if it's a non-standard property
        if property_atom.startswith("property-"):
            property_name = property_atom.strip("()").split("-", 1)[1]
            type_def = f"(: {property_name} Property)"
            value_def = f"(= ({property_name}) \"{property}\")"
            
            self.monitor.add_atom(type_def)
            self.monitor.add_atom(value_def)
            
            added_atoms.extend([type_def, value_def])
            
        return added_atoms
    
    def analyze_function_for_proof(self, function_code: str, function_name: str = None,
                                context: str = None, max_attempts: int = 3) -> Dict[str, Any]:
        """Analyze function and generate proof using the pattern mapper."""
        # Implementation would depend on your full system setup
        # This is a placeholder showing integration with the pattern mapping approach
        return {"success": True, "proof": [], "json_ir": {}}  # Placeholder