#!/usr/bin/env python3
"""
Enhanced Evolution Integration
Bridges semantic evolution with existing MeTTa donor generation system
"""

from typing import List, Dict, Any, Optional, Callable, Union
import inspect
import textwrap

try:
    from metta_generator.evolution.semantic_evolution import SemanticEvolutionEngine
    SEMANTIC_EVOLUTION_AVAILABLE = True
except ImportError as e:
    print(f"Semantic evolution not available: {e}")
    SEMANTIC_EVOLUTION_AVAILABLE = False

class EnhancedEvolutionIntegrator:
    """Integrates semantic evolution with existing donor generation"""
    
    def __init__(self, metta_space=None, reasoning_engine=None):
        self.metta_space = metta_space
        self.reasoning_engine = reasoning_engine
        self.semantic_engine = None
        
        if SEMANTIC_EVOLUTION_AVAILABLE:
            self.semantic_engine = SemanticEvolutionEngine(
                metta_space=metta_space,
                reasoning_engine=reasoning_engine,
                population_size=12,  # Smaller for integration
                max_generations=6    # Faster for integration
            )
            print("  Enhanced evolution integrator: Semantic evolution enabled")
        else:
            print("  Enhanced evolution integrator: Semantic evolution not available")
    
    def is_semantic_evolution_available(self) -> bool:
        """Check if semantic evolution is available"""
        return SEMANTIC_EVOLUTION_AVAILABLE and self.semantic_engine is not None
    
    def generate_semantic_donors(self, func: Union[Callable, str], 
                                function_type: str = "search") -> List[Dict[str, Any]]:
        """Generate donors using semantic evolution"""
        if not self.is_semantic_evolution_available():
            return []
        
        print(f"    Generating semantic evolution donors for {function_type} function...")
        
        try:
            # Prepare function for semantic evolution
            if isinstance(func, str):
                # Extract callable from string code
                reference_func = self._extract_callable_from_code(func)
                if not reference_func:
                    print("      Failed to extract callable from code string")
                    return []
            else:
                reference_func = func
            
            # Setup semantic evolution for this function
            self.semantic_engine.setup_for_function(reference_func)
            
            # Determine target semantics based on function analysis
            target_semantics = self._analyze_function_semantics(reference_func, function_type)
            
            # Run semantic evolution
            results = self.semantic_engine.evolve_solutions(target_semantics)
            
            print(f"      Generated {len(results)} semantic evolution candidates")
            return results
            
        except Exception as e:
            print(f"      Semantic evolution failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _extract_callable_from_code(self, code: str) -> Optional[Callable]:
        """Extract callable function from code string"""
        try:
            # Execute the code to get the function
            exec_globals = {}
            exec(code, exec_globals)
            
            # Find the main function (not helper functions)
            main_function = None
            for name, obj in exec_globals.items():
                if callable(obj) and not name.startswith('_'):
                    if main_function is None or len(inspect.signature(obj).parameters) >= 2:
                        main_function = obj
            
            return main_function
        except Exception as e:
            print(f"        Error extracting callable: {e}")
            return None
    
    def _analyze_function_semantics(self, func: Callable, function_type: str) -> Dict[str, Any]:
        """Analyze function to determine target semantics"""
        semantics = {
            "purpose": "generic",
            "input_constraints": ["input_received"],
            "output_spec": "processed_result"
        }
        
        # Analyze function name for semantic clues
        func_name = func.__name__.lower()
        
        if any(keyword in func_name for keyword in ["max", "maximum", "largest", "highest"]):
            semantics["purpose"] = "maximize"
            semantics["output_spec"] = "maximum_element"
        elif any(keyword in func_name for keyword in ["min", "minimum", "smallest", "lowest"]):
            semantics["purpose"] = "minimize"
            semantics["output_spec"] = "minimum_element"
        elif any(keyword in func_name for keyword in ["find", "search", "locate", "get"]):
            semantics["purpose"] = "search"
            semantics["output_spec"] = "found_element"
        elif any(keyword in func_name for keyword in ["sum", "total", "aggregate"]):
            semantics["purpose"] = "aggregate"
            semantics["output_spec"] = "aggregated_result"
        elif any(keyword in func_name for keyword in ["sort", "order", "arrange"]):
            semantics["purpose"] = "sort"
            semantics["output_spec"] = "sorted_collection"
        elif any(keyword in func_name for keyword in ["transform", "convert", "map"]):
            semantics["purpose"] = "transform"
            semantics["output_spec"] = "transformed_collection"
        elif any(keyword in func_name for keyword in ["clean", "normalize", "format"]):
            semantics["purpose"] = "normalize"
            semantics["output_spec"] = "normalized_result"
        elif any(keyword in func_name for keyword in ["average", "mean"]):
            semantics["purpose"] = "average"
            semantics["output_spec"] = "average_value"
        
        # Analyze function signature for constraints
        try:
            sig = inspect.signature(func)
            param_count = len(sig.parameters)
            
            if param_count >= 3:
                # Likely a range-based function
                semantics["input_constraints"].extend(["valid_indices", "range_bounds"])
            
            if param_count == 1:
                # Single parameter, likely collection processing
                semantics["input_constraints"].append("collection_input")
            
        except Exception:
            pass
        
        # Analyze function type hint
        if function_type == "search":
            semantics["input_constraints"].extend(["searchable_collection", "search_criteria"])
        elif function_type == "sort":
            semantics["input_constraints"].append("comparable_elements")
        elif function_type == "transform":
            semantics["input_constraints"].append("transformable_elements")
        elif function_type == "aggregate":
            semantics["input_constraints"].append("numeric_elements")
        
        return semantics
    
    def integrate_with_existing_candidates(self, existing_candidates: List[Dict[str, Any]], 
                                         semantic_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Integrate semantic evolution candidates with existing donor candidates"""
        if not semantic_candidates:
            return existing_candidates
        
        print(f"    Integrating {len(semantic_candidates)} semantic candidates with {len(existing_candidates)} existing candidates")
        
        # Combine candidates
        combined_candidates = existing_candidates.copy()
        
        # Add semantic candidates with proper integration
        for semantic_candidate in semantic_candidates:
            # Ensure compatibility with existing format
            integrated_candidate = self._ensure_candidate_compatibility(semantic_candidate)
            combined_candidates.append(integrated_candidate)
        
        # Re-rank all candidates
        ranked_candidates = self._rank_integrated_candidates(combined_candidates)
        
        print(f"    Integration complete: {len(ranked_candidates)} total candidates")
        return ranked_candidates
    
    def _ensure_candidate_compatibility(self, semantic_candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure semantic candidate is compatible with existing format"""
        # Most fields should already be compatible, but ensure all required fields exist
        compatible_candidate = semantic_candidate.copy()
        
        # Ensure all required fields exist
        required_fields = [
            "name", "description", "code", "strategy", "pattern_family",
            "data_structures_used", "operations_used", "metta_derivation",
            "confidence", "final_score", "properties", "complexity_estimate",
            "applicability_scope", "generator_used"
        ]
        
        for field in required_fields:
            if field not in compatible_candidate:
                # Provide sensible defaults
                defaults = {
                    "pattern_family": "evolved",
                    "data_structures_used": ["list"],
                    "operations_used": ["evolution"],
                    "metta_derivation": ["(semantic-evolution-generated)"],
                    "properties": ["evolved"],
                    "complexity_estimate": "same",
                    "applicability_scope": "medium",
                    "generator_used": "SemanticEvolutionEngine"
                }
                compatible_candidate[field] = defaults.get(field, "unknown")
        
        # Add semantic evolution marker
        if "semantic-evolution" not in compatible_candidate["properties"]:
            compatible_candidate["properties"].append("semantic-evolution")
        
        return compatible_candidate
    
    def _rank_integrated_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank integrated candidates considering semantic evolution quality"""
        def candidate_score(candidate):
            base_score = candidate.get("final_score", candidate.get("confidence", 0.5))
            
            # Boost semantic evolution candidates if they have good test results
            if candidate.get("strategy") == "semantic_evolution":
                metadata = candidate.get("semantic_evolution_metadata", {})
                correctness = metadata.get("correctness_score", 0.5)
                
                # Significant boost for high correctness
                if correctness > 0.8:
                    base_score += 0.15
                elif correctness > 0.6:
                    base_score += 0.1
                elif correctness > 0.4:
                    base_score += 0.05
                
                # Boost for semantic validation
                if metadata.get("semantic_validation", False):
                    base_score += 0.05
                
                # Boost for comprehensive testing
                test_results = metadata.get("test_results", {})
                if test_results.get("passed", 0) > test_results.get("failed", 0):
                    base_score += 0.05
            
            return min(1.0, base_score)
        
        # Sort by enhanced score
        return sorted(candidates, key=candidate_score, reverse=True)
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        stats = {
            "semantic_evolution_available": self.is_semantic_evolution_available(),
            "engine_initialized": self.semantic_engine is not None
        }
        
        if self.semantic_engine:
            stats.update({
                "population_size": self.semantic_engine.population_size,
                "max_generations": self.semantic_engine.max_generations,
                "evolution_summary": self.semantic_engine.get_evolution_summary() if hasattr(self.semantic_engine, 'get_evolution_summary') else {}
            })
        
        return stats

def integrate_semantic_evolution_with_base_generator(base_generator, enable_semantic=True):
    """Integration function to add semantic evolution to base generator"""
    if not enable_semantic or not SEMANTIC_EVOLUTION_AVAILABLE:
        return base_generator
    
def integrate_semantic_evolution_with_base_generator(base_generator, enable_semantic=True):
    """Integration function to add semantic evolution to base generator"""
    if not enable_semantic or not SEMANTIC_EVOLUTION_AVAILABLE:
        return base_generator
    
    # Add semantic evolution capability
    base_generator.semantic_integrator = EnhancedEvolutionIntegrator(
        metta_space=base_generator.metta_space,
        reasoning_engine=base_generator.reasoning_engine
    )
    
    # Store original generate_donors_from_function method
    base_generator._original_generate_donors = base_generator.generate_donors_from_function
    
    def enhanced_generate_donors_from_function(func, strategies=None, use_semantic_evolution=True):
        """Enhanced donor generation with semantic evolution integration"""
        print("  Using enhanced donor generation with semantic evolution integration")
        
        # Generate candidates using original method
        original_candidates = base_generator._original_generate_donors(func, strategies)
        
        if not use_semantic_evolution or not base_generator.semantic_integrator.is_semantic_evolution_available():
            print("    Semantic evolution not used, returning original candidates")
            return original_candidates
        
        # Determine function type for semantic evolution
        function_type = "search"  # Default
        if callable(func):
            func_name = func.__name__.lower()
            if any(keyword in func_name for keyword in ["sort", "order"]):
                function_type = "sort"
            elif any(keyword in func_name for keyword in ["transform", "convert", "clean"]):
                function_type = "transform"
            elif any(keyword in func_name for keyword in ["sum", "average", "aggregate"]):
                function_type = "aggregate"
        
        # Generate semantic evolution candidates
        semantic_candidates = base_generator.semantic_integrator.generate_semantic_donors(func, function_type)
        
        # Integrate candidates
        integrated_candidates = base_generator.semantic_integrator.integrate_with_existing_candidates(
            original_candidates, semantic_candidates
        )
        
        return integrated_candidates
    
    # Replace the method
    base_generator.generate_donors_from_function = enhanced_generate_donors_from_function
    
    return base_generator

def create_enhanced_evolution_demo():
    """Create a demonstration of the enhanced evolution integration"""
    print("=== Enhanced Evolution Integration Demo ===")
    
    # Test function
    def find_max_in_range(numbers, start_idx, end_idx):
        """Find the maximum value in a list within a specific range."""
        if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
            return None
        
        max_val = numbers[start_idx]
        for i in range(start_idx + 1, end_idx):
            if numbers[i] > max_val:
                max_val = numbers[i]
        
        return max_val
    
    try:
        # Create integrator
        integrator = EnhancedEvolutionIntegrator()
        
        if integrator.is_semantic_evolution_available():
            print("Semantic evolution is available - running demo")
            
            # Generate semantic donors
            semantic_candidates = integrator.generate_semantic_donors(find_max_in_range, "search")
            
            print(f"Generated {len(semantic_candidates)} semantic candidates")
            
            # Show top candidate
            if semantic_candidates:
                best_candidate = semantic_candidates[0]
                metadata = best_candidate.get("semantic_evolution_metadata", {})
                
                print(f"\nBest Candidate: {best_candidate['name']}")
                print(f"  Final Score: {best_candidate['final_score']:.3f}")
                print(f"  Correctness: {metadata.get('correctness_score', 0):.3f}")
                print(f"  Semantic Roles: {metadata.get('semantic_roles', [])}")
                
                # Show code
                print(f"  Generated Code:")
                for i, line in enumerate(best_candidate['code'].split('\n')[:10], 1):
                    print(f"    {i:2d}: {line}")
            
            # Show integration stats
            stats = integrator.get_integration_stats()
            print(f"\nIntegration Stats: {stats}")
            
        else:
            print("Semantic evolution is not available")
            
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_enhanced_evolution_demo()