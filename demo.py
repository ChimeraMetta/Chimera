import os
import logging
from typing import Dict, List, Any

from hyperon import *
from dynamic_monitor import DynamicMonitor
from static_analyzer import decompose_function
from impl_generator import ProofGuidedImplementationGenerator
from proofs.verifier import MeTTaPropertyVerifier
from proofs.analyzer import ImmuneSystemProofAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("proof_system")

def integrate_with_immune_system(analyzer):
    """
    Integrate the proof-guided implementation generator with ImmuneSystemProofAnalyzer.
    
    Args:
        analyzer: ImmuneSystemProofAnalyzer instance
        
    Returns:
        Enhanced analyzer
    """
    # Add property verifier
    analyzer.property_verifier = MeTTaPropertyVerifier(analyzer.monitor)
    
    # Add implementation generator
    analyzer.implementation_generator = ProofGuidedImplementationGenerator(
        analyzer, 
        model_name=analyzer.model_name, 
        api_key=analyzer.api_key
    )
    
    # Load MeTTa ontology rules if not already loaded
    load_metta_ontology(analyzer.monitor)
    
    # Add method to generate verified alternatives
    def generate_verified_alternatives(self, function_code, function_name, count=3):
        """
        Generate and verify alternative implementations of a function.
        
        Args:
            function_code: Original function source code
            function_name: Name of the function
            count: Number of alternatives to generate
            
        Returns:
            List of alternative implementations with verification results
        """
        return self.implementation_generator.batch_generate_alternatives(
            function_code, function_name, count
        )
    
    # Add method to select the best alternative based on criteria
    def select_best_alternative(self, alternatives, criteria="property_preservation"):
        """
        Select the best alternative implementation based on criteria.
        
        Args:
            alternatives: List of alternative implementations
            criteria: Selection criteria (property_preservation, efficiency, simplicity)
            
        Returns:
            Best alternative implementation
        """
        if not alternatives:
            return None
        
        # Filter successful alternatives
        successful = [alt for alt in alternatives if alt.get("success", False)]
        if not successful:
            return None
        
        # Select based on criteria
        if criteria == "property_preservation":
            # Select the alternative with all properties preserved
            preserved = [alt for alt in successful 
                       if alt.get("verification_result", {}).get("properties_preserved", False)]
            return preserved[0] if preserved else successful[0]
            
        elif criteria == "efficiency":
            # Select alternatives with the "optimized" strategy
            optimized = [alt for alt in successful if alt.get("strategy") == "optimized"]
            return optimized[0] if optimized else successful[0]
            
        elif criteria == "simplicity":
            # Select alternatives with the "simplified" strategy
            simplified = [alt for alt in successful if alt.get("strategy") == "simplified"]
            return simplified[0] if simplified else successful[0]
            
        # Default to first successful alternative
        return successful[0]
    
    # Add method for adaptive donor selection using verified alternatives
    def select_verified_donor(self, target_function, candidate_functions, target_properties):
        """
        Select a verified donor function from candidates that satisfies target properties.
        
        Args:
            target_function: Function that needs a donor
            candidate_functions: List of candidate donor functions
            target_properties: List of properties the donor must satisfy
            
        Returns:
            Selected donor function with verification results
        """
        # Get property atoms from target properties using ontology
        property_atoms = self._map_requirements_to_properties(target_properties)
        
        # First, identify potential donors based on proof properties
        potential_donors = []
        
        for i, func_code in enumerate(candidate_functions):
            # Generate proof for candidate function
            proof_result = self.analyze_function_for_proof(
                func_code, f"candidate_{i}"
            )
            
            if not proof_result["success"]:
                continue
            
            # Use MeTTa to check if this candidate satisfies target properties
            func_id = f"candidate_{i}"
            query = f"""
            (match &self 
               (suitable-donor-candidate {func_id} {self._format_property_list(property_atoms)})
               True)
            """
            
            results = self.monitor.query(query)
            is_suitable = len(results) > 0
            
            if is_suitable:
                # Calculate compatibility score based on property coverage
                compatibility_score = self._calculate_compatibility_score(
                    proof_result.get("proof", []), 
                    property_atoms
                )
                
                potential_donors.append({
                    "function": func_code,
                    "proof": proof_result.get("proof", []),
                    "compatibility_score": compatibility_score,
                    "properties_satisfied": property_atoms
                })
        
        # If no potential donors found, return None
        if not potential_donors:
            return None
        
        # Sort donors by compatibility score
        potential_donors.sort(key=lambda d: d["compatibility_score"], reverse=True)
        
        # For the top donors, generate verified alternatives
        verified_donors = []
        
        for i, donor in enumerate(potential_donors[:3]):  # Top 3 donors
            # Extract donor code
            donor_code = donor["function"]
            donor_name = f"verified_donor_{i}"
            
            # Generate verified alternatives
            alternatives = self.generate_verified_alternatives(
                donor_code, donor_name, count=2
            )
            
            # Add to verified donors
            verified_donors.append({
                "original_donor": donor,
                "alternatives": alternatives,
                "compatibility_score": donor["compatibility_score"]
            })
        
        # Select the best donor based on compatibility score and alternatives
        best_donor = max(verified_donors, key=lambda d: d["compatibility_score"])
        
        # Select the best alternative for the best donor
        best_alternative = self.select_best_alternative(
            best_donor["alternatives"], "property_preservation"
        )
        
        return {
            "donor": best_donor["original_donor"],
            "alternative": best_alternative,
            "compatibility_score": best_donor["compatibility_score"]
        }
    
    def _map_requirements_to_properties(self, requirements):
        """Map donor requirements to property atoms using the ontology."""
        property_atoms = []
        
        for req in requirements:
            # Clean the requirement string
            clean_req = req.lower().strip()
            
            # Query the ontology for property mapping
            for req_type in ["bounds-requirement", "index-requirement", 
                           "order-requirement", "sorted-requirement",
                           "null-requirement", "empty-requirement",
                           "termination-requirement", "loop-requirement",
                           "error-requirement", "not-found-requirement"]:
                query = f"""
                (match &self 
                   (donor-requirement-maps {req_type} $property)
                   $property)
                """
                
                results = self.monitor.query(query)
                if results:
                    property_atoms.extend(results)
                    break
            
            # If no mapping found, use default property types based on keywords
            if "bound" in clean_req or "index" in clean_req:
                property_atoms.append("bound-check")
            elif "order" in clean_req or "sort" in clean_req:
                property_atoms.append("ordering-check")
            elif "null" in clean_req or "empty" in clean_req:
                property_atoms.append("null-check")
            elif "terminat" in clean_req or "loop" in clean_req:
                property_atoms.append("termination-guarantee")
            elif "error" in clean_req or "not found" in clean_req:
                property_atoms.append("error-handling")
        
        return list(set(property_atoms))  # Remove duplicates
    
    def _format_property_list(self, properties):
        """Format a list of properties for MeTTa queries."""
        if not properties:
            return "()"
        
        return f"({' '.join(properties)})"
    
    def _calculate_compatibility_score(self, proof_components, property_atoms):
        """Calculate compatibility score based on property coverage."""
        # Extract properties from proof components
        if hasattr(self, "property_verifier"):
            extracted_properties = set(self.property_verifier.extract_properties_from_components(proof_components))
            
            # Calculate coverage
            required_properties = set(property_atoms)
            covered_properties = extracted_properties.intersection(required_properties)
            
            return len(covered_properties) / len(required_properties) if required_properties else 0.0
        
        # Fallback if property verifier not available
        return 0.8  # Default score
    
    # Add methods to analyzer
    setattr(analyzer.__class__, "generate_verified_alternatives", generate_verified_alternatives)
    setattr(analyzer.__class__, "select_best_alternative", select_best_alternative)
    setattr(analyzer.__class__, "select_verified_donor", select_verified_donor)
    setattr(analyzer.__class__, "_map_requirements_to_properties", _map_requirements_to_properties)
    setattr(analyzer.__class__, "_format_property_list", _format_property_list)
    setattr(analyzer.__class__, "_calculate_compatibility_score", _calculate_compatibility_score)
    
    return analyzer

def load_metta_ontology(monitor):
    """
    Load MeTTa ontology rules for proof verification.
    
    Args:
        monitor: MeTTa monitor instance
        
    Returns:
        True if loaded successfully, False otherwise
    """
    # Check if ontology files exist
    ontology_files = [
        os.path.join(os.path.dirname(__file__), "metta", "proof_ontology.metta"),
        os.path.join(os.path.dirname(__file__), "metta", "proof_verification.metta")
    ]
    
    success = True
    
    for ontology_file in ontology_files:
        if os.path.exists(ontology_file):
            # Load ontology rules
            try:
                # Use monitor's load_metta_rules if available
                if hasattr(monitor, "load_metta_rules"):
                    file_success = monitor.load_metta_rules(ontology_file)
                else:
                    # Fallback to manual loading
                    with open(ontology_file, 'r') as f:
                        content = f.read()
                        
                    # Parse the atoms
                    atoms = monitor.metta.parse_all(content)
                    
                    # Add each atom to the space
                    for atom in atoms:
                        monitor.metta_space.add_atom(atom)
                    
                    file_success = True
                
                if file_success:
                    logger.info(f"Successfully loaded ontology from {ontology_file}")
                else:
                    logger.warning(f"Failed to load ontology from {ontology_file}")
                    success = False
                    
            except Exception as e:
                logger.error(f"Error loading ontology {ontology_file}: {e}")
                success = False
        else:
            logger.warning(f"Ontology file not found: {ontology_file}")
            success = False
    
    return success

def demo_verified_alternative_generation():
    """
    Demonstrate generating verified alternative implementations.
    """
    # Create MeTTa monitor
    monitor = DynamicMonitor()
    
    # Load ontology rules
    load_metta_ontology(monitor)
    
    # Binary search example
    binary_search_code = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
    api_key = os.environ.get("OPENAI_API_KEY")
    analyzer = ImmuneSystemProofAnalyzer(metta_space=monitor.metta_space, api_key=api_key)
    
    # Integrate with implementation generator
    analyzer = integrate_with_immune_system(analyzer)
    
    # Generate verified alternatives
    alternatives = analyzer.generate_verified_alternatives(
        binary_search_code,
        "binary_search",
        count=2
    )
    
    # Print results
    for i, alt in enumerate(alternatives):
        print(f"\n--- Alternative {i+1} ({alt.get('strategy', 'unknown')}) ---")
        print(f"Success: {alt.get('success', False)}")
        print(f"Properties preserved: {alt.get('verification_result', {}).get('properties_preserved', False)}")
        print("\nCode:")
        print(alt.get("alternative_function", "No code generated"))
        
    # Select best alternative
    best_alt = analyzer.select_best_alternative(alternatives)
    if best_alt:
        print("\n--- Best Alternative ---")
        print(f"Strategy: {best_alt.get('strategy', 'unknown')}")
        print("\nCode:")
        print(best_alt.get("alternative_function", "No code selected"))

    # Normalize phone numbers example
    normalize_phone_numbers_code = """
def normalize_phone_numbers(text):
    import re
    
    # Find all potential phone numbers
    pattern = r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
    matches = re.findall(pattern, text)
    
    # Normalize each match
    normalized = []
    for match in matches:
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', match)
        
        # Ensure 10 digits (strip leading 1 if present, will add it back later)
        if len(digits) == 11 and digits.startswith('1'):
            digits = digits[1:]
        
        # Skip invalid numbers
        if len(digits) != 10:
            continue
            
        # Format as E.164 (+1XXXXXXXXXX)
        normalized.append(f"+1{digits}")
    
    return normalized
"""
    logger.info("\n--- Testing normalize_phone_numbers ---")
    alternatives_phone = analyzer.generate_verified_alternatives(
        normalize_phone_numbers_code,
        "normalize_phone_numbers",
        count=2
    )
    for i, alt in enumerate(alternatives_phone):
        print(f"\n--- Phone Alternative {i+1} ({alt.get('strategy', 'unknown')}) ---")
        print(f"Success: {alt.get('success', False)}")
        print(f"Properties preserved: {alt.get('verification_result', {}).get('properties_preserved', False)}")
        print("\nCode:")
        print(alt.get("alternative_function", "No code generated"))
    best_alt_phone = analyzer.select_best_alternative(alternatives_phone)
    if best_alt_phone:
        print("\n--- Best Phone Alternative ---")
        print(f"Strategy: {best_alt_phone.get('strategy', 'unknown')}")
        print("\nCode:")
        print(best_alt_phone.get("alternative_function", "No code selected"))

    # Weighted moving average example
    weighted_moving_average_code = """
def weighted_moving_average(data, window_size=5, weights=None):
    if not data:
        return []
    
    # Input validation
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    
    if window_size > len(data):
        window_size = len(data)
    
    # Default weights: linearly increasing importance
    if weights is None:
        weights = list(range(1, window_size + 1))
    
    # Ensure weights match window size
    if len(weights) != window_size:
        raise ValueError("Weights must have the same length as window size")
    
    # Calculate weight sum for normalization
    weight_sum = sum(weights)
    
    # Calculate weighted moving average
    result = []
    for i in range(len(data)):
        if i < window_size - 1:
            # For initial points, use available data only
            window = data[:i+1]
            curr_weights = weights[-(i+1):]
            curr_weight_sum = sum(curr_weights)
        else:
            # For later points, use full window
            window = data[i-(window_size-1):i+1]
            curr_weights = weights
            curr_weight_sum = weight_sum
        
        # Calculate weighted average for current window
        weighted_sum = sum(w * v for w, v in zip(curr_weights, window))
        result.append(weighted_sum / curr_weight_sum)
    
    return result
"""
    logger.info("\n--- Testing weighted_moving_average ---")
    alternatives_wma = analyzer.generate_verified_alternatives(
        weighted_moving_average_code,
        "weighted_moving_average",
        count=2
    )
    for i, alt in enumerate(alternatives_wma):
        print(f"\n--- WMA Alternative {i+1} ({alt.get('strategy', 'unknown')}) ---")
        print(f"Success: {alt.get('success', False)}")
        print(f"Properties preserved: {alt.get('verification_result', {}).get('properties_preserved', False)}")
        print("\nCode:")
        print(alt.get("alternative_function", "No code generated"))
    best_alt_wma = analyzer.select_best_alternative(alternatives_wma)
    if best_alt_wma:
        print("\n--- Best WMA Alternative ---")
        print(f"Strategy: {best_alt_wma.get('strategy', 'unknown')}")
        print("\nCode:")
        print(best_alt_wma.get("alternative_function", "No code selected"))

    # Validate credit card example
    validate_credit_card_code = """
def validate_credit_card(card_number):
    # Remove spaces and hyphens
    card_number = card_number.replace(' ', '').replace('-', '')
    
    # Check if string contains only digits
    if not card_number.isdigit():
        return False
    
    # Check length (most card numbers are 13-19 digits)
    if not (13 <= len(card_number) <= 19):
        return False
    
    # Luhn algorithm implementation
    # 1. Double every second digit from right to left
    # 2. If doubling results in a number > 9, subtract 9
    # 3. Sum all digits
    # 4. If sum is divisible by 10, card number is valid
    
    digits = [int(d) for d in card_number]
    checksum = 0
    
    for i in range(len(digits) - 1, -1, -1):
        digit = digits[i]
        
        # Double every second digit from right to left
        if (len(digits) - i) % 2 == 0:
            digit *= 2
            # If result is > 9, subtract 9
            if digit > 9:
                digit -= 9
        
        checksum += digit
    
    # Valid if sum is divisible by 10
    return checksum % 10 == 0
"""
    logger.info("\n--- Testing validate_credit_card ---")
    alternatives_cc = analyzer.generate_verified_alternatives(
        validate_credit_card_code,
        "validate_credit_card",
        count=2
    )
    for i, alt in enumerate(alternatives_cc):
        print(f"\n--- CC Alternative {i+1} ({alt.get('strategy', 'unknown')}) ---")
        print(f"Success: {alt.get('success', False)}")
        print(f"Properties preserved: {alt.get('verification_result', {}).get('properties_preserved', False)}")
        print("\nCode:")
        print(alt.get("alternative_function", "No code generated"))
    best_alt_cc = analyzer.select_best_alternative(alternatives_cc)
    if best_alt_cc:
        print("\n--- Best CC Alternative ---")
        print(f"Strategy: {best_alt_cc.get('strategy', 'unknown')}")
        print("\nCode:")
        print(best_alt_cc.get("alternative_function", "No code selected"))

# Remove extra newline
if __name__ == "__main__":
    demo_verified_alternative_generation()