import os
import logging
import sys
import re
from colorama import init, Fore, Style

# Initialize colorama
init()

# Import existing components
from hyperon import *
from reflectors.dynamic_monitor import DynamicMonitor
from reflectors.static_analyzer import decompose_file, decompose_function
from executors.impl_generator import ProofGuidedImplementationGenerator
from proofs.verifier import MeTTaPropertyVerifier
from proofs.analyzer import ImmuneSystemProofAnalyzer

# Custom formatter for colorful logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to different log levels"""
    
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)

# Configure logging
handler = logging.StreamHandler()
formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger("proof_system")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

ONTOLOGY_PATH = "metta/code_ontology.metta"
COMPLEXITY_THRESHOLD = 15  # Threshold for considering a function complex

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
    load_metta_proof_ontology(analyzer.monitor)
    
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
    
    # Add methods to analyzer
    setattr(analyzer.__class__, "generate_verified_alternatives", generate_verified_alternatives)
    setattr(analyzer.__class__, "select_best_alternative", select_best_alternative)
    
    return analyzer

def load_metta_proof_ontology(monitor):
    """
    Load MeTTa ontology rules for proof verification.
    
    Args:
        monitor: MeTTa monitor instance
        
    Returns:
        True if loaded successfully, False otherwise
    """
    # Check if ontology files exist
    ontology_files = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "metta", "proof_ontology.metta"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "metta", "proof_verification.metta")
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

def analyze_codebase(path, analyzer=None):
    """Analyze a Python file or directory of Python files."""
    if os.path.isfile(path) and path.endswith('.py'):
        # Analyze a single file
        analyze_file(path, analyzer)
    elif os.path.isdir(path):
        # Analyze all Python files in directory
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.py'):
                    analyze_file(os.path.join(root, file), analyzer)
    else:
        print(f"{Fore.RED}Invalid path or not a Python file: {path}{Style.RESET_ALL}")

def analyze_file(file_path, analyzer=None):
    """Analyze a single Python file and add to the ontology."""
    print(f"{Fore.CYAN}Analyzing {file_path}...{Style.RESET_ALL}")
    
    # Run static analysis
    analysis_result = decompose_file(file_path)
    
    # Add analysis results to MeTTa
    if "metta_atoms" in analysis_result and analysis_result["metta_atoms"]:
        atoms_added = 0
        for atom_str in analysis_result["metta_atoms"]:
            try:
                monitor.add_atom(atom_str)
                atoms_added += 1
            except Exception as e:
                print(f"{Fore.RED}Error adding atom {atom_str}: {e}{Style.RESET_ALL}")
                pass
                
        print(f"{Fore.GREEN}Added {atoms_added}/{len(analysis_result['metta_atoms'])} atoms from {file_path}{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}No MeTTa atoms generated for {file_path}{Style.RESET_ALL}")

def analyze_function_complexity_and_optimize(file_path, analyzer=None):
    """
    Analyze complexity of functions in a file and generate optimized alternatives
    for complex functions.
    """
    print(f"{Fore.CYAN}Analyzing complexity in {file_path} and generating optimized alternatives...{Style.RESET_ALL}")
    
    # Decompose the file
    result = decompose_file(file_path)
    
    if "error" in result and result["error"]:
        print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
        return
    
    atoms = result["metta_atoms"]
    
    # Extract function definitions
    function_defs = {}
    function_source = {}
    
    # Also extract function source code if available
    if "functions" in result:
        for func_info in result["functions"]:
            if "name" in func_info and "source" in func_info:
                function_source[func_info["name"]] = func_info["source"]
    
    # Extract from atoms as fallback
    for atom in atoms:
        if atom.startswith("(function-def "):
            parts = atom.strip("()").split()
            if len(parts) >= 5:
                func_name = parts[1]
                scope = " ".join(parts[2:-2])
                line_start = int(parts[-2])
                line_end = int(parts[-1])
                function_defs[func_name] = {
                    "scope": scope,
                    "line_start": line_start,
                    "line_end": line_end,
                    "operations": 0,
                    "loops": 0,
                    "calls": 0
                }
    
    # Count operations (bin_op)
    for atom in atoms:
        if atom.startswith("(bin-op "):
            parts = atom.strip("()").split()
            if len(parts) >= 6:
                op = parts[1]
                scope = " ".join(parts[4:-1])
                line = int(parts[-1])
                
                # Find which function this operation belongs to
                for func_name, func_info in function_defs.items():
                    if (scope == func_info["scope"] or 
                        scope.startswith(func_info["scope"]) or 
                        func_info["scope"] == "global") and \
                       line >= func_info["line_start"] and \
                       line <= func_info["line_end"]:
                        func_info["operations"] += 1
    
    # Count loops (both explicit and implicit)
    for atom in atoms:
        if atom.startswith("(loop-pattern ") or atom.startswith("(implicit-loop "):
            parts = atom.strip("()").split()
            if len(parts) >= 5:
                loop_id = parts[1]
                loop_type = parts[2]
                scope = " ".join(parts[3:-1])
                line = int(parts[-1])
                
                # Find which function this loop belongs to
                for func_name, func_info in function_defs.items():
                    if (scope == func_info["scope"] or 
                        scope.startswith(func_info["scope"]) or 
                        func_info["scope"] == "global") and \
                       line >= func_info["line_start"] and \
                       line <= func_info["line_end"]:
                        func_info["loops"] += 1
    
    # Count function calls
    for atom in atoms:
        if atom.startswith("(function-call "):
            parts = atom.strip("()").split()
            if len(parts) >= 5:
                callee = parts[1]
                args = parts[2]
                scope = " ".join(parts[3:-1])
                line = int(parts[-1])
                
                # Find which function this call belongs to
                for func_name, func_info in function_defs.items():
                    if (scope == func_info["scope"] or 
                        scope.startswith(func_info["scope"]) or 
                        func_info["scope"] == "global") and \
                       line >= func_info["line_start"] and \
                       line <= func_info["line_end"]:
                        func_info["calls"] += 1
    
    # Calculate overall complexity scores
    for func_name, func_info in function_defs.items():
        # Weighted score: operations + 3*loops + 0.5*calls
        func_info["score"] = (
            func_info["operations"] + 
            3 * func_info["loops"] + 
            0.5 * func_info["calls"]
        )
    
    # Sort functions by complexity score
    sorted_functions = sorted(
        function_defs.items(), 
        key=lambda x: x[1]["score"], 
        reverse=True
    )
    
    # Print complexity ranking
    print(f"\n{Fore.CYAN}=== Function Complexity Analysis ==={Style.RESET_ALL}")
    print(f"{Fore.GREEN}Function complexity ranking:{Style.RESET_ALL}")
    for i, (func_name, func_info) in enumerate(sorted_functions):
        print(f"{i+1}. {func_name}: score {func_info['score']:.1f} ({func_info['operations']} operations, {func_info['loops']} loops, {func_info['calls']} calls)")
        if i > 20:  # Only show top 20 functions
            print(f"{Fore.YELLOW}(... and more functions){Style.RESET_ALL}")
            break
    
    # Identify complex functions based on criteria
    complex_funcs = []
    for func_name, func_info in function_defs.items():
        if (func_info["operations"] > 10 or 
            func_info["loops"] > 2 or 
            func_info["score"] > COMPLEXITY_THRESHOLD):
            complex_funcs.append((func_name, func_info))
    
    # Sort complex functions by score
    complex_funcs.sort(key=lambda x: x[1]["score"], reverse=True)
    
    print(f"\n{Fore.CYAN}=== Complex Functions Detected ==={Style.RESET_ALL}")
    if complex_funcs:
        for i, (func_name, func_info) in enumerate(complex_funcs):
            print(f"{i+1}. {func_name}: score {func_info['score']:.1f} ({func_info['operations']} operations, {func_info['loops']} loops, {func_info['calls']} calls)")
    else:
        print(f"{Fore.GREEN}No complex functions detected{Style.RESET_ALL}")
    
    # Generate alternative implementations for complex functions
    if analyzer and complex_funcs:
        print(f"\n{Fore.CYAN}=== Generating Optimized Alternatives ==={Style.RESET_ALL}")
        
        for i, (func_name, func_info) in enumerate(complex_funcs[:5]):  # Limit to top 5 most complex functions
            print(f"{Fore.GREEN}\nGenerating alternatives for function: {func_name}{Style.RESET_ALL}")
            
            # Get function source code
            func_source = None
            if func_name in function_source:
                func_source = function_source[func_name]
            else:
                # Try to extract source from file
                try:
                    with open(file_path, 'r') as f:
                        file_content = f.read()
                        
                    # Extract function using line numbers
                    lines = file_content.split('\n')
                    start_line = func_info["line_start"] - 1  # 0-indexed
                    end_line = func_info["line_end"] - 1      # 0-indexed
                    
                    if 0 <= start_line < len(lines) and start_line <= end_line < len(lines):
                        func_source = '\n'.join(lines[start_line:end_line+1])
                except Exception as e:
                    logger.error(f"Error extracting source for {func_name}: {e}")
            
            if not func_source:
                print(f"{Fore.RED}Could not extract source code for {func_name}. Skipping...{Style.RESET_ALL}")
                continue
            
            # Generate alternatives
            try:
                alternatives = analyzer.generate_verified_alternatives(
                    func_source,
                    func_name,
                    count=2  # Generate 2 alternatives to keep processing time reasonable
                )
                
                # Analyze complexity of alternatives
                print(f"{Fore.GREEN}\nAnalyzing complexity of alternative implementations...{Style.RESET_ALL}")
                
                for j, alt in enumerate(alternatives):
                    if not alt.get("success", False) or not alt.get("alternative_function"):
                        continue
                        
                    alt_code = alt.get("alternative_function", "")
                    
                    # Analyze the alternative's complexity
                    try:
                        # Extract function name from the alternative implementation
                        alt_func_name = func_name  # Default to original name
                        func_def_match = re.match(r'def\s+([a-zA-Z0-9_]+)\s*\(', alt_code)
                        if func_def_match:
                            alt_func_name = func_def_match.group(1)
                        
                        # Use decompose_function to analyze the alternative
                        # decompose_function() accepts a string with function code directly
                        alt_analysis = decompose_function(alt_code)
                        
                        if alt_analysis and "metta_atoms" in alt_analysis:
                            # Create a new temporary MeTTa space for this analysis
                            temp_monitor = DynamicMonitor()
                            
                            # Add the atoms to the space
                            for atom_str in alt_analysis["metta_atoms"]:
                                try:
                                    temp_monitor.add_atom(atom_str)
                                except Exception as e:
                                    logger.debug(f"Error adding atom {atom_str}: {e}")
                            
                            # Calculate complexity metrics
                            operations = 0
                            loops = 0
                            calls = 0
                            
                            # Count binary operations
                            bin_ops = temp_monitor.query(f"(match &self (bin-op $op $left $right $scope $line) $op)")
                            operations = len(bin_ops)
                            
                            # Count loops
                            loop_patterns = temp_monitor.query(f"(match &self (loop-pattern $id $type $scope $line) $type)")
                            implicit_loops = temp_monitor.query(f"(match &self (implicit-loop $id $type $scope $line) $type)")
                            loops = len(loop_patterns) + len(implicit_loops)
                            
                            # Count function calls
                            func_calls = temp_monitor.query(f"(match &self (function-call $name $args $scope $line) $name)")
                            calls = len(func_calls)
                            
                            # Calculate complexity score using the same formula as the original
                            alt_score = operations + 3 * loops + 0.5 * calls
                            
                            # Store complexity information in the alternative
                            alt["complexity"] = {
                                "operations": operations,
                                "loops": loops,
                                "calls": calls,
                                "score": alt_score
                            }
                            
                            # Calculate complexity reduction percentage
                            original_score = func_info["score"]
                            if original_score > 0:  # Avoid division by zero
                                reduction_percentage = ((original_score - alt_score) / original_score) * 100
                                alt["complexity_reduction"] = reduction_percentage
                            else:
                                alt["complexity_reduction"] = 0
                            
                        else:
                            logger.warning(f"Could not analyze complexity of alternative {j+1} for {func_name}")
                            alt["complexity"] = {"score": 0, "operations": 0, "loops": 0, "calls": 0}
                            alt["complexity_reduction"] = 0
                            
                    except Exception as e:
                        logger.error(f"Error analyzing complexity of alternative {j+1}: {e}")
                        alt["complexity"] = {"score": 0, "operations": 0, "loops": 0, "calls": 0}
                        alt["complexity_reduction"] = 0
                
                # Print results with complexity analysis
                for j, alt in enumerate(alternatives):
                    print(f"{Fore.GREEN}\n--- Alternative {j+1} ({alt.get('strategy', 'unknown')}) ---{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}Success: {alt.get('success', False)}{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}Properties preserved: {alt.get('verification_result', {}).get('properties_preserved', False)}{Style.RESET_ALL}")
                    
                    # Print complexity metrics if available
                    if "complexity" in alt:
                        complexity = alt["complexity"]
                        print(f"{Fore.GREEN}Complexity score: {complexity['score']:.1f} ({complexity['operations']} operations, {complexity['loops']} loops, {complexity['calls']} calls){Style.RESET_ALL}")
                        
                        if "complexity_reduction" in alt:
                            print(f"{Fore.GREEN}Complexity reduction: {alt['complexity_reduction']:.1f}%{Style.RESET_ALL}")
                    
                    print(f"{Fore.GREEN}\nCode:{Style.RESET_ALL}")
                    print(alt.get("alternative_function", "No code generated"))
                
                # Sort alternatives by complexity reduction if available
                if all("complexity_reduction" in alt for alt in alternatives if alt.get("success", False)):
                    successful_alternatives = [alt for alt in alternatives if alt.get("success", False)]
                    successful_alternatives.sort(key=lambda x: x.get("complexity_reduction", 0), reverse=True)
                    
                    # Update alternatives with sorted list
                    alternatives = successful_alternatives
                
                # Select best alternative based on complexity reduction and property preservation
                best_alt = None
                if alternatives:
                    # First filter for property preservation
                    preserved_alternatives = [alt for alt in alternatives 
                                            if alt.get("success", False) and 
                                            alt.get("verification_result", {}).get("properties_preserved", False)]
                    
                    if preserved_alternatives:
                        # Then select the one with the highest complexity reduction
                        best_alt = max(preserved_alternatives, 
                                     key=lambda x: x.get("complexity_reduction", 0))
                    else:
                        # If none preserve properties, just use the highest complexity reduction
                        successful = [alt for alt in alternatives if alt.get("success", False)]
                        if successful:
                            best_alt = max(successful, 
                                         key=lambda x: x.get("complexity_reduction", 0))
                
                if best_alt:
                    print(f"{Fore.GREEN}\n--- Best Alternative ---{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}Strategy: {best_alt.get('strategy', 'unknown')}{Style.RESET_ALL}")
                    
                    # Print complexity metrics
                    if "complexity" in best_alt:
                        complexity = best_alt["complexity"]
                        print(f"{Fore.GREEN}Complexity score: {complexity['score']:.1f} ({complexity['operations']} operations, {complexity['loops']} loops, {complexity['calls']} calls){Style.RESET_ALL}")
                        
                        # Compare with original
                        old_score = func_info["score"]
                        print(f"{Fore.GREEN}Original complexity score: {old_score:.1f}{Style.RESET_ALL}")
                        
                        if "complexity_reduction" in best_alt:
                            reduction = best_alt["complexity_reduction"]
                            print(f"{Fore.GREEN}Actual complexity reduction: {reduction:.1f}%{Style.RESET_ALL}")
                    
                    print(f"{Fore.GREEN}\nCode:{Style.RESET_ALL}")
                    print(best_alt.get("alternative_function", "No code selected"))
                    
                    # Add suggestions for where to save optimized version
                    original_path = file_path
                    optimized_dir = os.path.join(os.path.dirname(original_path), "optimized")
                    optimized_path = os.path.join(optimized_dir, os.path.basename(original_path))
                    
                    print(f"{Fore.GREEN}\nTo use this optimized implementation:{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}1. Create directory: {optimized_dir}{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}2. Save to: {optimized_path}{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}3. Replace the original function with this optimized version{Style.RESET_ALL}")
                    
            except Exception as e:
                logger.error(f"Error generating alternatives for {func_name}: {e}")
                import traceback
                traceback.print_exc()

# Main functionality
if __name__ == "__main__":
    # Initialize MeTTa monitor
    monitor = DynamicMonitor()
    
    # Load the MeTTa reasoning rules
    monitor.load_metta_rules(ONTOLOGY_PATH)
    
    if len(sys.argv) < 2:
        print("Usage: python optimized_analyzer.py <path_to_file_or_directory> [api_key]")
        sys.exit(1)
    
    # Path to analyze
    path = sys.argv[1]
    
    # Get API key from command line or environment
    api_key = None
    if len(sys.argv) > 2:
        api_key = sys.argv[2]
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    # Initialize the analyzer if API key is available
    analyzer = None
    if api_key:
        try:
            analyzer = ImmuneSystemProofAnalyzer(metta_space=monitor.metta_space, api_key=api_key)
            analyzer = integrate_with_immune_system(analyzer)
            print("Initialized proof-guided implementation generator")
        except Exception as e:
            logger.error(f"Error initializing analyzer: {e}")
            print("Could not initialize proof-guided implementation generator. Will only analyze complexity.")
    else:
        print("No API key provided. Will only analyze complexity without generating alternatives.")
    
    # Run analysis
    analyze_codebase(path)
    
    # Analyze function complexity and generate alternatives
    if os.path.isfile(path) and path.endswith('.py'):
        analyze_function_complexity_and_optimize(path, analyzer)
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    analyze_function_complexity_and_optimize(file_path, analyzer)