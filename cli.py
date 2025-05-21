import argparse
import os
import sys
import logging
from typing import Union
from colorama import init, Fore, Style
import inquirer
from inquirer import themes
from io import StringIO

# Initialize colorama
init()

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

# Custom theme for inquirer that matches our color scheme
class ChimeraTheme(themes.GreenPassion):
    def __init__(self):
        super().__init__()
        self.Checkbox.selected_icon = f"{Fore.GREEN}✓{Style.RESET_ALL}"
        self.Checkbox.unselected_icon = " "
        self.Checkbox.selected = f"{Fore.GREEN}●{Style.RESET_ALL}"
        self.Checkbox.unselected = "○"
        self.Checkbox.selection_color = "green"
        self.Checkbox.selection_cursor = f"{Fore.GREEN}❯{Style.RESET_ALL}"

# --- Path Setup ---
_WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
_EXEC_DIR = os.path.join(_WORKSPACE_ROOT, 'executors')
if _EXEC_DIR not in sys.path:
    sys.path.insert(0, _EXEC_DIR)

# --- Imports from project modules (now in exec directory) ---
from executors import full_analyzer
from executors import complexity as complexity_analyzer_module
from reflectors.dynamic_monitor import DynamicMonitor
from proofs.analyzer import ImmuneSystemProofAnalyzer

def setup_colored_logging():
    """Setup colored logging configuration"""
    handler = logging.StreamHandler()
    formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

# --- Command Functions ---

def analyze_file_and_collect_complex_functions(file_path: str, complex_functions: dict) -> None:
    """
    Analyzes a Python file for function complexity and collects complex functions.
    
    Args:
        file_path (str): Path to the Python file to analyze
        complex_functions (dict): Dictionary to store complex functions found
    """
    # Get the complexity analysis results
    results = complexity_analyzer_module.analyze_function_complexity_and_optimize(file_path, None)  # Pass None to skip optimization
    
    # The results are printed to stdout by the analyzer, we need to parse them
    # Look for the "=== Complex Functions Detected ===" section in the output
    # Capture stdout
    old_stdout = sys.stdout
    captured_output = StringIO()
    sys.stdout = captured_output
    
    try:
        # Run the analysis again to capture the output
        complexity_analyzer_module.analyze_function_complexity_and_optimize(file_path, None)
        output = captured_output.getvalue()
        
        # Parse the output to find complex functions
        in_complex_section = False
        for line in output.split('\n'):
            if "=== Complex Functions Detected ===" in line:
                in_complex_section = True
                continue
            elif in_complex_section and line.strip() and not line.startswith('==='):
                # Parse lines like: "1. analyze_text: score 89.0 (29 operations, 15 loops, 30 calls)"
                try:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        func_name = parts[0].split('.')[-1].strip()
                        score_part = parts[1].split('(')[0].strip()
                        score = float(score_part.split()[-1])
                        complex_functions[func_name] = {
                            'file': file_path,
                            'complexity': score
                        }
                except (ValueError, IndexError):
                    continue
            elif in_complex_section and line.startswith('==='):
                in_complex_section = False
    finally:
        # Restore stdout
        sys.stdout = old_stdout

def run_summary_command(target_path: str):
    print(f"{Fore.CYAN}Running summary for: {target_path}{Style.RESET_ALL}")
    
    local_monitor = DynamicMonitor()
    
    # Construct absolute path to the ontology file relative to the exec directory
    # exec.full_analyzer.ONTOLOGY_PATH is like "metta/code_ontology.metta"
    ontology_file_path = os.path.join(_WORKSPACE_ROOT, full_analyzer.ONTOLOGY_PATH)
    if not os.path.exists(ontology_file_path):
        print(f"{Fore.YELLOW}Warning: Ontology file not found at {ontology_file_path}. Analysis might be incomplete.{Style.RESET_ALL}")
    else:
        local_monitor.load_metta_rules(ontology_file_path)

    # Temporarily set the global monitor instance for functions in full_analyzer module
    original_global_monitor_full = getattr(full_analyzer, 'monitor', None)
    full_analyzer.monitor = local_monitor
    
    try:
        # Replicate the __main__ execution flow of full_analyzer.py
        print(f"{Fore.GREEN}Analyzing codebase structure with full_analyzer.analyze_codebase...{Style.RESET_ALL}")
        full_analyzer.analyze_codebase(target_path)
        
        print(f"{Fore.GREEN}Analyzing type safety...{Style.RESET_ALL}")
        full_analyzer.analyze_type_safety() # Uses global full_analyzer.monitor
        
        print(f"{Fore.GREEN}Analyzing temporal evolution...{Style.RESET_ALL}")
        # analyze_temporal_evolution in full_analyzer.py can take monitor explicitly
        full_analyzer.analyze_temporal_evolution(target_path, local_monitor)
        
        print(f"{Fore.GREEN}Analyzing function complexity (static)...{Style.RESET_ALL}")
        full_analyzer.analyze_function_complexity(target_path) # Uses global full_analyzer.monitor
        
        # find_function_relationships was commented out in original full_analyzer.py __main__
        # print("Finding function relationships...")
        # full_analyzer.find_function_relationships() 

        print(f"{Fore.GREEN}Analyzing function call relationships (detailed)...{Style.RESET_ALL}")
        full_analyzer.analyze_function_call_relationships(target_path) # Uses global full_analyzer.monitor indirectly via decompose_file
        
        print(f"{Fore.GREEN}Finding type relationships...{Style.RESET_ALL}")
        full_analyzer.find_type_relationships() # Uses global full_analyzer.monitor
        
        print(f"{Fore.GREEN}Finding class relationships...{Style.RESET_ALL}")
        full_analyzer.find_class_relationships() # Uses global full_analyzer.monitor
        
        print(f"{Fore.GREEN}Finding module relationships...{Style.RESET_ALL}")
        full_analyzer.find_module_relationships() # Uses global full_analyzer.monitor
        
        print(f"{Fore.GREEN}Finding operation patterns...{Style.RESET_ALL}")
        full_analyzer.find_operation_patterns() # Uses global full_analyzer.monitor
        
        print(f"{Fore.GREEN}Analyzing structural patterns...{Style.RESET_ALL}")
        full_analyzer.analyze_structural_patterns() # Uses global full_analyzer.monitor
        
        print(f"{Fore.GREEN}Analyzing domain concepts...{Style.RESET_ALL}")
        full_analyzer.analyze_domain_concepts() # Uses global full_analyzer.monitor

    except Exception as e:
        logging.error(f"An error occurred during summary analysis: {e}")
        logging.exception("Full traceback for summary analysis error:")
    finally:
        # Restore original monitor attribute in full_analyzer module
        if original_global_monitor_full is not None:
            full_analyzer.monitor = original_global_monitor_full
        elif hasattr(full_analyzer, 'monitor'):
            delattr(full_analyzer, 'monitor')

    print(f"{Fore.CYAN}Summary analysis for {target_path} complete.{Style.RESET_ALL}")

def run_analyze_command(target_path: str, api_key: Union[str, None] = None):
    print(f"{Fore.CYAN}Running 'analyze' command for: {target_path} (API key: {'Provided' if api_key else 'Not provided'}){Style.RESET_ALL}")
    
    local_monitor = DynamicMonitor()

    # Construct absolute path to the ontology file relative to the exec directory
    # exec.complexity_analyzer_module.ONTOLOGY_PATH is like "metta/code_ontology.metta"
    ontology_file_path = os.path.join(_WORKSPACE_ROOT, complexity_analyzer_module.ONTOLOGY_PATH)
    if not os.path.exists(ontology_file_path):
        print(f"{Fore.YELLOW}Warning: Ontology file not found at {ontology_file_path}. Analysis might be incomplete.{Style.RESET_ALL}")
    else:
        local_monitor.load_metta_rules(ontology_file_path)

    # Temporarily set the global monitor for complexity_analyzer_module
    original_global_monitor_complexity = getattr(complexity_analyzer_module, 'monitor', None)
    complexity_analyzer_module.monitor = local_monitor

    # Setup logger for complexity_analyzer_module if it's not already configured by its import
    # This ensures its internal logging works as expected.
    if not any(handler for handler in complexity_analyzer_module.logger.handlers):
        handler = logging.StreamHandler()
        formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        complexity_analyzer_module.logger.addHandler(handler)
        complexity_analyzer_module.logger.setLevel(logging.INFO)
        complexity_analyzer_module.logger.info("CLI re-initialized logger for complexity_analyzer_module.")

    analyzer_instance_for_complexity = None
    if api_key:
        # === DIAGNOSTIC PRINT START ===
        print(f"{Fore.BLUE}[DEBUG] API key provided, attempting to initialize ImmuneSystemProofAnalyzer...{Style.RESET_ALL}")
        # === DIAGNOSTIC PRINT END ===
        try:
            complexity_analyzer_module.logger.info("Initializing proof-guided implementation generator via CLI...")
            # The ImmuneSystemProofAnalyzer needs the *global* monitor from its module context for some ops if not passed around carefully
            # Here, we pass our local_monitor (which is set as global for the module) to its constructor.
            analyzer_instance_for_complexity = ImmuneSystemProofAnalyzer(metta_space=local_monitor.metta_space, api_key=api_key)
            analyzer_instance_for_complexity = complexity_analyzer_module.integrate_with_immune_system(analyzer_instance_for_complexity)
            complexity_analyzer_module.logger.info("Proof-guided implementation generator initialized successfully via CLI.")
        except Exception as e:
            complexity_analyzer_module.logger.error(f"Error initializing ImmuneSystemProofAnalyzer via CLI: {e}")
            complexity_analyzer_module.logger.exception("Full traceback for analyzer initialization error:")
            print(f"{Fore.YELLOW}Could not initialize proof-guided implementation generator. Proceeding with complexity analysis only.{Style.RESET_ALL}")
            # Ensure analyzer_instance_for_complexity remains None if initialization fails
            analyzer_instance_for_complexity = None 
    else:
        print(f"{Fore.YELLOW}No API key provided. Proceeding with complexity analysis only (no alternative generation).{Style.RESET_ALL}")

    # === DIAGNOSTIC PRINT START ===
    print(f"{Fore.BLUE}[DEBUG] Analyzer instance before calling analyze_function_complexity_and_optimize: {type(analyzer_instance_for_complexity)}{Style.RESET_ALL}")
    # === DIAGNOSTIC PRINT END ===

    try:
        # Replicate the __main__ execution flow of complexity.py
        print(f"{Fore.GREEN}Analyzing codebase with complexity_analyzer_module.analyze_codebase...{Style.RESET_ALL}")
        # complexity_analyzer_module.analyze_codebase calls analyze_file, which uses global monitor
        complexity_analyzer_module.analyze_codebase(target_path) # analyzer_instance_for_complexity is not used by this specific analyze_codebase

        print(f"{Fore.GREEN}Analyzing function complexity and optimizing for {target_path}...{Style.RESET_ALL}")
        
        # Dictionary to store complex functions found during analysis
        complex_functions = {}
        
        if os.path.isfile(target_path) and target_path.endswith('.py'):
            analyze_file_and_collect_complex_functions(target_path, complex_functions)
        elif os.path.isdir(target_path):
            for root, _, files_in_dir in os.walk(target_path):
                for f_name in files_in_dir:
                    if f_name.endswith('.py'):
                        file_path_to_analyze = os.path.join(root, f_name)
                        print(f"{Fore.GREEN}Analyzing: {file_path_to_analyze}{Style.RESET_ALL}")
                        analyze_file_and_collect_complex_functions(file_path_to_analyze, complex_functions)
        else:
            print(f"{Fore.RED}Path is not a valid Python file or directory: {target_path}{Style.RESET_ALL}")
            return

        # If we have complex functions and an API key, offer to generate alternatives
        if complex_functions and analyzer_instance_for_complexity:
            print(f"\n{Fore.CYAN}Found {len(complex_functions)} complex functions that could be optimized:{Style.RESET_ALL}")
            
            # Create checkbox choices
            choices = [
                (f"{func_name} (Complexity: {info['complexity']}, File: {info['file']})", func_name)
                for func_name, info in complex_functions.items()
            ]
            
            # Ask user to select functions
            questions = [
                inquirer.Checkbox('selected_functions',
                                message="Select functions to optimize (use space to select, enter to confirm)",
                                choices=choices,
                                theme=ChimeraTheme())
            ]
            
            answers = inquirer.prompt(questions)
            
            if answers and answers['selected_functions']:
                for selected_func in answers['selected_functions']:
                    func_info = complex_functions[selected_func]
                    print(f"\n{Fore.GREEN}Generating alternative implementation for {selected_func}...{Style.RESET_ALL}")
                    
                    # Extract the function source code
                    try:
                        with open(func_info['file'], 'r') as f:
                            file_content = f.read()
                            
                        # Find the function definition
                        import ast
                        tree = ast.parse(file_content)
                        func_source = None
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef) and node.name == selected_func:
                                # Get the function's source code
                                func_source = ast.get_source_segment(file_content, node)
                                break
                        
                        if func_source:
                            # Generate alternatives for just this function
                            alternatives = analyzer_instance_for_complexity.generate_verified_alternatives(
                                func_source,
                                selected_func,
                                count=2  # Generate 2 alternatives to keep processing time reasonable
                            )
                            
                            # Print results
                            print(f"\n{Fore.CYAN}=== Alternative Implementations for {selected_func} ==={Style.RESET_ALL}")
                            for i, alt in enumerate(alternatives, 1):
                                print(f"\n{Fore.GREEN}--- Alternative {i} ---{Style.RESET_ALL}")
                                print(f"Success: {alt.get('success', False)}")
                                print(f"Properties preserved: {alt.get('verification_result', {}).get('properties_preserved', False)}")
                                print(f"Strategy: {alt.get('strategy', 'unknown')}")
                                print(f"\nCode:\n{alt.get('alternative_function', 'No code generated')}")
                            
                            # Select best alternative
                            best_alt = analyzer_instance_for_complexity.select_best_alternative(alternatives)
                            if best_alt:
                                print(f"\n{Fore.GREEN}=== Best Alternative ==={Style.RESET_ALL}")
                                print(f"Strategy: {best_alt.get('strategy', 'unknown')}")
                                print(f"Properties preserved: {best_alt.get('verification_result', {}).get('properties_preserved', False)}")
                                print(f"\nCode:\n{best_alt.get('alternative_function', 'No code selected')}")
                                
                                # Add suggestions for where to save optimized version
                                original_path = func_info['file']
                                optimized_dir = os.path.join(os.path.dirname(original_path), "optimized")
                                optimized_path = os.path.join(optimized_dir, os.path.basename(original_path))
                                
                                print(f"\n{Fore.GREEN}To use this optimized implementation:{Style.RESET_ALL}")
                                print(f"{Fore.GREEN}1. Create directory: {optimized_dir}{Style.RESET_ALL}")
                                print(f"{Fore.GREEN}2. Save to: {optimized_path}{Style.RESET_ALL}")
                                print(f"{Fore.GREEN}3. Replace the original function with this optimized version{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.RED}Could not extract source code for {selected_func}. Skipping...{Style.RESET_ALL}")
                            
                    except Exception as e:
                        print(f"{Fore.RED}Error generating alternatives for {selected_func}: {e}{Style.RESET_ALL}")
                        logging.exception("Full traceback for alternative generation error:")
            else:
                print(f"{Fore.YELLOW}No functions selected. Skipping optimization.{Style.RESET_ALL}")
            
    except Exception as e:
        logging.error(f"An error occurred during 'analyze' command execution: {e}")
        logging.exception("Full traceback for 'analyze' command error:")
    finally:
        # Restore original monitor attribute
        if original_global_monitor_complexity is not None:
            complexity_analyzer_module.monitor = original_global_monitor_complexity
        elif hasattr(complexity_analyzer_module, 'monitor'):
            delattr(complexity_analyzer_module, 'monitor')
            
    print(f"{Fore.CYAN}'Analyze' command for {target_path} complete.{Style.RESET_ALL}")

# --- Main CLI Logic ---

if __name__ == "__main__":
    # Setup colored logging
    logger = setup_colored_logging()

    parser = argparse.ArgumentParser(
        description=f"{Fore.CYAN}Chimera Indexer: A CLI tool for analyzing Python codebases.{Style.RESET_ALL}",
        epilog=f"Example usage:\n"
               f"  {Fore.GREEN}python cli.py summary /path/to/your/code{Style.RESET_ALL}\n"
               f"  {Fore.GREEN}python cli.py analyze /path/to/your/file.py --api_key YOUR_API_KEY{Style.RESET_ALL}\n"
               f"  {Fore.GREEN}python cli.py analyze /path/to/your/dir --api_key $OPENAI_API_KEY{Style.RESET_ALL}",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "command", 
        choices=["summary", "analyze"], 
        help="The analysis command to execute:\n"
             "  summary: Performs a comprehensive static analysis of the codebase structure,\n"
             "           relationships, patterns, and concepts (using exec/full_analyzer.py).\n"
             "  analyze: Focuses on function complexity analysis and offers potential\n"
             "           AI-driven optimization suggestions if an API key is provided\n"
             "           (using exec/complexity.py)."
    )
    parser.add_argument(
        "path", 
        help="The path to the target Python file or directory to analyze."
    )
    parser.add_argument(
        "--api_key", 
        metavar='API_KEY',
        help="[Optional] OpenAI API key required by the 'analyze' command for generating\n"
             "alternative code implementations. If omitted, the tool checks the\n"
             "OPENAI_API_KEY environment variable. If neither is provided, 'analyze'\n"
             "runs only the complexity analysis without suggesting alternatives.",
        default=None
    )

    args = parser.parse_args()

    effective_api_key = args.api_key
    if args.command == "analyze" and not effective_api_key:
        effective_api_key = os.environ.get("OPENAI_API_KEY")
        if effective_api_key:
            logging.info("Using OpenAI API key from OPENAI_API_KEY environment variable.")
        # No explicit message if still not found, run_analyze_command will state it.

    if not os.path.exists(args.path):
        logging.error(f"Error: The path '{args.path}' does not exist.")
        sys.exit(1)

    if args.command == "summary":
        run_summary_command(args.path)
    elif args.command == "analyze":
        run_analyze_command(args.path, api_key=effective_api_key)
    
    print("CLI execution finished.") 