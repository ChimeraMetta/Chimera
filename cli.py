import argparse
import os
import sys
import logging
from typing import Union
from colorama import init, Fore, Style

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
    if results:
        for func_name, complexity in results.items():
            if complexity > 10:  # Consider functions with complexity > 10 as complex
                complex_functions[func_name] = {
                    'file': file_path,
                    'complexity': complexity
                }

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
            for i, (func_name, info) in enumerate(complex_functions.items(), 1):
                print(f"{i}. {func_name} (Complexity: {info['complexity']}, File: {info['file']})")
            
            while True:
                choice = input(f"\n{Fore.YELLOW}Would you like to see alternative implementations? (yes/no): {Style.RESET_ALL}").lower()
                if choice in ['yes', 'y']:
                    while True:
                        try:
                            func_num = int(input(f"\n{Fore.YELLOW}Enter the number of the function to optimize (1-{len(complex_functions)}): {Style.RESET_ALL}"))
                            if 1 <= func_num <= len(complex_functions):
                                selected_func = list(complex_functions.keys())[func_num - 1]
                                func_info = complex_functions[selected_func]
                                print(f"\n{Fore.GREEN}Generating alternative implementation for {selected_func}...{Style.RESET_ALL}")
                                complexity_analyzer_module.analyze_function_complexity_and_optimize(
                                    func_info['file'],
                                    analyzer_instance_for_complexity,
                                    target_function=selected_func
                                )
                                break
                            else:
                                print(f"{Fore.RED}Invalid number. Please enter a number between 1 and {len(complex_functions)}.{Style.RESET_ALL}")
                        except ValueError:
                            print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
                    
                    another = input(f"\n{Fore.YELLOW}Would you like to optimize another function? (yes/no): {Style.RESET_ALL}").lower()
                    if another not in ['yes', 'y']:
                        break
                elif choice in ['no', 'n']:
                    break
                else:
                    print(f"{Fore.RED}Please enter 'yes' or 'no'.{Style.RESET_ALL}")
            
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