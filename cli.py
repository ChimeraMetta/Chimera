import argparse
import os
import sys
import logging
from typing import Union
import inquirer
from inquirer import themes
from io import StringIO

# --- Imports from project modules (now in exec directory) ---
from executors import full_analyzer
from executors import complexity as complexity_analyzer_module
from reflectors.dynamic_monitor import DynamicMonitor
from proofs.analyzer import ImmuneSystemProofAnalyzer
from common.logging_utils import get_logger, Fore, Style

_WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))

# Setup logger for this module
logger = get_logger(__name__)

# Custom theme for inquirer that matches our color scheme
class ChimeraTheme(themes.GreenPassion):
    def __init__(self):
        super().__init__()
        # Assuming Fore and Style are available (e.g., imported from logging_utils or globally)
        self.Checkbox.selected_icon = f"{Fore.GREEN}✓{Style.RESET_ALL}"
        self.Checkbox.unselected_icon = " "
        self.Checkbox.selected_color = Fore.GREEN # Inquirer might handle RESET_ALL
        self.Checkbox.unselected_color = Style.RESET_ALL # Or rely on autoreset
        # For List prompt, cursor color can be set if supported by theme
        if hasattr(self.List, 'selection_cursor'):
            self.List.selection_cursor = f"{Fore.GREEN}❯{Style.RESET_ALL}"
        if hasattr(self.List, 'selection_color'):
            self.List.selection_color = Fore.GREEN

def run_summary_command(target_path: str):
    local_monitor = DynamicMonitor()
    ontology_file_path = os.path.join(_WORKSPACE_ROOT, full_analyzer.ONTOLOGY_PATH)
    if not os.path.exists(ontology_file_path):
        logger.warning(f"Ontology file not found at {ontology_file_path}. Summary analysis might be incomplete.")
    else:
        local_monitor.load_metta_rules(ontology_file_path)

    original_global_monitor_full = getattr(full_analyzer, 'monitor', None)
    full_analyzer.monitor = local_monitor

    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        logger.info("Starting comprehensive codebase summary analysis...")
        logger.info(f"Target: {target_path}")
        logger.info("Analyzing codebase structure...")
        full_analyzer.analyze_codebase_structure(target_path)
        logger.info("Analyzing temporal aspects (git history)...")
        full_analyzer.analyze_temporal_aspects(target_path)
        logger.info("Analyzing structural patterns...")
        full_analyzer.analyze_structural_patterns()
        logger.info("Analyzing domain concepts...")
        full_analyzer.analyze_domain_concepts()
    except Exception as e:
        logger.error(f"An error occurred during summary analysis: {e}")
        logger.exception("Full traceback for summary analysis error:")
    finally:
        sys.stdout = old_stdout
        analyzer_direct_output = captured_output.getvalue()
        if analyzer_direct_output.strip(): # Only log if there's actual output
            logger.info(f"[full_analyzer direct output]:\n{analyzer_direct_output}")
            
        if original_global_monitor_full is not None:
            full_analyzer.monitor = original_global_monitor_full
        elif hasattr(full_analyzer, 'monitor'):
            delattr(full_analyzer, 'monitor')
    logger.info(f"Summary analysis for {target_path} complete.")

def run_analyze_command(target_path: str, api_key: Union[str, None] = None):
    logger.info(f"Running 'analyze' command for: {target_path} (API key: {'Provided' if api_key else 'Not provided'})")
    
    local_monitor = DynamicMonitor()
    ontology_file_path = os.path.join(_WORKSPACE_ROOT, complexity_analyzer_module.ONTOLOGY_PATH)
    if not os.path.exists(ontology_file_path):
        logger.warning(f"Ontology file not found at {ontology_file_path}. Analysis might be incomplete.")
    else:
        local_monitor.load_metta_rules(ontology_file_path)

    original_global_monitor_complexity = getattr(complexity_analyzer_module, 'monitor', None)
    complexity_analyzer_module.monitor = local_monitor
    
    # Check if complexity_analyzer_module has its own logger and if it's configured
    # This is a temporary measure. Ideally, complexity_analyzer_module.py should be refactored.
    if hasattr(complexity_analyzer_module, 'logger') and isinstance(getattr(complexity_analyzer_module, 'logger'), logging.Logger):
        comp_logger = getattr(complexity_analyzer_module, 'logger')
        if not comp_logger.handlers:
            logger.debug(f"Complexity module logger ('{comp_logger.name}') has no handlers. It might not produce output unless configured elsewhere or prints directly.")
            # If necessary, one could add a basic handler here for it:
            # basic_handler = logging.StreamHandler()
            # basic_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # basic_handler.setFormatter(basic_formatter)
            # comp_logger.addHandler(basic_handler)
            # comp_logger.setLevel(logging.INFO) # Or an appropriate level
            # logger.info(f"Temporarily added a basic handler to {comp_logger.name}")

    analyzer_instance_for_complexity = None
    if api_key:
        logger.debug("API key provided. Attempting to initialize ImmuneSystemProofAnalyzer...")
        try:
            # Log before initialization attempt
            logger.info("Initializing proof-guided implementation generator...")
            analyzer_instance_for_complexity = ImmuneSystemProofAnalyzer(metta_space=local_monitor.metta_space, api_key=api_key)
            analyzer_instance_for_complexity = complexity_analyzer_module.integrate_with_immune_system(analyzer_instance_for_complexity)
            logger.info("Proof-guided implementation generator initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing ImmuneSystemProofAnalyzer: {e}")
            logger.exception("Full traceback for ImmuneSystemProofAnalyzer initialization error:")
            logger.warning("Could not initialize proof-guided implementation generator. Proceeding with complexity analysis only.")
            analyzer_instance_for_complexity = None 
    else:
        logger.warning("No API key provided. AI-driven optimization suggestions will be unavailable. Proceeding with complexity analysis only.")

    logger.debug(f"Analyzer instance for complexity: {type(analyzer_instance_for_complexity)}")
    
    # Capture stdout from complexity_analyzer_module calls
    old_stdout_complexity = sys.stdout
    sys.stdout = captured_output_complexity = StringIO()
    try:
        logger.info(f"Running complexity analysis for {target_path}...")
        complexity_analyzer_module.analyze_function_complexity_and_optimize(target_path, analyzer_instance_for_complexity)
    except Exception as e:
        logger.error(f"An error occurred during 'analyze_function_complexity_and_optimize': {e}")
        logger.exception("Full traceback for complexity analysis error:")
    finally:
        sys.stdout = old_stdout_complexity
        complexity_direct_output = captured_output_complexity.getvalue()
        if complexity_direct_output.strip(): # Only log if there's actual output
            logger.info(f"[complexity_analyzer direct output]:\n{complexity_direct_output}")

        if original_global_monitor_complexity is not None:
            complexity_analyzer_module.monitor = original_global_monitor_complexity
        elif hasattr(complexity_analyzer_module, 'monitor'):
            delattr(complexity_analyzer_module, 'monitor')

    if analyzer_instance_for_complexity and os.path.isfile(target_path):
        logger.info(f"Starting interactive function optimization for file: {target_path}")
        
        decomposed_file_info = complexity_analyzer_module.decompose_file(target_path)
        if decomposed_file_info and "functions" in decomposed_file_info and decomposed_file_info["functions"]:
            functions_in_file = [f_info["name"] for f_info in decomposed_file_info["functions"]]
            
            if not functions_in_file:
                logger.info("No functions found in the decomposed file info to offer for optimization.")
            else:
                questions = [
                    inquirer.List('selected_func',
                                  message="Select a function to attempt optimization (or 'skip')",
                                  choices=functions_in_file + ['skip'],
                                  default='skip'),
                ]
                current_theme = ChimeraTheme() # Use the custom theme
                try:
                    logger.info("Displaying interactive prompt for function selection...")
                    answers = inquirer.prompt(questions, theme=current_theme)
                    selected_func = answers['selected_func'] if answers and 'selected_func' in answers else 'skip'
                except Exception as e: 
                    logger.warning(f"Could not display interactive prompt (อาจไม่ใช่ TTY): {e}. Skipping function selection.")
                    selected_func = 'skip'

                if selected_func and selected_func != 'skip':
                    logger.info(f"User selected function: '{selected_func}' for optimization.")
                    func_info = next((f for f in decomposed_file_info["functions"] if f["name"] == selected_func), None)
                    
                    if func_info and "source" in func_info:
                        func_source = func_info["source"]
                        try:
                            logger.info(f"Generating up to 2 alternative implementations for '{selected_func}'...")
                            alternatives = analyzer_instance_for_complexity.generate_verified_alternatives(
                                func_source, selected_func, count=2)
                            
                            if alternatives:
                                logger.info(f"=== {len(alternatives)} Alternative Implementations Generated for {selected_func} ===")
                                for i, alt in enumerate(alternatives, 1):
                                    logger.info(f"--- Alternative {i} ---")
                                    logger.info(f"  Strategy: {alt.get('strategy', 'N/A')}")
                                    logger.info(f"  Success: {alt.get('success', False)}")
                                    logger.info(f"  Properties Preserved: {alt.get('verification_result', {}).get('properties_preserved', False)}")
                                    alt_code = alt.get('alternative_function', 'No code generated')
                                    logger.info(f"  Code:\n{alt_code}") 
                                
                                best_alt = analyzer_instance_for_complexity.select_best_alternative(alternatives)
                                if best_alt:
                                    logger.info("=== Best Alternative Selected ===")
                                    logger.info(f"  Strategy: {best_alt.get('strategy', 'N/A')}")
                                    logger.info(f"  Properties Preserved: {best_alt.get('verification_result', {}).get('properties_preserved', False)}")
                                    best_code = best_alt.get('alternative_function', 'No code selected')
                                    logger.info(f"  Code:\n{best_code}")
                                    
                                    original_path_func = func_info.get('file', target_path) 
                                    optimized_dir = os.path.join(os.path.dirname(original_path_func), "optimized_code") # Changed dir name
                                    optimized_file_name = f"{os.path.splitext(os.path.basename(original_path_func))[0]}_{selected_func}_optimized.py"
                                    optimized_path = os.path.join(optimized_dir, optimized_file_name)
                                    
                                    logger.info("Suggestion for using this optimized implementation:")
                                    logger.info(f"  1. Ensure directory exists or create: {optimized_dir}")
                                    logger.info(f"  2. Save the BEST alternative code to a new file: {optimized_path}")
                                    logger.info(f"  3. Manually review and integrate this new function into your codebase, replacing the original '{selected_func}' in '{original_path_func}'.")
                                else:
                                    logger.warning("No best alternative was selected from the generated options.")
                            else:
                                logger.info(f"No alternatives were generated for '{selected_func}'.")
                                
                        except Exception as e:
                            logger.error(f"Error generating or selecting alternatives for '{selected_func}': {e}")
                            logger.exception("Full traceback for alternative generation/selection error:")
                    else:
                        logger.error(f"Could not retrieve source code for '{selected_func}'. Skipping optimization for this function.")
                elif selected_func == 'skip':
                    logger.info("User chose to skip function optimization.")
                else: # Should not happen if logic is correct
                    logger.warning("No function selected for optimization or an unexpected state occurred.")
        else:
            logger.warning(f"No functions found in the analysis of {target_path} to offer for optimization, or file info is missing.")
            
    logger.info(f"'Analyze' command for {target_path} complete.")

# --- Main CLI Logic ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chimera Indexer: A CLI tool for analyzing Python codebases.",
        epilog=f"Example usage:\n"
               f"  python cli.py summary /path/to/your/code\n"
               f"  python cli.py analyze /path/to/your/file.py --api_key YOUR_API_KEY\n"
               f"  python cli.py analyze /path/to/your/dir --api_key $OPENAI_API_KEY",
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

    if not os.path.exists(args.path):
        logger.error(f"Error: The path '{args.path}' does not exist. Please provide a valid file or directory path.")
        sys.exit(1)

    effective_api_key = args.api_key
    if not effective_api_key:
        effective_api_key = os.getenv("OPENAI_API_KEY")
        if effective_api_key:
            logger.info("Using OPENAI_API_KEY from environment variable for 'analyze' command.")

    if args.command == "summary":
        run_summary_command(args.path)
    elif args.command == "analyze":
        if not effective_api_key:
            logger.info("No API key provided or found in environment. 'analyze' will run without AI optimization features.")
        run_analyze_command(args.path, effective_api_key)
    else:
        logger.error(f"Unknown command: {args.command}") 
        parser.print_help()
        sys.exit(1)

    logger.info("CLI command execution finished.")
    sys.exit(0)