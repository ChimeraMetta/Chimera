import argparse
import os
import sys
import logging

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

# --- Command Functions ---

def run_summary_command(target_path: str):
    print(f"Running summary for: {target_path}")
    
    local_monitor = DynamicMonitor()
    
    # Construct absolute path to the ontology file relative to the exec directory
    # exec.full_analyzer.ONTOLOGY_PATH is like "metta/code_ontology.metta"
    ontology_file_path = os.path.join(_EXEC_DIR, full_analyzer.ONTOLOGY_PATH)
    if not os.path.exists(ontology_file_path):
        print(f"Warning: Ontology file not found at {ontology_file_path}. Analysis might be incomplete.")
    else:
        local_monitor.load_metta_rules(ontology_file_path)

    # Temporarily set the global monitor instance for functions in full_analyzer module
    original_global_monitor_full = getattr(full_analyzer, 'monitor', None)
    full_analyzer.monitor = local_monitor
    
    try:
        # Replicate the __main__ execution flow of full_analyzer.py
        print(f"Analyzing codebase structure with full_analyzer.analyze_codebase...")
        full_analyzer.analyze_codebase(target_path)
        
        print("Analyzing type safety...")
        full_analyzer.analyze_type_safety() # Uses global full_analyzer.monitor
        
        print(f"Analyzing temporal evolution...")
        # analyze_temporal_evolution in full_analyzer.py can take monitor explicitly
        full_analyzer.analyze_temporal_evolution(target_path, local_monitor)
        
        print(f"Analyzing function complexity (static)...")
        full_analyzer.analyze_function_complexity(target_path) # Uses global full_analyzer.monitor
        
        # find_function_relationships was commented out in original full_analyzer.py __main__
        # print("Finding function relationships...")
        # full_analyzer.find_function_relationships() 

        print(f"Analyzing function call relationships (detailed)...")
        full_analyzer.analyze_function_call_relationships(target_path) # Uses global full_analyzer.monitor indirectly via decompose_file
        
        print("Finding type relationships...")
        full_analyzer.find_type_relationships() # Uses global full_analyzer.monitor
        
        print("Finding class relationships...")
        full_analyzer.find_class_relationships() # Uses global full_analyzer.monitor
        
        print("Finding module relationships...")
        full_analyzer.find_module_relationships() # Uses global full_analyzer.monitor
        
        print("Finding operation patterns...")
        full_analyzer.find_operation_patterns() # Uses global full_analyzer.monitor
        
        print("Analyzing structural patterns...")
        full_analyzer.analyze_structural_patterns() # Uses global full_analyzer.monitor
        
        print("Analyzing domain concepts...")
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

    print(f"Summary analysis for {target_path} complete.")

def run_analyze_command(target_path: str, api_key: str | None = None):
    print(f"Running 'analyze' command for: {target_path} (API key: {'Provided' if api_key else 'Not provided'})")
    
    local_monitor = DynamicMonitor()

    # Construct absolute path to the ontology file relative to the exec directory
    # exec.complexity_analyzer_module.ONTOLOGY_PATH is like "metta/code_ontology.metta"
    ontology_file_path = os.path.join(_EXEC_DIR, complexity_analyzer_module.ONTOLOGY_PATH)
    if not os.path.exists(ontology_file_path):
        print(f"Warning: Ontology file not found at {ontology_file_path}. Analysis might be incomplete.")
    else:
        local_monitor.load_metta_rules(ontology_file_path)

    # Temporarily set the global monitor for complexity_analyzer_module
    original_global_monitor_complexity = getattr(complexity_analyzer_module, 'monitor', None)
    complexity_analyzer_module.monitor = local_monitor

    # Setup logger for complexity_analyzer_module if it's not already configured by its import
    # This ensures its internal logging works as expected.
    if not any(handler for handler in complexity_analyzer_module.logger.handlers):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        complexity_analyzer_module.logger.info("CLI re-initialized logger for complexity_analyzer_module.")

    analyzer_instance_for_complexity = None
    if api_key:
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
            print("Could not initialize proof-guided implementation generator. Proceeding with complexity analysis only.")
    else:
        print("No API key provided. Proceeding with complexity analysis only (no alternative generation).")

    try:
        # Replicate the __main__ execution flow of complexity.py
        print(f"Analyzing codebase with complexity_analyzer_module.analyze_codebase...")
        # complexity_analyzer_module.analyze_codebase calls analyze_file, which uses global monitor
        complexity_analyzer_module.analyze_codebase(target_path) # analyzer_instance_for_complexity is not used by this specific analyze_codebase

        print(f"Analyzing function complexity and optimizing for {target_path}...")
        if os.path.isfile(target_path) and target_path.endswith('.py'):
            complexity_analyzer_module.analyze_function_complexity_and_optimize(target_path, analyzer_instance_for_complexity)
        elif os.path.isdir(target_path):
            for root, _, files_in_dir in os.walk(target_path):
                for f_name in files_in_dir:
                    if f_name.endswith('.py'):
                        file_path_to_analyze = os.path.join(root, f_name)
                        print(f"Analyzing and optimizing: {file_path_to_analyze}")
                        complexity_analyzer_module.analyze_function_complexity_and_optimize(file_path_to_analyze, analyzer_instance_for_complexity)
        else:
            print(f"Path is not a valid Python file or directory: {target_path}")
            
    except Exception as e:
        logging.error(f"An error occurred during 'analyze' command execution: {e}")
        logging.exception("Full traceback for 'analyze' command error:")
    finally:
        # Restore original monitor attribute
        if original_global_monitor_complexity is not None:
            complexity_analyzer_module.monitor = original_global_monitor_complexity
        elif hasattr(complexity_analyzer_module, 'monitor'):
            delattr(complexity_analyzer_module, 'monitor')
            
    print(f"'Analyze' command for {target_path} complete.")

# --- Main CLI Logic ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - CLI - %(message)s')

    parser = argparse.ArgumentParser(
        description="Chimera Indexer: A CLI tool for analyzing Python codebases.",
        epilog="Example usage:\\n"
               "  python cli.py summary /path/to/your/code\\n"
               "  python cli.py analyze /path/to/your/file.py --api_key YOUR_API_KEY\\n"
               "  python cli.py analyze /path/to/your/dir --api_key $OPENAI_API_KEY",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "command", 
        choices=["summary", "analyze"], 
        help="The analysis command to execute:\\n"
             "  summary: Performs a comprehensive static analysis of the codebase structure,\\n"
             "           relationships, patterns, and concepts (using exec/full_analyzer.py).\\n"
             "  analyze: Focuses on function complexity analysis and offers potential\\n"
             "           AI-driven optimization suggestions if an API key is provided\\n"
             "           (using exec/complexity.py)."
    )
    parser.add_argument(
        "path", 
        help="The path to the target Python file or directory to analyze."
    )
    parser.add_argument(
        "--api_key", 
        metavar='API_KEY',
        help="[Optional] OpenAI API key required by the 'analyze' command for generating\\n"
             "alternative code implementations. If omitted, the tool checks the\\n"
             "OPENAI_API_KEY environment variable. If neither is provided, 'analyze'\\n"
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