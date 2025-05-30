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
from executors.exporter import (
    export_from_summary_analysis, 
    export_from_complexity_analysis, 
    import_metta_file,
    combine_metta_files,
    verify_export
)

_WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
_INTERMEDIATE_EXPORT_DIR = os.path.join(_WORKSPACE_ROOT, ".chimera_exports")
_SUMMARY_EXPORT_FILE = os.path.join(_INTERMEDIATE_EXPORT_DIR, "summary_export.metta")
_ANALYZE_EXPORT_FILE = os.path.join(_INTERMEDIATE_EXPORT_DIR, "analyze_export.metta")
_IMPORTED_ATOMSPACE_FILE = os.path.join(_INTERMEDIATE_EXPORT_DIR, "imported_atomspace.metta")

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
    logger.info(f"Running 'summary' command for: {target_path}")

    # Initialize a local monitor for this command (keep for ontology loading)
    local_monitor = DynamicMonitor()
    ontology_file_path = os.path.join(_WORKSPACE_ROOT, full_analyzer.ONTOLOGY_PATH)
    
    if not os.path.exists(ontology_file_path):
        logger.warning(f"Ontology file not found at {ontology_file_path}. Summary analysis might be incomplete.")
    else:
        local_monitor.load_metta_rules(ontology_file_path)

    # Temporarily replace global monitor in full_analyzer if it exists
    original_global_monitor_full = getattr(full_analyzer, 'monitor', None)
    full_analyzer.monitor = local_monitor

    # Capture stdout for this specific analyzer
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        logger.info("Starting comprehensive codebase summary analysis...")
        logger.info(f"Target: {target_path}")
        logger.info("Analyzing codebase structure...")
        full_analyzer.analyze_codebase(target_path)
        logger.info("Summary analysis completed successfully.")
        logger.info("Analyzing temporal aspects (git history)...")
        full_analyzer.analyze_temporal_evolution(target_path, monitor=local_monitor)
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
        if analyzer_direct_output.strip():
            logger.info(f"[full_analyzer direct output]:\n{analyzer_direct_output}")
            
        # Restore original monitor if it was replaced
        if original_global_monitor_full is not None:
            full_analyzer.monitor = original_global_monitor_full
        elif hasattr(full_analyzer, 'monitor'):
            delattr(full_analyzer, 'monitor')

        # DIRECT EXPORT: Export atoms directly from static analysis (no atomspace needed)
        try:
            if not os.path.exists(_INTERMEDIATE_EXPORT_DIR):
                os.makedirs(_INTERMEDIATE_EXPORT_DIR, exist_ok=True)
                logger.info(f"Created intermediate export directory: {_INTERMEDIATE_EXPORT_DIR}")
            
            logger.info(f"Directly exporting static analysis atoms to: {_SUMMARY_EXPORT_FILE}")
            
            # Direct export from static analysis - no monitor needed!
            export_success = export_from_summary_analysis(target_path, _SUMMARY_EXPORT_FILE)
            
            if export_success:
                # Verify the export
                verification = verify_export(_SUMMARY_EXPORT_FILE)
                if verification["success"]:
                    logger.info(f"Summary atoms directly exported successfully: {verification['atom_count']} atoms ({verification['file_size']} bytes)")
                else:
                    logger.warning(f"Export verification failed: {verification.get('error', 'Unknown error')}")
            else:
                logger.warning(f"Failed to directly export summary atoms.")
                
        except Exception as e:
            logger.error(f"Error during direct summary atom export: {e}")

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
    
    analyzer_instance_for_complexity = None
    if api_key:
        logger.debug("API key provided. Attempting to initialize ImmuneSystemProofAnalyzer...")
        try:
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
    
    # Store analysis results for export
    analysis_results = {
        "complexity_metrics": {},
        "optimization_suggestions": []
    }
    
    # Capture stdout from complexity_analyzer_module calls
    old_stdout_complexity = sys.stdout
    sys.stdout = captured_output_complexity = StringIO()
    try:
        logger.info(f"Running complexity analysis for {target_path}...")
        
        # Your existing complexity analysis
        complexity_analyzer_module.analyze_function_complexity_and_optimize(target_path, analyzer_instance_for_complexity)
        
        # If you have access to complexity results, add them to analysis_results
        # This depends on what your complexity analyzer returns
        # analysis_results["complexity_metrics"] = some_complexity_data
        
    except Exception as e:
        logger.error(f"An error occurred during 'analyze_function_complexity_and_optimize': {e}")
        logger.exception("Full traceback for complexity analysis error:")
    finally:
        sys.stdout = old_stdout_complexity
        complexity_direct_output = captured_output_complexity.getvalue()
        if complexity_direct_output.strip():
            logger.info(f"[complexity_analyzer direct output]:\n{complexity_direct_output}")

        if original_global_monitor_complexity is not None:
            complexity_analyzer_module.monitor = original_global_monitor_complexity
        elif hasattr(complexity_analyzer_module, 'monitor'):
            delattr(complexity_analyzer_module, 'monitor')

        # DIRECT EXPORT: Export atoms directly from static analysis (no atomspace needed)
        try:
            if not os.path.exists(_INTERMEDIATE_EXPORT_DIR):
                os.makedirs(_INTERMEDIATE_EXPORT_DIR, exist_ok=True)
                logger.info(f"Created intermediate export directory: {_INTERMEDIATE_EXPORT_DIR}")
            
            logger.info(f"Directly exporting complexity analysis atoms to: {_ANALYZE_EXPORT_FILE}")
            
            # Direct export from static analysis - no monitor needed!
            export_success = export_from_complexity_analysis(target_path, _ANALYZE_EXPORT_FILE, analysis_results)
            
            if export_success:
                # Verify the export
                verification = verify_export(_ANALYZE_EXPORT_FILE)
                if verification["success"]:
                    logger.info(f"Analyze atoms directly exported successfully: {verification['atom_count']} atoms ({verification['file_size']} bytes)")
                else:
                    logger.warning(f"Export verification failed: {verification.get('error', 'Unknown error')}")
            else:
                logger.warning(f"Failed to directly export analyze atoms.")
                
        except Exception as e:
            logger.error(f"Error during direct analyze atom export: {e}")

    # Your existing interactive optimization logic
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
                current_theme = ChimeraTheme()
                try:
                    logger.info("Displaying interactive prompt for function selection...")
                    answers = inquirer.prompt(questions, theme=current_theme)
                    selected_func = answers['selected_func'] if answers and 'selected_func' in answers else 'skip'
                except Exception as e: 
                    logger.warning(f"Could not display interactive prompt: {e}. Skipping function selection.")
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
                                    optimized_dir = os.path.join(os.path.dirname(original_path_func), "optimized_code")
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
                else:
                    logger.warning("No function selected for optimization or an unexpected state occurred.")
        else:
            logger.warning(f"No functions found in the analysis of {target_path} to offer for optimization, or file info is missing.")
            
    logger.info(f"'Analyze' command for {target_path} complete.")

def run_import_command(source_metta_path: str, overwrite: bool = False):
    """
    Import atoms from an external .metta file into the current atomspace.
    
    Args:
        source_metta_path: Path to the .metta file to import
        overwrite: Whether to overwrite conflicting atoms
    """
    logger.info(f"Running 'import' command from: {source_metta_path} (overwrite: {overwrite})")
    
    # Validate source file exists
    if not os.path.exists(source_metta_path):
        logger.error(f"Source file does not exist: {source_metta_path}")
        return
    
    # Verify it's a .metta file
    if not source_metta_path.endswith('.metta'):
        logger.warning(f"File does not have .metta extension: {source_metta_path}")
    
    # Create intermediate export directory if it doesn't exist
    if not os.path.exists(_INTERMEDIATE_EXPORT_DIR):
        os.makedirs(_INTERMEDIATE_EXPORT_DIR, exist_ok=True)
        logger.info(f"Created intermediate export directory: {_INTERMEDIATE_EXPORT_DIR}")
    
    try:
        # Import the file into our intermediate atomspace file
        result = import_metta_file(source_metta_path, _IMPORTED_ATOMSPACE_FILE, overwrite_conflicts=overwrite)
        
        if result["success"]:
            logger.info(f"Import successful: {result['message']}")
            
            # Provide detailed statistics
            if "imported" in result:
                logger.info(f"  - New atoms imported: {result['imported']}")
            if "overwritten" in result:
                logger.info(f"  - Atoms overwritten: {result['overwritten']}")
            if "skipped" in result:
                logger.info(f"  - Atoms skipped (conflicts): {result['skipped']}")
            
            # Verify the imported atomspace
            verification = verify_export(_IMPORTED_ATOMSPACE_FILE)
            if verification["success"]:
                logger.info(f"Imported atomspace verified: {verification['atom_count']} total atoms, {verification['file_size']} bytes")
            else:
                logger.warning(f"Import verification failed: {verification.get('error', 'Unknown error')}")
            
            # Provide guidance on next steps
            logger.info(f"\nNext steps:")
            logger.info(f"  1. The imported atoms are now available in: {_IMPORTED_ATOMSPACE_FILE}")
            logger.info(f"  2. Use 'export' command to create a consolidated atomspace including imported atoms")
            logger.info(f"  3. Imported atoms will be included in future export operations")
            
        else:
            logger.error(f"Import failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"An unexpected error occurred during import: {e}")
        logger.exception("Full traceback for import command error:")

    logger.info(f"'import' command for {source_metta_path} complete.")

def run_export_atomspace_command(output_metta_path: str):
    logger.info(f"Running 'export' command. Output will be saved to: {output_metta_path}")

    # Collect all available .metta files to combine
    files_to_combine = []
    
    # Add base ontology if available
    try:
        if hasattr(full_analyzer, 'ONTOLOGY_PATH') and isinstance(full_analyzer.ONTOLOGY_PATH, str):
            ontology_file_path = os.path.join(_WORKSPACE_ROOT, full_analyzer.ONTOLOGY_PATH)
            if os.path.exists(ontology_file_path):
                files_to_combine.append(ontology_file_path)
                logger.info(f"Including base ontology: {ontology_file_path}")
            else:
                logger.warning(f"Base ontology file not found: {ontology_file_path}")
        else:
            logger.warning("No base ontology path available")
    except Exception as e:
        logger.error(f"Error checking base ontology: {e}")

    # Add imported atomspace if it exists
    if os.path.exists(_IMPORTED_ATOMSPACE_FILE):
        files_to_combine.append(_IMPORTED_ATOMSPACE_FILE)
        logger.info(f"Including imported atomspace: {_IMPORTED_ATOMSPACE_FILE}")
    else:
        logger.info(f"No imported atomspace found: {_IMPORTED_ATOMSPACE_FILE}")

    # Add summary export if it exists
    if os.path.exists(_SUMMARY_EXPORT_FILE):
        files_to_combine.append(_SUMMARY_EXPORT_FILE)
        logger.info(f"Including summary export: {_SUMMARY_EXPORT_FILE}")
    else:
        logger.info(f"Summary export file not found: {_SUMMARY_EXPORT_FILE}")

    # Add analyze export if it exists
    if os.path.exists(_ANALYZE_EXPORT_FILE):
        files_to_combine.append(_ANALYZE_EXPORT_FILE)
        logger.info(f"Including analyze export: {_ANALYZE_EXPORT_FILE}")
    else:
        logger.info(f"Analyze export file not found: {_ANALYZE_EXPORT_FILE}")

    # Check if we have any files to combine
    if not files_to_combine:
        logger.warning("No files found to export. Run 'summary', 'analyze', or 'import' commands first.")
        # Create an empty export file with just metadata
        try:
            import time
            with open(output_metta_path, 'w') as f:
                f.write(f"; MeTTa Atomspace Export\n")
                f.write(f"; Exported: {time.ctime()}\n")
                f.write(f"; No atoms available - run 'summary', 'analyze', or 'import' commands first\n")
                f.write(f";\n")
            logger.info(f"Created empty export file: {output_metta_path}")
        except Exception as e:
            logger.error(f"Error creating empty export file: {e}")
        return

    # Combine all files into the final export
    logger.info(f"Combining {len(files_to_combine)} files into consolidated export: {output_metta_path}")
    try:
        success = combine_metta_files(files_to_combine, output_metta_path, "consolidated_export")
        
        if success:
            # Verify the final export
            verification = verify_export(output_metta_path)
            if verification["success"]:
                logger.info(f"Consolidated atomspace successfully exported: {verification['atom_count']} atoms, {verification['file_size']} bytes")
                logger.info(f"Combined from {len(files_to_combine)} source files:")
                for file_path in files_to_combine:
                    logger.info(f"  - {os.path.basename(file_path)}")
            else:
                logger.warning(f"Export verification failed: {verification.get('error', 'Unknown error')}")
        else:
            logger.error(f"Failed to combine files into consolidated export: {output_metta_path}")
            
    except Exception as e:
        logger.error(f"An unexpected error occurred during consolidated export: {e}")
        logger.exception("Full traceback for export command error:")

    logger.info(f"'export' command for {output_metta_path} complete.")

# --- Main CLI Logic ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chimera Indexer: A CLI tool for analyzing Python codebases and managing MeTTa atomspaces.",
        epilog=f"Example usage:\n"
               f"  python cli.py summary /path/to/your/code\n"
               f"  python cli.py analyze /path/to/your/file.py --api_key YOUR_API_KEY\n"
               f"  python cli.py analyze /path/to/your/dir --api_key $OPENAI_API_KEY\n"
               f"  python cli.py import /path/to/existing/atomspace.metta\n"
               f"  python cli.py import /path/to/atomspace.metta --overwrite\n"
               f"  python cli.py export /path/to/output/atomspace.metta",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "command", 
        choices=["summary", "analyze", "import", "export"],
        help="The command to execute:\n"
             "  summary: Performs a comprehensive static analysis of the codebase structure,\n"
             "           relationships, patterns, and concepts (using exec/full_analyzer.py).\n"
             "  analyze: Focuses on function complexity analysis and offers potential\n"
             "           AI-driven optimization suggestions if an API key is provided\n"
             "           (using exec/complexity.py).\n"
             "  import:  Imports atoms from an external .metta file into the current atomspace.\n"
             "           Imported atoms will be included in subsequent export operations.\n"
             "  export:  Exports a consolidated MeTTa atomspace (including imported atoms,\n"
             "           summary analysis, and complexity analysis) to a specified .metta file."
    )
    parser.add_argument(
        "path", 
        help="The path argument, meaning depends on the command:\n"
             "  For 'summary', 'analyze': Path to the target Python file or directory to analyze.\n"
             "  For 'import': Path to the .metta file to import atoms from.\n"
             "  For 'export': Path to the output .metta file where the atomspace will be saved."
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="[Optional] For 'import' command: overwrite conflicting atoms instead of skipping them.\n"
             "Default behavior is to skip atoms that already exist in the atomspace.",
        default=False
    )

    args = parser.parse_args()

    # Path validation based on command
    if args.command in ["summary", "analyze"]:
        if not os.path.exists(args.path):
            logger.error(f"Error: The input path '{args.path}' for command '{args.command}' does not exist. Please provide a valid file or directory path.")
            sys.exit(1)
    elif args.command == "import":
        if not os.path.exists(args.path):
            logger.error(f"Error: The import source file '{args.path}' does not exist. Please provide a valid .metta file path.")
            sys.exit(1)
        if not os.path.isfile(args.path):
            logger.error(f"Error: The import path '{args.path}' must be a file, not a directory.")
            sys.exit(1)
    elif args.command == "export":
        output_path_str = str(args.path) # Ensure it's a string
        output_dir = os.path.dirname(output_path_str)
        if not output_dir: # Handles cases like "filename.metta" where dirname is empty
            output_dir = "." # Assume current directory

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
            except OSError as e:
                logger.error(f"Error: Could not create output directory '{output_dir}': {e}")
                sys.exit(1)
        
        # Check if the output path itself is a directory (which is not allowed for a file output)
        if os.path.isdir(output_path_str):
            logger.error(f"Error: The output path '{output_path_str}' for 'export' must be a file path, not a directory.")
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
    elif args.command == "import":
        run_import_command(args.path, args.overwrite)
    elif args.command == "export":
        run_export_atomspace_command(args.path)
    else:
        logger.error(f"Unknown command: {args.command}") 
        parser.print_help()
        sys.exit(1)

    logger.info("CLI command execution finished.")
    sys.exit(0)