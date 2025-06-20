import argparse
import os
import sys
from typing import Union
import inquirer
from io import StringIO
import ast
import time
import importlib.util

# --- Imports from project modules (now in exec directory) ---
from executors import full_analyzer
from executors import complexity as complexity_analyzer_module
from reflectors.dynamic_monitor import DynamicMonitor
from proofs.analyzer import ImmuneSystemProofAnalyzer
from common.logging_utils import get_logger, ColoredHelpFormatter, ChimeraTheme
from executors.export_importer import (
    export_from_summary_analysis, 
    export_from_complexity_analysis, 
    import_metta_file,
    combine_metta_files,
    verify_export,
    export_from_metta_generation
)
from metta_generator.base import MeTTaPoweredModularDonorGenerator
from metta_generator.operation_substitution import OperationSubstitutionGenerator
from metta_generator.data_struct_adaptation import DataStructureAdaptationGenerator  
from metta_generator.algo_transformation import AlgorithmTransformationGenerator

_WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
_INTERMEDIATE_EXPORT_DIR = os.path.join(_WORKSPACE_ROOT, ".chimera_exports")
_SUMMARY_EXPORT_FILE = os.path.join(_INTERMEDIATE_EXPORT_DIR, "summary_export.metta")
_ANALYZE_EXPORT_FILE = os.path.join(_INTERMEDIATE_EXPORT_DIR, "analyze_export.metta")
_IMPORTED_ATOMSPACE_FILE = os.path.join(_INTERMEDIATE_EXPORT_DIR, "imported_atomspace.metta")

# Setup logger for this module
logger = get_logger(__name__)

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
    using_metta_generator = False # Flag to indicate which optimization path
    metta_donor_generator_instance = None # For the new Modular MeTTa Donor Generator
    
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
        logger.warning("No API key provided. AI-driven optimization suggestions will be unavailable.")
        logger.info("Defaulting to MeTTa-based donor generation for optimization suggestions.")
        using_metta_generator = True
        analyzer_instance_for_complexity = None # Ensure it's None for MeTTa path
        try:
            logger.info("Initializing Modular MeTTa Donor Generator for fallback...")
            metta_donor_generator_instance = MeTTaPoweredModularDonorGenerator(metta_space=local_monitor.metta_space)

            logger.info("Registering specialized donor generators for fallback...")
            op_sub_gen = OperationSubstitutionGenerator()
            metta_donor_generator_instance.registry.register_generator(op_sub_gen)
            logger.info("   OperationSubstitutionGenerator registered for fallback")

            ds_adapt_gen = DataStructureAdaptationGenerator()
            metta_donor_generator_instance.registry.register_generator(ds_adapt_gen)
            logger.info("   DataStructureAdaptationGenerator registered for fallback")

            algo_transform_gen = AlgorithmTransformationGenerator()
            metta_donor_generator_instance.registry.register_generator(algo_transform_gen)
            logger.info("   AlgorithmTransformationGenerator registered for fallback")

            total_gens = len(metta_donor_generator_instance.registry.generators)
            supported_strats = len(metta_donor_generator_instance.registry.get_supported_strategies())
            logger.info(f"  Total fallback generators registered: {total_gens}")
            logger.info(f"  Supported fallback strategies: {supported_strats}")

            if os.path.exists(ontology_file_path):
                ontology_loaded = metta_donor_generator_instance.load_ontology(ontology_file_path)
                if ontology_loaded:
                    logger.info("   MeTTa ontology loaded successfully for fallback generator.")
                else:
                    logger.warning("  ⚠ Fallback MeTTa ontology could not be loaded, continuing with defaults.")
            else:
                logger.warning(f"  ⚠ MeTTa ontology file not found at {ontology_file_path} for fallback generator, continuing with defaults.")
            logger.info("Modular MeTTa Donor Generator initialized successfully for fallback.")
        except Exception as e:
            logger.error(f"Error initializing Modular MeTTa Donor Generator for fallback: {e}")
            logger.exception("Full traceback for Modular MeTTa generator initialization error:")
            using_metta_generator = False 
            metta_donor_generator_instance = None

    logger.debug(f"Analyzer instance for complexity: {type(analyzer_instance_for_complexity)}")
    logger.debug(f"Using MeTTa generator for suggestions: {using_metta_generator}")
    
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

    # Interactive optimization logic updated
    if (analyzer_instance_for_complexity or using_metta_generator) and os.path.isfile(target_path):
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
                        
                        if using_metta_generator and metta_donor_generator_instance:
                            logger.info(f"Generating donor candidates for '{selected_func}' using Modular MeTTa Donor Generator...")
                            try:
                                metta_candidates = metta_donor_generator_instance.generate_donors_from_function(func_source)
                                
                                if metta_candidates:
                                    logger.info(f"=== {len(metta_candidates)} MeTTa Donor Candidates Generated for {selected_func} (Modular Generator) ===")
                                    best_metta_alt_display = None # To store the display dict of the best MeTTa candidate
                                    
                                    for i, mc in enumerate(metta_candidates, 1):
                                        # Create a display dictionary for consistent logging
                                        alt_display = {
                                            'strategy': mc.get('strategy', 'N/A'),
                                            'success': True, # Assuming generation itself means success
                                            'verification_result': {'properties_preserved': mc.get('properties', [])},
                                            'alternative_function': mc.get('code', 'No code generated'),
                                            'description': mc.get('description', ''),
                                            'name': mc.get('name', f"{selected_func}_metta_alt_{i}"),
                                            'score': mc.get('final_score', 0.0)
                                        }
                                        logger.info(f"--- MeTTa Alternative {i} ({alt_display['name']}) ---")
                                        logger.info(f"  Strategy: {alt_display['strategy']}")
                                        logger.info(f"  Score: {alt_display['score']:.2f}")
                                        logger.info(f"  Description: {alt_display['description']}")
                                        logger.info(f"  Properties Preserved: {alt_display['verification_result']['properties_preserved']}")
                                        logger.info(f"  Code:\n{alt_display['alternative_function']}")
                                        
                                        if i == 1: # First one is the best due to prior sorting in metta_generator
                                            best_metta_alt_display = alt_display

                                    if best_metta_alt_display:
                                        logger.info("=== Best MeTTa Alternative Selected (Top Scoring - Modular Generator) ===")
                                        logger.info(f"  Name: {best_metta_alt_display['name']}")
                                        logger.info(f"  Strategy: {best_metta_alt_display.get('strategy', 'N/A')}")
                                        logger.info(f"  Score: {best_metta_alt_display.get('score', 0.0):.2f}")
                                        logger.info(f"  Description: {best_metta_alt_display.get('description', 'N/A')}")
                                        logger.info(f"  Properties Preserved: {best_metta_alt_display.get('verification_result', {}).get('properties_preserved', [])}")
                                        best_code = best_metta_alt_display.get('alternative_function', 'No code selected')
                                        logger.info(f"  Code:\n{best_code}")
                                        
                                        original_path_func = func_info.get('file', target_path) 
                                        optimized_dir = os.path.join(os.path.dirname(original_path_func), "optimized_code_metta")
                                        safe_candidate_name = best_metta_alt_display['name'].replace(' ', '_').replace('/', '_').replace('\\\\', '_') # Ensure filename safety
                                        optimized_file_name = f"{os.path.splitext(os.path.basename(original_path_func))[0]}_{safe_candidate_name}_optimized.py"
                                        optimized_path = os.path.join(optimized_dir, optimized_file_name)
                                        
                                        logger.info("Suggestion for using this MeTTa-generated implementation:")
                                        logger.info(f"  1. Ensure directory exists or create: {optimized_dir}")
                                        logger.info(f"  2. Save the BEST alternative code to a new file: {optimized_path}")
                                        logger.info(f"  3. Manually review and integrate this new function into your codebase, replacing the original '{selected_func}' in '{original_path_func}'.")
                                    else:
                                        logger.warning("No best MeTTa alternative was selected (list empty or processing issue).")
                                else:
                                    logger.info(f"No MeTTa donor candidates were generated for '{selected_func}' by the Modular Generator.")
                                    
                            except Exception as e:
                                logger.error(f"Error during Modular MeTTa donor generation for '{selected_func}': {e}")
                                logger.exception("Full traceback for Modular MeTTa donor generation error:")

                        elif analyzer_instance_for_complexity: # Original API key path
                            try:
                                logger.info(f"Generating up to 2 alternative implementations for '{selected_func}' using API...")
                                alternatives = analyzer_instance_for_complexity.generate_verified_alternatives(
                                    func_source, selected_func, count=2)
                                
                                if alternatives:
                                    logger.info(f"=== {len(alternatives)} Alternative Implementations Generated for {selected_func} (API) ===")
                                    for i, alt in enumerate(alternatives, 1):
                                        logger.info(f"--- API Alternative {i} ---")
                                        logger.info(f"  Strategy: {alt.get('strategy', 'N/A')}")
                                        logger.info(f"  Success: {alt.get('success', False)}")
                                        logger.info(f"  Properties Preserved: {alt.get('verification_result', {}).get('properties_preserved', False)}")
                                        alt_code = alt.get('alternative_function', 'No code generated')
                                        logger.info(f"  Code:\n{alt_code}") 
                                    
                                    best_api_alt = analyzer_instance_for_complexity.select_best_alternative(alternatives)
                                    if best_api_alt:
                                        logger.info("=== Best API Alternative Selected ===")
                                        logger.info(f"  Strategy: {best_api_alt.get('strategy', 'N/A')}")
                                        logger.info(f"  Properties Preserved: {best_api_alt.get('verification_result', {}).get('properties_preserved', False)}")
                                        best_code_api = best_api_alt.get('alternative_function', 'No code selected')
                                        logger.info(f"  Code:\n{best_code_api}")
                                        
                                        original_path_func = func_info.get('file', target_path) 
                                        optimized_dir = os.path.join(os.path.dirname(original_path_func), "optimized_code")
                                        optimized_file_name = f"{os.path.splitext(os.path.basename(original_path_func))[0]}_{selected_func}_optimized.py"
                                        optimized_path = os.path.join(optimized_dir, optimized_file_name)
                                        
                                        logger.info("Suggestion for using this optimized implementation:")
                                        logger.info(f"  1. Ensure directory exists or create: {optimized_dir}")
                                        logger.info(f"  2. Save the BEST alternative code to a new file: {optimized_path}")
                                        logger.info(f"  3. Manually review and integrate this new function into your codebase, replacing the original '{selected_func}' in '{original_path_func}'.")
                                    else:
                                        logger.warning("No best alternative was selected from the API-generated options.")
                                else:
                                    logger.info(f"No alternatives were generated for '{selected_func}' using the API.")
                                    
                            except Exception as e:
                                logger.error(f"Error generating or selecting API alternatives for '{selected_func}': {e}")
                                logger.exception("Full traceback for API alternative generation/selection error:")
                        else:
                            # This case should ideally not be reached if the top `if (analyzer_instance_for_complexity or using_metta_generator)` is true
                            # but as a safeguard:
                            logger.info("No optimization engine (API or MeTTa) configured or initialized for generating alternatives.")
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

def run_evolve_command():
    """
    Run the basic evolution test from metta_generator.base
    """
    logger.info(f"Running 'evolve' command - testing basic evolution system")
    
    try:
        from metta_generator.evolution.semantic_evolution import demonstrate_semantic_evolution
        logger.info("Starting semantic evolution demonstration...")
        success = demonstrate_semantic_evolution()
            
        if success:
            logger.info("Semantic evolution demonstration completed successfully")
        else:
            logger.warning("Semantic evolution demonstration completed with issues")
                
    except ImportError as e:
        logger.error(f"Semantic evolution not available: {e}")

    logger.info(f"'evolve' command complete.")

def run_visualize_command():
    """
    Run the enhanced donor generation visualization using a demonstration function.
    This command demonstrates the approach with the find_max_in_range function.
    """
    logger.info(f"Running 'visualize' command for demonstration purposes")
    logger.info(f"Note: This command demonstrates the donor generation approach using a predefined function")

    # Define the demonstration function directly
    def find_max_in_range(numbers, start_idx, end_idx):
        """Find the maximum value in a list within a specific range."""
        if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
            return None
                
        max_val = numbers[start_idx]
        for i in range(start_idx + 1, end_idx):
            if numbers[i] > max_val:
                max_val = numbers[i]
                
        return max_val

    logger.info(f"Demonstrating enhanced donor generation with: {find_max_in_range.__name__}")
    logger.info(f"Function signature: find_max_in_range(numbers, start_idx, end_idx)")
    logger.info(f"Purpose: Find the maximum value in a list within a specific range")

    # Initialize and run the Enhanced MeTTa Donor Generation Visualizer
    try:
        # Import the enhanced visualizer
        from visualizer import EnhancedDonorGenerationVisualizer
        
        # Determine ontology file path
        ontology_file_path = os.path.join(_WORKSPACE_ROOT, full_analyzer.ONTOLOGY_PATH)
        ontology_file = ontology_file_path if os.path.exists(ontology_file_path) else None
        
        if ontology_file:
            logger.info(f"Using ontology file: {ontology_file}")
        else:
            logger.warning(f"Ontology file not found at {ontology_file_path}. Using defaults.")
        
        # Create the enhanced visualizer with the demo function
        visualizer = EnhancedDonorGenerationVisualizer(find_max_in_range, ontology_file)
        
        # Customize plot save directory
        plot_base_dir = "evolution_plots"
        func_plot_dir = os.path.join(plot_base_dir, f"{find_max_in_range.__name__}_demo")
        visualizer.plot_save_dir = func_plot_dir
        visualizer._ensure_plot_directory()

        logger.info(f"Starting enhanced donor evolution demonstration for '{find_max_in_range.__name__}'.")
        logger.info(f"Using ModularMettaDonorGenerator with specialized generators:")
        if hasattr(visualizer, 'metta_generator') and visualizer.metta_generator:
            generator_count = len(visualizer.metta_generator.registry.generators)
            strategy_count = len(visualizer.metta_generator.registry.get_supported_strategies())
            logger.info(f"  - {generator_count} registered generators")
            logger.info(f"  - {strategy_count} supported strategies")
        else:
            logger.info(f"  - Running in simulation mode (MeTTa components not available)")
        logger.info(f"Demonstration plots and data will be saved in: {os.path.abspath(func_plot_dir)}")

        # Run the evolution process with demonstration settings
        successful_candidates = visualizer.run_evolution_process(
            max_iterations=6,  # Reduced for demo
            target_success_rate=0.7  # Slightly lower target for demo
        )
        
        # Show comprehensive final summary
        visualizer.show_final_summary(successful_candidates)
        
        # Save demonstration plots and data
        final_plot_file = visualizer.save_final_plot()
        if final_plot_file:
             logger.info(f"Final demonstration plot saved: {os.path.abspath(final_plot_file)}")
        
        visualizer.save_strategy_analysis_plots()
        
        data_filename = f"{find_max_in_range.__name__}_demo_evolution_data.json"
        full_data_path = os.path.join(func_plot_dir, data_filename)
        visualizer.save_evolution_data_with_code(full_data_path)
        logger.info(f"Demonstration evolution data saved to: {os.path.abspath(full_data_path)}")
        
        # Log summary of results for the demonstration
        if successful_candidates:
            logger.info(f"🎉 Demonstration successful: {len(successful_candidates)} successful candidates found")
            
            # Log top successful candidates
            logger.info("Top successful candidates from demonstration:")
            for i, (candidate, event) in enumerate(successful_candidates[:3], 1):
                logger.info(f"  {i}. {candidate['name']}")
                logger.info(f"     Strategy: {candidate['strategy']}")
                logger.info(f"     Success Rate: {event.success_rate:.1%}")
                logger.info(f"     Generator: {event.generator_used}")
            
            # Show a sample of the best candidate's code
            if successful_candidates:
                best_candidate, best_event = successful_candidates[0]
                logger.info(f"\nBest candidate code preview:")
                logger.info(f"Strategy: {best_candidate['strategy']}")
                logger.info(f"Success Rate: {best_event.success_rate:.1%}")
                # Show first few lines of code
                code_lines = best_candidate['code'].split('\n')[:10]
                for line_num, line in enumerate(code_lines, 1):
                    logger.info(f"  {line_num:2d}: {line}")
                if len(best_candidate['code'].split('\n')) > 10:
                    logger.info(f"     ... (truncated, see full code in saved files)")
        else:
            logger.info("No successful candidates found in demonstration, but evolution data captured for analysis.")
        
        logger.info(f"Enhanced demonstration visualization complete.")
        logger.info(f"Output directory: {os.path.abspath(func_plot_dir)}")
        
        # Provide guidance on what the user learned
        logger.info(f"\n" + "="*60)
        logger.info(f"DEMONSTRATION COMPLETE - What you learned:")
        logger.info(f"="*60)
        logger.info(f"✓ How the ModularMettaDonorGenerator creates alternative implementations")
        logger.info(f"✓ How different strategies (operation substitution, data adaptation, etc.) work")
        logger.info(f"✓ How candidates are tested against the original function's constraints")
        logger.info(f"✓ How the system evolves toward better solutions over iterations")
        logger.info(f"✓ The visualization shows real-time progress and final analysis")
        logger.info(f"")
        logger.info(f"Next steps:")
        logger.info(f"  - Use 'generate' command on your own Python files")
        logger.info(f"  - Use 'analyze' command for complexity analysis with API key")
        logger.info(f"  - Examine the saved plots and code in: {os.path.abspath(func_plot_dir)}")

    except ImportError as e:
        logger.error(f"Could not import enhanced visualizer: {e}")
        logger.error("Please ensure the enhanced visualizer is available.")
        return
    except Exception as e:
        logger.error(f"An error occurred during demonstration visualization: {e}")
        logger.exception("Full traceback for demonstration visualization error:")

    logger.info(f"Enhanced 'visualize' demonstration command complete.")

def run_metta_generate_command(target_path: str = None):
    """
    Run MeTTa donor generation using the NEW MeTTa-Powered Modular system.
    Uses predefined test functions if no target_path, or parses functions from target file.
    """
    if target_path:
        logger.info(f"Running 'generate' command using NEW MeTTa-Powered Modular system on file: {target_path}")
    else:
        logger.info(f"Running 'generate' command using NEW MeTTa-Powered Modular system with predefined test functions")
    
    # Initialize local monitor for this command
    local_monitor = DynamicMonitor()
    ontology_file_path = os.path.join(_WORKSPACE_ROOT, full_analyzer.ONTOLOGY_PATH)
    
    if not os.path.exists(ontology_file_path):
        logger.warning(f"Ontology file not found at {ontology_file_path}. MeTTa generation might be incomplete.")
    else:
        local_monitor.load_metta_rules(ontology_file_path)

    # Determine which functions to process
    test_functions = []
    
    if target_path:
        # Parse functions from the target file
        logger.info(f"Parsing functions from target file: {target_path}")
        
        try:
            with open(target_path, 'r', encoding='utf-8') as source_file:
                file_content = source_file.read()
                tree = ast.parse(file_content, filename=target_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get function source code
                    func_start_line = node.lineno - 1
                    func_end_line = node.end_lineno if hasattr(node, 'end_lineno') else func_start_line + 10
                    
                    source_lines = file_content.split('\n')
                    
                    # Find actual end of function by looking for next function or end of file
                    actual_end = len(source_lines)
                    for i, line in enumerate(source_lines[func_end_line:], func_end_line):
                        if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                            actual_end = i
                            break
                    
                    func_source = '\n'.join(source_lines[func_start_line:actual_end])
                    
                    # Create a description for the function
                    description = f"Function from {os.path.basename(target_path)}"
                    if node.args.args:
                        params = [arg.arg for arg in node.args.args]
                        description += f" with parameters: {', '.join(params)}"
                    
                    test_functions.append((description, func_source, node.name))
                    
            logger.info(f"Found {len(test_functions)} functions in {target_path}")
            
        except Exception as e:
            logger.error(f"Error parsing Python file {target_path}: {e}")
            logger.info("Falling back to predefined test functions")
            target_path = None  # Fall back to predefined functions
    
    if not target_path or not test_functions:
        # Use predefined test functions
        logger.info("Using predefined test functions from MeTTa-powered system")
        
        def find_max_in_range(numbers, start_idx, end_idx):
            """Find the maximum value in a list within a specific range."""
            if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
                return None
                    
            max_val = numbers[start_idx]
            for i in range(start_idx + 1, end_idx):
                if numbers[i] > max_val:
                    max_val = numbers[i]
                    
            return max_val

        def clean_and_normalize_text(text):
            """Clean and normalize text input."""
            if not text or not isinstance(text, str):
                return ""
                    
            # Remove extra whitespace and convert to lowercase
            cleaned = text.strip().lower()
                    
            # Replace multiple spaces with single space
            import re
            cleaned = re.sub(r'\s+', ' ', cleaned)
                    
            return cleaned

        def calculate_moving_average(numbers, window_size):
            """Calculate moving average with specified window size."""
            if not numbers or window_size <= 0 or window_size > len(numbers):
                return []
                    
            averages = []
            for i in range(len(numbers) - window_size + 1):
                window_sum = sum(numbers[i:i + window_size])
                averages.append(window_sum / window_size)
                    
            return averages

        # Convert predefined functions to the same format
        predefined_functions = [
            ("Search Function with MeTTa Reasoning", find_max_in_range, find_max_in_range.__name__),
            ("String Processing with MeTTa Adapters", clean_and_normalize_text, clean_and_normalize_text.__name__),  
            ("Numeric Calculation with MeTTa Transformers", calculate_moving_average, calculate_moving_average.__name__)
        ]
        
        test_functions = predefined_functions
    
    logger.info(f"Testing with {len(test_functions)} functions using NEW MeTTa-Powered Modular system")
    
    # Initialize the NEW MeTTa-Powered Modular Donor Generator
    try:
        logger.info("Initializing NEW MeTTa-Powered Modular Donor Generator...")
        
        # Create the new generator with MeTTa reasoning as the core engine
        metta_generator = MeTTaPoweredModularDonorGenerator(metta_space=local_monitor.metta_space)
        
        # Load ontology for MeTTa reasoning
        if os.path.exists(ontology_file_path):
            ontology_loaded = metta_generator.load_ontology(ontology_file_path)
            if ontology_loaded:
                logger.info("   MeTTa reasoning ontology loaded successfully")
            else:
                logger.warning("  Warning: MeTTa reasoning ontology could not be loaded, using defaults")
        else:
            logger.warning(f"  Warning: Ontology file not found at {ontology_file_path}, using MeTTa defaults")
        
        # Log information about the MeTTa-powered system
        logger.info(f"   MeTTa reasoning engine: {type(metta_generator.reasoning_engine).__name__}")
        logger.info(f"   MeTTa pattern detector: {type(metta_generator.pattern_detector).__name__}")
        logger.info(f"   MeTTa strategy manager: {type(metta_generator.strategy_manager).__name__}")
        logger.info(f"   Available generators: {len(metta_generator.generators)}")
        
        for i, generator in enumerate(metta_generator.generators, 1):
            supported_strategies = generator.get_supported_strategies()
            logger.info(f"     {i}. {generator.generator_name}: {len(supported_strategies)} strategies")
            
    except Exception as e:
        logger.error(f"Error initializing NEW MeTTa-Powered Modular Donor Generator: {e}")
        logger.exception("Full traceback for NEW MeTTa generator initialization error:")
        return

    # Process each function with the NEW MeTTa-powered system
    all_results = {}
    successful_generations = 0
    best_alternatives = {}  # Store best alternative for each function
    metta_reasoning_stats = {"patterns_detected": 0, "reasoning_decisions": 0, "metta_guided_generations": 0}
    
    for test_data in test_functions:
        if len(test_data) == 3:
            test_name, func_or_source, func_name = test_data
            
            # Handle both callable functions and source code strings
            if callable(func_or_source):
                func_to_process = func_or_source
                try:
                    import inspect
                    func_source = inspect.getsource(func_or_source)
                except:
                    func_source = f"# Source not available for {func_name}"
            else:
                # func_or_source is already source code
                func_source = func_or_source
                func_to_process = func_source
        else:
            # Handle legacy format (test_name, func)
            test_name, func_to_process = test_data
            func_name = func_to_process.__name__ if callable(func_to_process) else "unknown_function"
            try:
                import inspect
                func_source = inspect.getsource(func_to_process) if callable(func_to_process) else str(func_to_process)
            except:
                func_source = f"# Source not available for {func_name}"
        
        logger.info(f"\nProcessing function: {test_name}")
        logger.info(f"  Function name: {func_name}")
        
        # Get description - try from docstring if callable, otherwise use test_name
        if callable(func_to_process) and func_to_process.__doc__:
            description = func_to_process.__doc__.strip()
        else:
            description = test_name
        logger.info(f"  Description: {description}")
        
        try:
            # Generate MeTTa donor candidates using the NEW MeTTa-powered system
            logger.info(f"  Generating candidates for '{func_name}' using NEW MeTTa-Powered Modular system...")
            
            # Use the NEW generation system with MeTTa reasoning throughout
            metta_candidates = metta_generator.generate_donors_from_function(func_to_process)
            
            if metta_candidates:
                logger.info(f"   Generated {len(metta_candidates)} MeTTa-powered candidates")
                
                # Count MeTTa reasoning usage
                metta_guided_count = sum(1 for c in metta_candidates 
                                       if any(prop in c.get('properties', []) for prop in ['metta-reasoned', 'reasoning-guided']))
                metta_reasoning_stats["metta_guided_generations"] += metta_guided_count
                
                # Store results with MeTTa reasoning attribution
                all_results[func_name] = {
                    "test_name": test_name,
                    "function_source": func_source,
                    "candidates": metta_candidates,
                    "generation_success": True,
                    "metta_reasoning_used": metta_guided_count,
                    "generators_used": [
                        candidate.get('generator_used', candidate['strategy']) 
                        for candidate in metta_candidates
                    ],
                    "metta_reasoning_traces": [
                        candidate.get('metta_reasoning_trace', [])
                        for candidate in metta_candidates
                    ]
                }
                
                # Find and store the best alternative
                best_candidate = metta_candidates[0] if metta_candidates else None
                if best_candidate:
                    best_alternatives[func_name] = best_candidate
                    metta_score = best_candidate.get('metta_score', 'N/A')
                    logger.info(f"   Best alternative: {best_candidate['name']} (final score: {best_candidate['final_score']:.2f}, MeTTa score: {metta_score})")
                
                # Log top candidates with MeTTa reasoning information
                for i, candidate in enumerate(metta_candidates[:3], 1):
                    generator_used = candidate.get('generator_used', 'UnknownGenerator')
                    metta_score = candidate.get('metta_score', 'N/A')
                    metta_traces = candidate.get('metta_reasoning_trace', [])
                    
                    logger.info(f"    {i}. {candidate['name']}")
                    logger.info(f"       Generated by: {generator_used}")
                    logger.info(f"       Strategy: {candidate['strategy']}")
                    logger.info(f"       Final Score: {candidate['final_score']:.2f}")
                    if metta_score != 'N/A':
                        logger.info(f"       MeTTa Score: {metta_score:.2f}")
                    logger.info(f"       Description: {candidate['description']}")
                    logger.info(f"       Pattern Family: {candidate['pattern_family']}")
                    
                    # Show MeTTa derivation if available
                    if candidate.get('metta_derivation'):
                        logger.info(f"       MeTTa Derivation: {candidate['metta_derivation'][0]}")
                    
                    # Show MeTTa reasoning traces
                    if metta_traces:
                        logger.info(f"       MeTTa Reasoning: {', '.join(metta_traces[:2])}")
                
                successful_generations += 1
            else:
                logger.warning(f"   No MeTTa-powered candidates generated for '{func_name}'")
                all_results[func_name] = {
                    "test_name": test_name,
                    "function_source": func_source,
                    "candidates": [],
                    "generation_success": False,
                    "metta_reasoning_used": 0
                }
                
        except Exception as e:
            logger.error(f"   Error generating candidates for '{func_name}': {e}")
            logger.exception("Full traceback for candidate generation error:")
            all_results[func_name] = {
                "test_name": test_name,
                "function_source": "Error retrieving source",
                "candidates": [],
                "generation_success": False,
                "metta_reasoning_used": 0,
                "error": str(e)
            }

    # Generate enhanced summary report with MeTTa reasoning statistics
    logger.info(f"\n" + "="*70)
    if target_path:
        logger.info(f"NEW METTA-POWERED MODULAR DONOR GENERATION SUMMARY (FILE: {os.path.basename(target_path)})")
    else:
        logger.info(f"NEW METTA-POWERED MODULAR DONOR GENERATION SUMMARY (PREDEFINED FUNCTIONS)")
    logger.info(f"="*70)
    logger.info(f"Functions processed: {len(test_functions)}")
    logger.info(f"Successful generations: {successful_generations}")
    logger.info(f"Failed generations: {len(test_functions) - successful_generations}")
    
    # MeTTa reasoning statistics
    total_metta_guided = sum(result.get("metta_reasoning_used", 0) for result in all_results.values())
    logger.info(f"\nMeTTa Reasoning Statistics:")
    logger.info(f"  Total MeTTa-guided candidates: {total_metta_guided}")
    if successful_generations > 0:
        avg_metta_per_func = total_metta_guided / successful_generations
        logger.info(f"  Average MeTTa-guided per function: {avg_metta_per_func:.1f}")
    
    # Generator usage statistics
    generator_stats = {}
    strategy_stats = {}
    total_candidates = 0
    
    for func_name, result in all_results.items():
        if result["generation_success"]:
            for candidate in result["candidates"]:
                total_candidates += 1
                
                # Track generator usage
                generator = candidate.get('generator_used', candidate['strategy'])
                generator_stats[generator] = generator_stats.get(generator, 0) + 1
                
                # Track strategy usage
                strategy = candidate['strategy']
                strategy_stats[strategy] = strategy_stats.get(strategy, 0) + 1
    
    logger.info(f"\nMeTTa-Powered Generator Usage Statistics:")
    for generator, count in sorted(generator_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_candidates * 100) if total_candidates > 0 else 0
        logger.info(f"  {generator}: {count} candidates ({percentage:.1f}%)")
    
    logger.info(f"\nMeTTa-Guided Strategy Usage Statistics:")
    for strategy, count in sorted(strategy_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_candidates * 100) if total_candidates > 0 else 0
        logger.info(f"  {strategy}: {count} candidates ({percentage:.1f}%)")
    
    # Show best alternatives prominently with MeTTa reasoning details
    if best_alternatives:
        logger.info(f"\n" + "="*70)
        logger.info(f"BEST METTA-REASONED ALTERNATIVES")
        logger.info(f"="*70)
        
        for func_name, best_candidate in best_alternatives.items():
            result = all_results[func_name]
            metta_score = best_candidate.get('metta_score', 'N/A')
            metta_traces = best_candidate.get('metta_reasoning_trace', [])
            
            logger.info(f"\nFunction: {func_name}")
            logger.info(f"Test: {result['test_name']}")
            logger.info(f"Best Alternative: {best_candidate['name']}")
            logger.info(f"Generated by: {best_candidate.get('generator_used', 'Unknown')}")
            logger.info(f"Strategy: {best_candidate['strategy']}")
            logger.info(f"Final Score: {best_candidate['final_score']:.2f}")
            if metta_score != 'N/A':
                logger.info(f"MeTTa Score: {metta_score:.2f}")
            logger.info(f"Confidence: {best_candidate['confidence']:.2f}")
            logger.info(f"Description: {best_candidate['description']}")
            logger.info(f"Pattern Family: {best_candidate['pattern_family']}")
            logger.info(f"Properties: {', '.join(best_candidate['properties'])}")
            
            # Show MeTTa reasoning details
            if best_candidate.get('metta_derivation'):
                logger.info(f"MeTTa Derivation: {best_candidate['metta_derivation'][0]}")
            if metta_traces:
                logger.info(f"MeTTa Reasoning Traces: {', '.join(metta_traces)}")
            
            logger.info(f"\nBest Alternative Code:")
            logger.info(f"-" * 50)
            code_lines = best_candidate['code'].split('\n')
            for i, line in enumerate(code_lines, 1):
                logger.info(f"{i:3d}: {line}")
            logger.info(f"-" * 50)
    
    # Show detailed results for each function with MeTTa reasoning info
    logger.info(f"\n" + "="*70)
    logger.info(f"DETAILED PER-FUNCTION METTA RESULTS")
    logger.info(f"="*70)
    
    for func_name, result in all_results.items():
        logger.info(f"\n{result['test_name']} ({func_name}):")
        if result["generation_success"]:
            candidates = result["candidates"]
            generators_used = set(result.get("generators_used", []))
            metta_reasoning_used = result.get("metta_reasoning_used", 0)
            
            logger.info(f"   {len(candidates)} candidates generated successfully")
            logger.info(f"   MeTTa-guided candidates: {metta_reasoning_used}")
            logger.info(f"   Generators used: {', '.join(generators_used)}")
            
            if candidates:
                best_candidate = candidates[0]
                metta_score = best_candidate.get('metta_score', 'N/A')
                logger.info(f"   Best: {best_candidate['name']} (final score: {best_candidate['final_score']:.2f}")
                if metta_score != 'N/A':
                    logger.info(f"         MeTTa score: {metta_score:.2f})")
                else:
                    logger.info(")")
                logger.info(f"   Best strategy: {best_candidate['strategy']}")
                logger.info(f"   Best generator: {best_candidate.get('generator_used', 'Unknown')}")
        else:
            if "error" in result:
                logger.info(f"   Error: {result['error']}")
            else:
                logger.info(f"   No candidates generated")

    # Save detailed results to file with MeTTa reasoning information
    try:
        output_dir = os.path.join(_WORKSPACE_ROOT, "metta_generation_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save enhanced summary report with MeTTa reasoning stats
        if target_path:
            summary_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(target_path))[0]}_metta_powered_summary.txt")
        else:
            summary_file = os.path.join(output_dir, f"predefined_functions_metta_powered_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"NEW MeTTa-Powered Modular Donor Generation Summary\n")
            f.write(f"{'='*70}\n")
            if target_path:
                f.write(f"Source file: {target_path}\n")
            else:
                f.write(f"Source: Predefined test functions\n")
            f.write(f"Generated on: {time.ctime()}\n")
            f.write(f"Functions processed: {len(test_functions)}\n")
            f.write(f"Successful generations: {successful_generations}\n")
            f.write(f"Total candidates: {total_candidates}\n")
            f.write(f"MeTTa-guided candidates: {total_metta_guided}\n\n")
            
            # MeTTa reasoning statistics
            f.write(f"MeTTa Reasoning Statistics:\n")
            f.write(f"{'='*35}\n")
            f.write(f"Total MeTTa-guided candidates: {total_metta_guided}\n")
            if successful_generations > 0:
                avg_metta_per_func = total_metta_guided / successful_generations
                f.write(f"Average MeTTa-guided per function: {avg_metta_per_func:.1f}\n")
            
            # Generator statistics
            f.write(f"\nMeTTa-Powered Generator Usage:\n")
            f.write(f"{'='*40}\n")
            for generator, count in sorted(generator_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_candidates * 100) if total_candidates > 0 else 0
                f.write(f"{generator}: {count} candidates ({percentage:.1f}%)\n")
            
            f.write(f"\nMeTTa-Guided Strategy Usage:\n")
            f.write(f"{'='*35}\n")
            for strategy, count in sorted(strategy_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_candidates * 100) if total_candidates > 0 else 0
                f.write(f"{strategy}: {count} candidates ({percentage:.1f}%)\n")
            
            # Best alternatives section with MeTTa reasoning
            f.write(f"\nBest MeTTa-Reasoned Alternatives:\n")
            f.write(f"{'='*45}\n")
            for func_name, best_candidate in best_alternatives.items():
                result = all_results[func_name]
                metta_score = best_candidate.get('metta_score', 'N/A')
                metta_traces = best_candidate.get('metta_reasoning_trace', [])
                
                f.write(f"\nFunction: {func_name}\n")
                f.write(f"Test: {result['test_name']}\n")
                f.write(f"Best Alternative: {best_candidate['name']}\n")
                f.write(f"Generated by: {best_candidate.get('generator_used', 'Unknown')}\n")
                f.write(f"Strategy: {best_candidate['strategy']}\n")
                f.write(f"Final Score: {best_candidate['final_score']:.2f}\n")
                if metta_score != 'N/A':
                    f.write(f"MeTTa Score: {metta_score:.2f}\n")
                f.write(f"Description: {best_candidate['description']}\n")
                f.write(f"Properties: {', '.join(best_candidate['properties'])}\n")
                
                # MeTTa reasoning details
                if best_candidate.get('metta_derivation'):
                    f.write(f"MeTTa Derivation: {best_candidate['metta_derivation'][0]}\n")
                if metta_traces:
                    f.write(f"MeTTa Reasoning Traces: {', '.join(metta_traces)}\n")
                    
                f.write(f"\nCode:\n{best_candidate['code']}\n")
                f.write(f"{'-'*60}\n")
            
            # Detailed results with MeTTa reasoning
            f.write(f"\nDetailed MeTTa Results:\n")
            f.write(f"{'='*35}\n")
            
            for func_name, result in all_results.items():
                f.write(f"\nTest: {result['test_name']}\n")
                f.write(f"Function: {func_name}\n")
                f.write(f"{'='*60}\n")
                if result["generation_success"]:
                    candidates = result["candidates"]
                    generators_used = set(result.get("generators_used", []))
                    metta_reasoning_used = result.get("metta_reasoning_used", 0)
                    
                    f.write(f"Candidates generated: {len(candidates)}\n")
                    f.write(f"MeTTa-guided candidates: {metta_reasoning_used}\n")
                    f.write(f"Generators used: {', '.join(generators_used)}\n")
                    
                    for i, candidate in enumerate(candidates, 1):
                        generator_used = candidate.get('generator_used', 'Unknown')
                        metta_score = candidate.get('metta_score', 'N/A')
                        metta_traces = candidate.get('metta_reasoning_trace', [])
                        
                        f.write(f"\n{i}. {candidate['name']}\n")
                        f.write(f"   Generated by: {generator_used}\n")
                        f.write(f"   Strategy: {candidate['strategy']}\n")
                        f.write(f"   Final Score: {candidate['final_score']:.2f}\n")
                        if metta_score != 'N/A':
                            f.write(f"   MeTTa Score: {metta_score:.2f}\n")
                        f.write(f"   Pattern Family: {candidate['pattern_family']}\n")
                        f.write(f"   Description: {candidate['description']}\n")
                        f.write(f"   Properties: {', '.join(candidate['properties'])}\n")
                        
                        # MeTTa reasoning evidence
                        if candidate.get('metta_derivation'):
                            f.write(f"   MeTTa Derivation: {candidate['metta_derivation'][0]}\n")
                        if metta_traces:
                            f.write(f"   MeTTa Reasoning: {', '.join(metta_traces)}\n")
                else:
                    f.write("No candidates generated\n")
                    if "error" in result:
                        f.write(f"Error: {result['error']}\n")
        
        # Save detailed code for successful generations with MeTTa reasoning attribution
        for func_name, result in all_results.items():
            if result["generation_success"] and result["candidates"]:
                func_output_dir = os.path.join(output_dir, func_name)
                os.makedirs(func_output_dir, exist_ok=True)
                
                for i, candidate in enumerate(result["candidates"], 1):
                    candidate_filename = f"{candidate['name']}.py"
                    candidate_path = os.path.join(func_output_dir, candidate_filename)
                    
                    generator_used = candidate.get('generator_used', 'UnknownGenerator')
                    metta_score = candidate.get('metta_score', 'N/A')
                    metta_traces = candidate.get('metta_reasoning_trace', [])
                    
                    with open(candidate_path, 'w') as f:
                        f.write(f"# NEW MeTTa-Powered Donor Candidate: {candidate['name']}\n")
                        f.write(f"# Original function: {func_name}\n")
                        f.write(f"# Test: {result['test_name']}\n")
                        f.write(f"# Generated by: {generator_used}\n")
                        f.write(f"# Strategy: {candidate['strategy']}\n")
                        f.write(f"# Pattern Family: {candidate['pattern_family']}\n")
                        f.write(f"# Final Score: {candidate['final_score']:.2f}\n")
                        if metta_score != 'N/A':
                            f.write(f"# MeTTa Score: {metta_score:.2f}\n")
                        f.write(f"# Confidence: {candidate['confidence']:.2f}\n")
                        f.write(f"# Description: {candidate['description']}\n")
                        f.write(f"# Properties: {', '.join(candidate['properties'])}\n")
                        if candidate.get('metta_derivation'):
                            f.write(f"# MeTTa Derivation: {candidate['metta_derivation'][0]}\n")
                        if metta_traces:
                            f.write(f"# MeTTa Reasoning: {', '.join(metta_traces)}\n")
                        f.write(f"# Generated on: {time.ctime()}\n")
                        f.write(f"\n{candidate['code']}\n")
        
        logger.info(f"\nDetailed results saved to: {output_dir}")
        logger.info(f"Summary report: {summary_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

    # DIRECT EXPORT: Export MeTTa atoms from generation with MeTTa reasoning info
    try:
        if not os.path.exists(_INTERMEDIATE_EXPORT_DIR):
            os.makedirs(_INTERMEDIATE_EXPORT_DIR, exist_ok=True)
        
        metta_generate_export_file = os.path.join(_INTERMEDIATE_EXPORT_DIR, "metta_powered_generate_export.metta")
        logger.info(f"Exporting MeTTa-powered generation atoms to: {metta_generate_export_file}")
        
        # Export generation results to MeTTa format with MeTTa reasoning information
        export_success = export_from_metta_generation(all_results, metta_generate_export_file)
        
        if export_success:
            verification = verify_export(metta_generate_export_file)
            if verification["success"]:
                logger.info(f"MeTTa-powered generation atoms exported successfully: {verification['atom_count']} atoms ({verification['file_size']} bytes)")
            else:
                logger.warning(f"Export verification failed: {verification.get('error', 'Unknown error')}")
        else:
            logger.warning(f"Failed to export MeTTa-powered generation atoms.")
            
    except Exception as e:
        logger.error(f"Error during MeTTa-powered generation atom export: {e}")

    logger.info(f"\n" + "="*70)
    logger.info(f"NEW METTA-POWERED GENERATE COMMAND COMPLETE")
    logger.info(f"="*70)
    logger.info(f"Successfully demonstrated NEW MeTTa-powered donor generation")
    logger.info(f"Total candidates generated: {total_candidates}")
    logger.info(f"MeTTa-guided candidates: {total_metta_guided}")
    logger.info(f"Best alternatives identified for {len(best_alternatives)} functions")
    logger.info(f"MeTTa-powered generators used: {len(generator_stats)}")
    logger.info(f"MeTTa-guided strategies applied: {len(strategy_stats)}")
    logger.info(f"Results saved to: {output_dir}")
    
    # Provide guidance on the NEW MeTTa-powered system
    logger.info(f"\n" + "="*70)
    logger.info(f"NEW METTA-POWERED SYSTEM FEATURES DEMONSTRATED:")
    logger.info(f"="*70)
    logger.info(f"✓ MeTTa reasoning engine for intelligent pattern detection")
    logger.info(f"✓ MeTTa-guided strategy selection and applicability checking")
    logger.info(f"✓ Symbolic reasoning fallbacks when MeTTa execution fails")
    logger.info(f"✓ MeTTa-derived transformations with reasoning traces")
    logger.info(f"✓ Enhanced candidate scoring with MeTTa confidence metrics")
    logger.info(f"✓ Learning and adaptation from generation feedback")
    logger.info(f"")
    logger.info(f"Key advantages over previous system:")
    logger.info(f"  - Symbolic reasoning as core decision-making engine")
    logger.info(f"  - Pattern detection guided by formal logic")
    logger.info(f"  - Transformation safety verified through MeTTa rules")
    logger.info(f"  - Detailed reasoning traces for transparency")
    logger.info(f"  - Self-improving through learning rules")
    logger.info(f"")
    logger.info(f"Next steps:")
    logger.info(f"  - Review MeTTa reasoning traces in generated candidates")
    logger.info(f"  - Examine the symbolic reasoning fallbacks for robustness")
    logger.info(f"  - Use 'export' command to save MeTTa reasoning atomspace")
    logger.info(f"  - Compare results with 'analyze' command for API-based optimization")

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
               f"  python cli.py export /path/to/output/atomspace.metta\n"
               f"  python cli.py visualize /path/to/your/file.py",
        formatter_class=ColoredHelpFormatter # Use the custom formatter
    )
    parser.add_argument(
        "command", 
        choices=["summary", "analyze", "import", "export", "visualize", "generate", "evolve"],
        help=(
            "The command to execute. Each command has specific behaviors:\n"
            "  summary: Codebase structure, patterns, and concepts analysis.\n"
            "  analyze: Function complexity analysis and AI-driven optimization.\n"
            "  generate: MeTTa-based donor generation for all functions in a file.\n"
            "  visualize: DEMONSTRATION - Show donor evolution process with example function.\n"
            "  import:  Import atoms from an external .metta file.\n"
            "  export:  Export a consolidated MeTTa atomspace."
        )
    )
    parser.add_argument(
        "path", 
        help="The path argument, meaning depends on the command:\n"
            "  For 'summary', 'analyze', 'generate': Path to the target Python file or directory to analyze.\n"
            "  For 'analyze': Path to the target Python file or directory to analyze.\n"
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
    elif args.command == "generate":
        if not os.path.exists(args.path):
            logger.error(f"Error: The input path '{args.path}' for command 'generate' does not exist. Please provide a valid Python file path.")
            sys.exit(1)
        if not os.path.isfile(args.path) or not args.path.endswith(".py"):
            logger.error(f"Error: The input path '{args.path}' for command 'generate' must be a Python file (e.g., example.py).")
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
    elif args.command == "visualize":
        run_visualize_command()
    elif args.command == "generate":
        run_metta_generate_command(args.path)
    elif args.command == "evolve":
        run_evolve_command()
    else:
        logger.error(f"Unknown command: {args.command}") 
        parser.print_help()
        sys.exit(1)

    logger.info("CLI command execution finished.")
    sys.exit(0)