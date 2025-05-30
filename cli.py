import os
import sys
import logging
from typing import Union, Callable, List, Optional
from io import StringIO
import threading
import asyncio

# --- Textual Imports ---
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container, VerticalScroll
from textual.widgets import Header, Footer, Button, Input, Label, RadioSet, RadioButton, Static, Log, SelectionList
from textual.screen import ModalScreen
from textual.reactive import reactive
from textual.worker import Worker, WorkerState
import time

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

# Setup logger for this module
logger = get_logger(__name__)

# --- Textual Logging Handler ---
class TextualLogHandler(logging.Handler):
    def __init__(self, log_widget: Log):
        super().__init__()
        self.log_widget = log_widget

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            # Use call_from_thread to safely update the widget from a different thread
            self.log_widget.app.call_from_thread(self.log_widget.write_line, msg)
        except Exception:
            self.handleError(record)

def setup_tui_logger(log_widget: Log):
    """Remove existing handlers and add TextualLogHandler to the root logger."""
    # Clear existing handlers from the root logger
    # for handler in logging.root.handlers[:]:
    #     logging.root.removeHandler(handler)
    
    # Clear existing handlers from our specific logger
    # This is safer than clearing root logger if other libraries use logging
    global logger 
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create Textual handler
    textual_handler = TextualLogHandler(log_widget)
    textual_handler.setFormatter(formatter)
    
    # Add handler to our logger
    logger.addHandler(textual_handler)
    logger.setLevel(logging.INFO) # Set desired level for TUI display

    # Also configure the root logger to capture logs from other modules if desired
    # but be careful not to disrupt Textual's own logging too much.
    # For now, focusing on our application's logger.
    # logging.getLogger().addHandler(textual_handler) # Optionally add to root logger
    # logging.getLogger().setLevel(logging.INFO)


def run_summary_command(target_path: str, app_logger: logging.Logger):
    app_logger.info(f"Running 'summary' command for: {target_path}")

    local_monitor = DynamicMonitor()
    ontology_file_path = os.path.join(_WORKSPACE_ROOT, full_analyzer.ONTOLOGY_PATH)
    
    if not os.path.exists(ontology_file_path):
        app_logger.warning(f"Ontology file not found at {ontology_file_path}. Summary analysis might be incomplete.")
    else:
        local_monitor.load_metta_rules(ontology_file_path)

    original_global_monitor_full = getattr(full_analyzer, 'monitor', None)
    full_analyzer.monitor = local_monitor

    try:
        app_logger.info("Starting comprehensive codebase summary analysis...")
        app_logger.info(f"Target: {target_path}")
        app_logger.info("Analyzing codebase structure...")
        full_analyzer.analyze_codebase(target_path)
        app_logger.info("Summary analysis completed successfully.")
        app_logger.info("Analyzing temporal aspects (git history)...")
        full_analyzer.analyze_temporal_evolution(target_path, monitor=local_monitor)
        app_logger.info("Analyzing structural patterns...")
        full_analyzer.analyze_structural_patterns()
        app_logger.info("Analyzing domain concepts...")
        full_analyzer.analyze_domain_concepts()
    except Exception as e:
        app_logger.error(f"An error occurred during summary analysis: {e}")
        app_logger.exception("Full traceback for summary analysis error:")
    finally:
        if original_global_monitor_full is not None:
            full_analyzer.monitor = original_global_monitor_full
        elif hasattr(full_analyzer, 'monitor'):
            delattr(full_analyzer, 'monitor')

        try:
            if not os.path.exists(_INTERMEDIATE_EXPORT_DIR):
                os.makedirs(_INTERMEDIATE_EXPORT_DIR, exist_ok=True)
                app_logger.info(f"Created intermediate export directory: {_INTERMEDIATE_EXPORT_DIR}")
            
            app_logger.info(f"Directly exporting static analysis atoms to: {_SUMMARY_EXPORT_FILE}")
            export_success = export_from_summary_analysis(target_path, _SUMMARY_EXPORT_FILE)
            
            if export_success:
                verification = verify_export(_SUMMARY_EXPORT_FILE)
                if verification["success"]:
                    app_logger.info(f"Summary atoms directly exported successfully: {verification['atom_count']} atoms ({verification['file_size']} bytes)")
                else:
                    app_logger.warning(f"Export verification failed: {verification.get('error', 'Unknown error')}")
            else:
                app_logger.warning(f"Failed to directly export summary atoms.")
        except Exception as e:
            app_logger.error(f"Error during direct summary atom export: {e}")
            app_logger.exception("Full traceback for summary export error:")

    app_logger.info(f"Summary analysis for {target_path} complete.")

# --- Function Selection Modal ---
class FunctionSelectionModal(ModalScreen[Optional[str]]):
    """A modal screen to select a function for optimization.
    Returns the selected function name, 'skip', or None if cancelled implicitly.
    """

    DEFAULT_CSS = """
    FunctionSelectionModal {
        align: center middle;
    }
    #function_selection_dialog {
        width: 80%;
        max-width: 70;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: $surface-lighten-1;
        border: thick $primary;
        border-title-color: $text;
        border-title-style: bold;
    }
    #function_select_list {
        height: auto;
        max-height: 15;
        border: solid $primary-darken-1;
        margin-bottom: 1;
    }
    .modal_buttons {
        width: 100%;
        height: auto;
        align-horizontal: right;
        padding-top: 1;
    }
    .modal_buttons Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        functions: List[str],
        file_path: str,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        super().__init__(name, id, classes)
        self.functions = functions
        self.file_path = file_path
        self._selected_function_in_list: Optional[str] = None # Stores value from SelectionList
        self.border_title = f"Optimize Function in {os.path.basename(file_path)}"

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="function_selection_dialog") as vs:
            vs.border_title = self.border_title
            yield Label("Select a function to attempt optimization, or skip:")
            if self.functions:
                # Value for SelectionList items is the function name itself or 'skip'
                selection_options = [(func_name, func_name) for func_name in self.functions]
                selection_options.append(("[Skip optimization for this file]", "skip"))
                
                yield SelectionList[str](
                    *selection_options,
                    id="function_select_list"
                )
            else:
                yield Static("No functions found in this file to offer for optimization.")
            
            with Horizontal(classes="modal_buttons"):
                # Disable select button initially if no functions or if nothing is selected
                can_select = bool(self.functions) # True if there are functions to select from
                yield Button("Select", variant="primary", id="select_function_button", disabled=not can_select)
                yield Button("Skip All", id="skip_function_button") # Changed label for clarity

    def on_mount(self) -> None:
        if self.functions:
            slist = self.query_one(SelectionList)
            slist.focus()
            # Pre-select "skip" or first function? For now, no pre-selection.
            self.query_one("#select_function_button", Button).disabled = True # Disabled until a selection
        else:
            self.query_one("#skip_function_button", Button).focus()

    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged[str]) -> None:
        if event.selection_list.selected:
            self._selected_function_in_list = event.selection_list.selected[0]
            self.query_one("#select_function_button", Button).disabled = False
        else:
            self._selected_function_in_list = None
            self.query_one("#select_function_button", Button).disabled = True

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "select_function_button":
            if self._selected_function_in_list:
                self.dismiss(self._selected_function_in_list) # Returns selected func_name or 'skip'
            # else: button should be disabled
        elif event.button.id == "skip_function_button":
            self.dismiss("skip")

# --- Modified run_analyze_command ---
async def run_analyze_command(target_path: str, api_key: Optional[str] = None, app_logger: logging.Logger = logger, app: Optional[App] = None):
    app_logger.info(f"Running 'analyze' command for: {target_path} (API key: {'Provided' if api_key else 'Not provided'})")
    
    local_monitor = DynamicMonitor()
    ontology_file_path = os.path.join(_WORKSPACE_ROOT, complexity_analyzer_module.ONTOLOGY_PATH)
    if not os.path.exists(ontology_file_path):
        app_logger.warning(f"Ontology file not found at {ontology_file_path}. Analysis might be incomplete.")
    else:
        local_monitor.load_metta_rules(ontology_file_path)

    original_global_monitor_complexity = getattr(complexity_analyzer_module, 'monitor', None)
    complexity_analyzer_module.monitor = local_monitor
    
    analyzer_instance_for_complexity = None
    if api_key:
        app_logger.debug("API key provided. Attempting to initialize ImmuneSystemProofAnalyzer...")
        try:
            app_logger.info("Initializing proof-guided implementation generator...")
            analyzer_instance_for_complexity = ImmuneSystemProofAnalyzer(metta_space=local_monitor.metta_space, api_key=api_key)
            analyzer_instance_for_complexity = complexity_analyzer_module.integrate_with_immune_system(analyzer_instance_for_complexity)
            app_logger.info("Proof-guided implementation generator initialized successfully.")
        except Exception as e:
            app_logger.error(f"Error initializing ImmuneSystemProofAnalyzer: {e}")
            app_logger.exception("Full traceback for ImmuneSystemProofAnalyzer initialization error:")
            app_logger.warning("Could not initialize proof-guided implementation generator. Proceeding with complexity analysis only.")
            analyzer_instance_for_complexity = None 
    else:
        app_logger.warning("No API key provided. AI-driven optimization suggestions will be unavailable. Proceeding with complexity analysis only.")

    analysis_results = {
        "complexity_metrics": {},
        "optimization_suggestions": []
    }

    try:
        app_logger.info(f"Running base complexity analysis for {target_path}...")
        complexity_analyzer_module.analyze_function_complexity_and_optimize(target_path, None) # Run base analysis first without interactive part
        app_logger.info("Base complexity analysis scan complete.")
    except Exception as e:
        app_logger.error(f"An error occurred during base complexity analysis: {e}")
        app_logger.exception("Full traceback for base complexity analysis error:")
    # Export logic remains the same, runs after base analysis and potential optimization
    # ... (export logic for _ANALYZE_EXPORT_FILE using analysis_results)

    # --- Interactive Optimization Part --- 
    selected_func_name: Optional[str] = None
    if app and analyzer_instance_for_complexity and os.path.isfile(target_path):
        app_logger.info(f"Preparing for interactive function optimization for file: {target_path}")
        
        # Decompose file to get function list (can this be slow? If so, workerize)
        decomposed_file_info = None
        try:
            decomposed_file_info = complexity_analyzer_module.decompose_file(target_path)
        except Exception as e:
            app_logger.error(f"Failed to decompose file {target_path}: {e}")
            app_logger.exception("Decomposition error traceback:")

        if decomposed_file_info and "functions" in decomposed_file_info and decomposed_file_info["functions"]:
            functions_in_file = [f_info["name"] for f_info in decomposed_file_info["functions"]]
            
            if not functions_in_file:
                app_logger.info("No functions found in the decomposed file to offer for optimization.")
            else:
                app_logger.info(f"Found functions: {functions_in_file}. Opening selection modal...")
                # Show modal and wait for result
                # `push_screen_wait` is what we need here if the worker is calling it via app.call_from_thread
                # but since run_analyze_command is now async and called by the worker directly,
                # it can call app.push_screen and await the result itself.
                
                # Callback to handle modal result
                # This needs to be an awaitable call if push_screen is used like this.
                # result = await app.push_screen_wait(FunctionSelectionModal(functions=functions_in_file, file_path=target_path))
                # selected_func_name = result
                
                # Simpler: Define a future, push screen, await future.
                # Or, make the worker responsible for this via app.call_from_thread if that's cleaner.
                # Let's try push_screen_wait directly if app context allows.
                # The worker function (`_execute_command_worker`) is async, so it can await this.
                selected_func_name = await app.push_screen_wait(FunctionSelectionModal(functions=functions_in_file, file_path=target_path))

                if selected_func_name and selected_func_name != 'skip':
                    app_logger.info(f"User selected function: '{selected_func_name}' for optimization.")
                    func_info = next((f for f in decomposed_file_info["functions"] if f["name"] == selected_func_name), None)
                    
                    if func_info and "source" in func_info:
                        func_source = func_info["source"]
                        try:
                            app_logger.info(f"Generating up to 2 alternative implementations for '{selected_func_name}'...")
                            # This call can be slow, should ideally be in a worker or async if possible
                            # For now, keeping it synchronous within this async command flow.
                            alternatives = analyzer_instance_for_complexity.generate_verified_alternatives(
                                func_source, selected_func_name, count=2)
                            
                            if alternatives:
                                app_logger.info(f"[b]--- {len(alternatives)} Alternative Implementations for [cyan]{selected_func_name}[/cyan] ---[/b]")
                                for i, alt in enumerate(alternatives, 1):
                                    app_logger.info(f"[b]Alternative {i}:[/b] Strategy: {alt.get('strategy', 'N/A')}, Success: {alt.get('success', False)}, Preserved: {alt.get('verification_result', {}).get('properties_preserved', False)}")
                                    alt_code = alt.get('alternative_function', 'No code generated')
                                    app_logger.info(f"[b]Code Alternative {i}:[/b]\n[green]{alt_code}[/green]")
                                
                                best_alt = analyzer_instance_for_complexity.select_best_alternative(alternatives)
                                if best_alt:
                                    app_logger.info("[b]--- Best Alternative Selected ---[/b]")
                                    app_logger.info(f"  Strategy: {best_alt.get('strategy', 'N/A')}")
                                    app_logger.info(f"  Properties Preserved: {best_alt.get('verification_result', {}).get('properties_preserved', False)}")
                                    best_code = best_alt.get('alternative_function', 'No code selected')
                                    app_logger.info(f"  [b]Code:[/b]\n[bright_green]{best_code}[/bright_green]")
                                    
                                    original_path_func = func_info.get('file', target_path) 
                                    optimized_dir = os.path.join(os.path.dirname(original_path_func), "optimized_code")
                                    optimized_file_name = f"{os.path.splitext(os.path.basename(original_path_func))[0]}_{selected_func_name}_optimized.py"
                                    optimized_path = os.path.join(optimized_dir, optimized_file_name)
                                    
                                    app_logger.info("[yellow]Suggestion for using this optimized implementation:[/yellow]")
                                    app_logger.info(f"  1. Ensure directory exists or create: [blue]{optimized_dir}[/blue]")
                                    app_logger.info(f"  2. Save the BEST alternative code to a new file: [blue]{optimized_path}[/blue]")
                                    app_logger.info(f"  3. Manually review and integrate this new function into your codebase.")
                                else:
                                    app_logger.warning("No best alternative was selected from the generated options.")
                            else:
                                app_logger.info(f"No alternatives were generated for '{selected_func_name}'.")
                        except Exception as e:
                            app_logger.error(f"Error generating or selecting alternatives for '{selected_func_name}': {e}")
                            app_logger.exception("Full traceback for alternative generation/selection error:")
                    else:
                        app_logger.error(f"Could not retrieve source code for '{selected_func_name}'. Skipping optimization.")
                elif selected_func_name == 'skip':
                    app_logger.info("User chose to skip function optimization for this file.")
                else: # None or unexpected
                    app_logger.info("No function selected for optimization or dialog was cancelled.")
        else:
            if decomposed_file_info is None: # Error during decomposition
                 app_logger.warning(f"Skipping interactive optimization for {target_path} due to earlier decomposition error.")
            else: # No functions found
                 app_logger.warning(f"No functions found in the analysis of {target_path} to offer for optimization, or file info is missing.")
    elif not app:
        app_logger.warning("Application context not available for interactive optimization. Skipping this step.")
    elif not analyzer_instance_for_complexity:
        app_logger.info("Proof-guided analyzer not available. Skipping interactive optimization.")
    elif not os.path.isfile(target_path):
        app_logger.info("Target path is not a file. Skipping interactive function optimization.")

    # --- Finalize and Export --- (This part should run after all potential optimizations)
    try:
        if not os.path.exists(_INTERMEDIATE_EXPORT_DIR):
            os.makedirs(_INTERMEDIATE_EXPORT_DIR, exist_ok=True)
            app_logger.info(f"Created intermediate export directory: {_INTERMEDIATE_EXPORT_DIR}")
        
        app_logger.info(f"Directly exporting complexity analysis atoms to: {_ANALYZE_EXPORT_FILE}")
        export_success = export_from_complexity_analysis(target_path, _ANALYZE_EXPORT_FILE, analysis_results)
        
        if export_success:
            verification = verify_export(_ANALYZE_EXPORT_FILE)
            if verification["success"]:
                app_logger.info(f"Analyze atoms exported: {verification['atom_count']} atoms ({verification['file_size']} bytes)")
            else:
                app_logger.warning(f"Export verification failed: {verification.get('error', 'Unknown error')}")
        else:
            app_logger.warning(f"Failed to export analyze atoms.")
    except Exception as e:
        app_logger.error(f"Error during direct analyze atom export: {e}")
        app_logger.exception("Full traceback for analyze export error:")
    finally:
        if original_global_monitor_complexity is not None:
            complexity_analyzer_module.monitor = original_global_monitor_complexity
        elif hasattr(complexity_analyzer_module, 'monitor'):
            delattr(complexity_analyzer_module, 'monitor')

    app_logger.info(f"'Analyze' command for {target_path} complete.")

def run_export_atomspace_command(output_metta_path: str, app_logger: logging.Logger = logger):
    app_logger.info(f"Running 'export' command. Output will be saved to: {output_metta_path}")

    files_to_combine = []
    
    try:
        if hasattr(full_analyzer, 'ONTOLOGY_PATH') and isinstance(full_analyzer.ONTOLOGY_PATH, str):
            ontology_file_path = os.path.join(_WORKSPACE_ROOT, full_analyzer.ONTOLOGY_PATH)
            if os.path.exists(ontology_file_path):
                files_to_combine.append(ontology_file_path)
                app_logger.info(f"Including base ontology: {ontology_file_path}")
            else:
                app_logger.warning(f"Base ontology file not found: {ontology_file_path}")
        else:
            app_logger.warning("No base ontology path available")
    except Exception as e:
        app_logger.error(f"Error checking base ontology: {e}")

    if os.path.exists(_SUMMARY_EXPORT_FILE):
        files_to_combine.append(_SUMMARY_EXPORT_FILE)
        app_logger.info(f"Including summary export: {_SUMMARY_EXPORT_FILE}")
    else:
        app_logger.info(f"Summary export file not found: {_SUMMARY_EXPORT_FILE}")

    if os.path.exists(_ANALYZE_EXPORT_FILE):
        files_to_combine.append(_ANALYZE_EXPORT_FILE)
        app_logger.info(f"Including analyze export: {_ANALYZE_EXPORT_FILE}")
    else:
        app_logger.info(f"Analyze export file not found: {_ANALYZE_EXPORT_FILE}")

    if not files_to_combine:
        app_logger.warning("No files found to export. Run 'summary' or 'analyze' commands first.")
        try:
            with open(output_metta_path, 'w') as f:
                f.write(";;\\n")
                f.write(";; MeTTa Atomspace Export\\n")
                f.write(f";; Exported: {time.ctime()}\\n")
                f.write(";; No atoms available - run 'summary' or 'analyze' commands first\\n")
                f.write(";;\\n")
            app_logger.info(f"Created empty export file: {output_metta_path}")
        except Exception as e:
            app_logger.error(f"Error creating empty export file: {e}")
        return

    app_logger.info(f"Combining {len(files_to_combine)} files into consolidated export: {output_metta_path}")
    try:
        success = combine_metta_files(files_to_combine, output_metta_path, "consolidated_export")
        
        if success:
            verification = verify_export(output_metta_path)
            if verification["success"]:
                app_logger.info(f"Consolidated atomspace successfully exported: {verification['atom_count']} atoms, {verification['file_size']} bytes")
                app_logger.info(f"Combined from {len(files_to_combine)} source files:")
                for file_path in files_to_combine:
                    app_logger.info(f"  - {os.path.basename(file_path)}")
            else:
                app_logger.warning(f"Export verification failed: {verification.get('error', 'Unknown error')}")
        else:
            app_logger.error(f"Failed to combine files into consolidated export: {output_metta_path}")
            
    except Exception as e:
        app_logger.error(f"An unexpected error occurred during consolidated export: {e}")
        app_logger.exception("Full traceback for export command error:")

    app_logger.info(f"'export' command for {output_metta_path} complete.")

def run_import_metta_command(source_file: str, target_file: str, overwrite: bool = True):
    """
    Helper function for importing one .metta file into another.
    Not used by default CLI but available if needed.
    """
    logger.info(f"Importing {source_file} into {target_file} (overwrite: {overwrite})")
    
    try:
        result = import_metta_file(source_file, target_file, overwrite)
        if result["success"]:
            logger.info(f"Import successful: {result['message']}")
        else:
            logger.error(f"Import failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error during import: {e}")


# --- Main CLI Logic ---

class ChimeraApp(App):
    """A Textual TUI for Chimera Indexer."""

    CSS_PATH = "chimera_cli.css"
    TITLE = "Chimera"

    COMMANDS = {
        "summary": "Performs a comprehensive static analysis of the codebase structure, relationships, patterns, and concepts (using exec/full_analyzer.py).",
        "analyze": "Focuses on function complexity analysis and offers potential AI-driven optimization suggestions if an API key is provided (using exec/complexity.py). The interactive part of selecting functions for optimization is now TUI-based.",
        "export": "Exports a MeTTa atomspace (typically after combining previous analysis results and an ontology) to a specified .metta file."
    }
    
    selected_command = reactive("summary", layout=True)
    path_input_placeholder = reactive("Path for summary/analyze or output for export", layout=True)
    
    BINDINGS = [("ctrl+q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.log_widget = None
        self._command_running = False
        self.help_text_widget = None # For contextual help

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Container(id="app-grid"):
            with Vertical(id="left-panel"):
                yield Label("Select Command:")
                with RadioSet(id="command_select"):
                    yield RadioButton("Summary", id="summary", value=True)
                    yield RadioButton("Analyze", id="analyze")
                    yield RadioButton("Export", id="export")
                
                # Contextual Help Area
                yield Label("Command Info:", classes="contextual-help-label") # Added label for help area
                self.help_text_widget = Static("Select a command to see its description.", id="contextual_help_text")
                yield self.help_text_widget
                
                yield Label("Target Path:", id="path_label")
                yield Input(placeholder=self.path_input_placeholder, id="path_input")
                
                yield Label("OpenAI API Key (for Analyze):", id="api_key_label", classes="hidden")
                yield Input(placeholder="sk-... or env OPENAI_API_KEY", id="api_key_input", password=True, classes="hidden")
                
                yield Button("Run Command", id="run_button", variant="primary")
            
            with Vertical(id="right-panel"):
                yield Label("Output Log:")
                self.log_widget = Log(id="output_log", highlight=True, auto_scroll=True)
                yield self.log_widget
        yield Footer()

    def on_mount(self) -> None:
        """Called when app starts."""
        if self.log_widget:
            setup_tui_logger(self.log_widget)
            logger.info("Chimera Indexer TUI started. Select a command and provide inputs.")
        else:
            print("Error: Log widget not initialized before on_mount.")

        self.update_path_placeholder()
        self.update_contextual_help() # Initial help text update
        self.query_one("#path_input", Input).focus()

    def update_contextual_help(self) -> None:
        """Updates the contextual help text based on the selected command."""
        if self.help_text_widget:
            description = self.COMMANDS.get(self.selected_command, "No description available.")
            self.help_text_widget.update(description)

    def watch_selected_command(self, old_value: str, new_value: str) -> None:
        """Called when selected_command changes."""
        self.update_path_placeholder()
        self.update_contextual_help() # Update help text when command changes
        
        api_key_input = self.query_one("#api_key_input", Input)
        api_key_label = self.query_one("#api_key_label", Label)

        if new_value == "analyze":
            api_key_input.remove_class("hidden")
            api_key_label.remove_class("hidden")
        else:
            api_key_input.add_class("hidden")
            api_key_label.add_class("hidden")
            api_key_input.value = ""

    def update_path_placeholder(self) -> None:
        """Updates the placeholder for the path input based on the selected command."""
        path_input = self.query_one("#path_input", Input)
        if self.selected_command == "export":
            path_input.placeholder = "Path to output .metta file"
        else:
            path_input.placeholder = "Path to target Python file or directory"

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle command selection change."""
        assert isinstance(event.pressed, RadioButton), "Expected RadioButton"
        
        new_command = event.pressed.id
        if new_command is None:
            logger.error("Selected radio button has no ID.")
            return

        self.selected_command = new_command
        logger.info(f"Selected command: [b]{self.selected_command}[/b]")

    def validate_path(self, path: str, command: str) -> bool:
        log_widget = self.log_widget
        if not path:
            log_widget.write_line("[bold red]Error: Path is required.[/bold red]")
            return False

        if command in ["summary", "analyze"]:
            if not os.path.exists(path):
                log_widget.write_line(f"[bold red]Error: Input path '{path}' does not exist.[/bold red]")
                return False
        elif command == "export":
            output_dir = os.path.dirname(path)
            if not output_dir: output_dir = "."
            
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    log_widget.write_line(f"Created output directory: {output_dir}")
                except OSError as e:
                    log_widget.write_line(f"[bold red]Error: Could not create output directory '{output_dir}': {e}[/bold red]")
                    return False
            if os.path.isdir(path):
                log_widget.write_line(f"[bold red]Error: Output path '{path}' for export must be a file, not a directory.[/bold red]")
                return False
        return True

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run_button" and not self._command_running:
            path_value = self.query_one("#path_input", Input).value.strip()
            api_key_value = self.query_one("#api_key_input", Input).value.strip()
            
            if not self.validate_path(path_value, self.selected_command):
                return

            effective_api_key = api_key_value
            if self.selected_command == "analyze" and not effective_api_key:
                effective_api_key = os.getenv("OPENAI_API_KEY")
                if effective_api_key:
                    logger.info("Using OPENAI_API_KEY from environment variable for 'analyze' command.")
                else:
                    logger.info("No API key provided or found in env for 'analyze'. AI features disabled.")
            
            self.query_one("#run_button", Button).disabled = True
            self._command_running = True
            logger.info(f"Executing command: [b]{self.selected_command}[/b] with path [i]'{path_value}'[/i]...")

            command_to_run: Callable = None
            args_for_command = []

            if self.selected_command == "summary":
                command_to_run = run_summary_command
                args_for_command = [path_value, logger]
            elif self.selected_command == "analyze":
                command_to_run = run_analyze_command
                args_for_command = [path_value, effective_api_key, logger, self]
            elif self.selected_command == "export":
                command_to_run = run_export_atomspace_command
                args_for_command = [path_value, logger]

            if command_to_run:
                self.run_worker(
                    self._execute_command_worker(command_to_run, args_for_command),
                    name=f"{self.selected_command}_worker",
                    group="command_workers",
                    exclusive=True
                )
            else:
                logger.error(f"Unknown command selected: {self.selected_command}")
                self.query_one("#run_button", Button).disabled = False
                self._command_running = False

    async def _execute_command_worker(self, command_func: Callable, args: list):
        try:
            if asyncio.iscoroutinefunction(command_func):
                # Pass the app instance if the command expects it (like run_analyze_command)
                if command_func.__name__ == 'run_analyze_command':
                    await command_func(*args, app=self) # Pass app instance
                else:
                    await command_func(*args) # For other potential async commands
            else:
                # Run synchronous functions in a separate thread via Textual's default worker behavior
                # or by explicitly using threading if needed for true parallelism for CPU-bound tasks.
                # For IO-bound, Textual's default worker handling of sync functions is usually fine.
                command_func(*args)
            
            # Capitalize first letter of command name for display
            cmd_name_display = command_func.__name__.replace('run_','').replace('_command','')
            cmd_name_display = cmd_name_display.capitalize()
            logger.info(f"Command [b]{cmd_name_display}[/b] finished.")

        except Exception as e:
            cmd_name_display_err = command_func.__name__.replace('run_','').replace('_command','').capitalize()
            logger.error(f"An unexpected error occurred in worker for {cmd_name_display_err}: {e}")
            logger.exception("Full traceback for worker error:")
        finally:
            self.app.call_from_thread(self._finalize_command_execution)

if __name__ == "__main__":
    app = ChimeraApp()
    app.run()