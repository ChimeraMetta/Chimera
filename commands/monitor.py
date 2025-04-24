"""
Monitor Command module for the Hybrid Code Analyzer.
Provides functionality to add dynamic monitoring to Python functions.
"""

import os
from typing import List, Dict, Any, Optional
from commands.file_processor import extract_functions, check_monitoring_status


class MonitorCommand:
    """Command to add dynamic monitoring to Python functions."""
    
    def __init__(self, ui, code_manager):
        """Initialize the command."""
        self.ui = ui
        self.code_manager = code_manager
    
    def execute(self, context=None, auto_fix=False):
        """Execute the monitor command."""
        self.ui.print_title("Add Dynamic Monitoring")
        
        if not self.code_manager.code_files:
            self.ui.print_warning("No Python files to modify. Run scan first.")
            return False
        
        # Get monitoring options if not provided
        if context is None:
            context = self.ui.prompt("Enter context (e.g., 'data_processing', leave empty for none)", "")
            if context == "":
                context = None
                
        if auto_fix is None:
            auto_fix = self.ui.confirm("Enable auto-fix?", False)
        
        # Let the user select files to modify
        selected_files = self._select_files()
        if not selected_files:
            return False
        
        modified_count = 0
        
        # Process each selected file
        for file_path in selected_files:
            file_modified = self._process_file(file_path, context, auto_fix)
            if file_modified:
                modified_count += 1
        
        if modified_count > 0:
            self.ui.print_success(f"Added monitoring to {modified_count} file(s)")
            return True
        else:
            self.ui.print_warning("No files were modified")
            return False
    
    def _select_files(self):
        """Let the user select files to modify."""
        self.ui.print_info("Select files to add monitoring:")
        
        # Create a list of files with their index
        file_options = []
        for i, file_path in enumerate(self.code_manager.code_files, 1):
            file_name = os.path.basename(file_path)
            file_options.append(f"{file_name} ({file_path})")
        
        file_options.append("All files")
        file_options.append("Back")
        
        choice = self.ui.select_option("Select a file:", file_options)
        
        if choice == len(file_options) - 1:  # Back option
            return []
        
        if choice == len(file_options) - 2:  # All files option
            return self.code_manager.code_files
        
        # Single file selected
        return [self.code_manager.code_files[choice]]
    
    def _process_file(self, file_path, context, auto_fix):
        """Process a single file for monitoring."""
        self.ui.print_info(f"Processing {os.path.basename(file_path)}...")
        
        # Get functions in the file
        functions = extract_functions(file_path)
        
        if not functions:
            self.ui.print_warning(f"No functions found in {file_path}")
            return False
        
        # Get current monitoring status
        status = check_monitoring_status(file_path)
        
        # Mark functions that already have monitoring
        for i, func in enumerate(functions):
            func['has_monitoring'] = any(
                f['name'] == func['name'] and f.get('has_monitoring', False) 
                for f in status.get('functions', [])
            )
        
        # Let the user select functions
        selected_functions = self._select_functions(functions, file_path)
        if not selected_functions:
            return False
        
        # Add monitoring to selected functions
        success = self.code_manager.add_dynamic_monitoring(file_path, selected_functions, context, auto_fix)
        
        return success
    
    def _select_functions(self, functions, file_path):
        """Let the user select functions to monitor."""
        self.ui.print_info(f"Select functions to monitor in {os.path.basename(file_path)}:")
        
        function_options = []
        for func in functions:
            prefix = "Method" if func.get('is_method', False) else "Function"
            status = "[Monitored]" if func.get('has_monitoring', False) else ""
            function_options.append(
                f"{prefix}: {func['full_name']} (lines {func['line_start']}-{func['line_end']}) {status}"
            )
        
        function_options.append("All functions")
        function_options.append("Skip this file")
        
        func_choice = self.ui.select_option("Select a function:", function_options)
        
        if func_choice == len(function_options) - 1:  # Skip this file
            return []
        
        if func_choice == len(function_options) - 2:  # All functions
            # Filter out functions that already have monitoring
            return [f for f in functions if not f.get('has_monitoring', False)]
        
        # Single function selected
        selected_func = functions[func_choice]
        if selected_func.get('has_monitoring', False):
            self.ui.print_warning(f"Function {selected_func['full_name']} already has monitoring")
            if self.ui.confirm("Do you want to replace the existing monitoring?"):
                return [selected_func]
            return []
        
        return [selected_func]