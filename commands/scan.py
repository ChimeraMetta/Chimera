"""
Scan Command module for the Hybrid Code Analyzer.
Provides functionality to scan directories for Python files.
"""

import os
from typing import List


class ScanCommand:
    """Command to scan directories for Python files."""
    
    def __init__(self, ui, code_manager):
        """Initialize the command."""
        self.ui = ui
        self.code_manager = code_manager
    
    def execute(self, directory=None):
        """Execute the scan command."""
        if not directory:
            directory = os.getcwd()
        
        self.ui.print_title("Scan for Python Files")
        self.ui.print(f"Scanning directory: {directory}")
        self.ui.print("")
        
        files = self.code_manager.scan_directory(directory)
        
        if files:
            self.ui.print_info("Python files found:")
            
            # Create a table to display files
            table = self.ui.create_table(["#", "File Path", "Size (KB)"])
            
            for i, file_path in enumerate(files[:20], 1):
                size_kb = round(os.path.getsize(file_path) / 1024, 2)
                self.ui.add_row(table, str(i), file_path, f"{size_kb} KB")
            
            if len(files) > 20:
                self.ui.add_row(table, "...", f"({len(files) - 20} more files)", "...")
            
            self.ui.print_table(table)
            
            return files
        else:
            self.ui.print_warning(f"No Python files found in {directory}")
            return []