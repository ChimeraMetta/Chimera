"""
Analyze Command module for the Hybrid Code Analyzer.
Provides functionality to run static analysis on Python files.
"""

import os
import ast
from typing import List, Dict, Any

# Check for analyzer modules
try:
    from static_analyzer import decompose_file
    ANALYZER_MODULES_AVAILABLE = True
except ImportError:
    ANALYZER_MODULES_AVAILABLE = False


class AnalyzeCommand:
    """Command to run static analysis on Python files."""
    
    def __init__(self, ui, code_manager):
        """Initialize the command."""
        self.ui = ui
        self.code_manager = code_manager
    
    def execute(self, output_file=None):
        """Execute the analyze command."""
        if not ANALYZER_MODULES_AVAILABLE:
            self.ui.print_error("Static analyzer module is not available")
            return False
        
        self.ui.print_title("Run Static Analysis")
        
        if not self.code_manager.code_files:
            self.ui.print_warning("No Python files to analyze. Run scan first.")
            return False
        
        self.ui.print_info(f"Will analyze {len(self.code_manager.code_files)} Python files")
        
        if self.ui.confirm("Start analysis?"):
            results = self.code_manager.analyze_all_files()
            
            if results:
                # Show summary
                self.ui.print_title("Analysis Summary")
                
                table = self.ui.create_table(["File", "MeTTa Atoms", "Functions", "Operations"])
                
                for file_path, result in results:
                    file_name = os.path.basename(file_path)
                    atoms_count = len(result.get('metta_atoms', []))
                    functions = len([a for a in result.get('structure', []) if a.get('type') == 'function_def'])
                    operations = len([a for a in result.get('structure', []) if a.get('type') == 'bin_op'])
                    
                    self.ui.add_row(table, file_name, str(atoms_count), str(functions), str(operations))
                
                self.ui.print_table(table)
                
                # Offer to export results
                if output_file or self.ui.confirm("Export analysis results to a file?"):
                    if not output_file:
                        output_file = self.ui.prompt("Enter output file path", "metta_analysis.json")
                    self.code_manager.export_analysis_results(results, output_file)
                
                return True
            else:
                self.ui.print_warning("No analysis results were generated")
                return False
        
        return False
    
    def analyze_single_file(self, file_path):
        """Analyze a single Python file and show results."""
        if not ANALYZER_MODULES_AVAILABLE:
            self.ui.print_error("Static analyzer module is not available")
            return None
        
        if not os.path.exists(file_path):
            self.ui.print_error(f"File not found: {file_path}")
            return None
        
        # Call decompose_file directly if available, otherwise fall back to code_manager
        if ANALYZER_MODULES_AVAILABLE:
            try:
                result = decompose_file(file_path)
                self.code_manager.analyzed_files.add(file_path)
            except Exception as e:
                self.ui.print_error(f"Analysis error: {e}")
                result = None
        else:
            result = self.code_manager.analyze_file(file_path)
        
        if result:
            self.ui.print_title(f"Analysis of {os.path.basename(file_path)}")
            
            # Display atom count
            self.ui.print_info(f"Generated {len(result['metta_atoms'])} MeTTa atoms")
            
            # Show structure summary
            if 'structure' in result:
                func_count = len([a for a in result['structure'] if a.get('type') == 'function_def'])
                class_count = len([a for a in result['structure'] if a.get('type') == 'class_def'])
                var_count = len([a for a in result['structure'] if a.get('type') in ['variable_assign', 'variable_ann_assign']])
                op_count = len([a for a in result['structure'] if a.get('type') == 'bin_op'])
                
                self.ui.print_info(f"Found {func_count} functions, {class_count} classes, {var_count} variables, {op_count} operations")
            
            # Show sample atoms (first 5)
            if result['metta_atoms']:
                self.ui.print_info("Sample MeTTa atoms:")
                for atom in result['metta_atoms'][:5]:
                    self.ui.print(f"  {atom}")
                if len(result['metta_atoms']) > 5:
                    self.ui.print(f"  ... and {len(result['metta_atoms']) - 5} more")
            
            return result
        
        return None