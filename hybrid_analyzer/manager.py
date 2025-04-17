"""
Code Manager module for the Hybrid Code Analyzer.
Manages Python code analysis and modification operations.
"""

import os
import json
import time
from typing import List, Dict, Any, Set, Tuple, Optional

# Import file processing utilities
from file_processor import extract_functions, add_monitoring_to_file

# Check for analyzer modules
try:
    from hybrid_analyzer.static_analyzer import decompose_file, decompose_source
    from dynamic_monitor import hybrid_transform, monitor
    ANALYZER_MODULES_AVAILABLE = True
except ImportError:
    ANALYZER_MODULES_AVAILABLE = False


class CodeManager:
    """
    Manages Python code analysis and modification.
    Handles both static analysis and dynamic monitoring integration.
    """
    
    def __init__(self, ui):
        """Initialize the code manager."""
        self.ui = ui
        self.code_files = []
        self.analyzed_files = set()
        self.modified_files = set()
        self.last_analysis_results = []
        
        # Check if analyzer modules are available
        if not ANALYZER_MODULES_AVAILABLE:
            self.ui.print_warning("Analyzer modules not found. Some features may be unavailable.")
    
    def scan_directory(self, directory):
        """Recursively scan a directory for Python files."""
        python_files = []
        
        with self.ui.progress("Scanning directory", 1) as progress:
            task = progress.add_task("Finding Python files...", total=None)
            
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(".py"):
                        full_path = os.path.join(root, file)
                        python_files.append(full_path)
            
            progress.update(task, total=len(python_files), description=f"Found {len(python_files)} Python files")
        
        self.code_files = sorted(python_files)
        self.ui.print_success(f"Found {len(python_files)} Python files in {directory}")
        return python_files
    
    def analyze_file(self, file_path):
        """Run static analysis on a single file."""
        if not ANALYZER_MODULES_AVAILABLE:
            self.ui.print_error("Static analyzer module not available")
            return None
        
        self.ui.print_info(f"Analyzing {os.path.basename(file_path)}...")
        try:
            # Run static analysis
            result = decompose_file(file_path)
            
            if "error" in result and result["error"]:
                self.ui.print_error(f"Analysis error: {result['error']}")
                return None
            
            self.analyzed_files.add(file_path)
            self.ui.print_success(f"Generated {len(result['metta_atoms'])} MeTTa atoms")
            return result
        
        except Exception as e:
            self.ui.print_error(f"Failed to analyze {file_path}: {e}")
            return None
    
    def analyze_all_files(self):
        """Run static analysis on all discovered files."""
        if not self.code_files:
            self.ui.print_warning("No Python files to analyze. Run scan first.")
            return []
        
        results = []
        total_atoms = 0
        
        with self.ui.progress("Analyzing files", len(self.code_files)) as progress:
            task = progress.add_task("Analyzing...", total=len(self.code_files))
            
            for file_path in self.code_files:
                progress.update(task, description=f"Analyzing {os.path.basename(file_path)}")
                result = self.analyze_file(file_path)
                
                if result:
                    results.append((file_path, result))
                    total_atoms += len(result['metta_atoms'])
                
                progress.update(task, advance=1)
        
        self.ui.print_success(f"Successfully analyzed {len(results)} files")
        self.ui.print_info(f"Generated {total_atoms} total MeTTa atoms")
        
        # Store the results for later use
        self.last_analysis_results = results
        
        return results
    
    def add_dynamic_monitoring(self, file_path, functions_to_monitor, context=None, auto_fix=False):
        """
        Add dynamic monitoring to selected functions in a file.
        
        Args:
            file_path: Path to the Python file
            functions_to_monitor: List of function details to monitor
            context: Optional context string for monitoring
            auto_fix: Whether to enable auto-fixing
            
        Returns:
            Boolean indicating success
        """
        if not functions_to_monitor:
            self.ui.print_warning(f"No functions selected for monitoring in {file_path}")
            return False
        
        try:
            success = add_monitoring_to_file(file_path, functions_to_monitor, context, auto_fix)
            
            if success:
                self.modified_files.add(file_path)
                self.ui.print_success(f"Added monitoring to {len(functions_to_monitor)} functions in {file_path}")
            
            return success
            
        except Exception as e:
            self.ui.print_error(f"Failed to add monitoring to {file_path}: {e}")
            return False
    
    def export_analysis_results(self, results, output_file):
        """Export static analysis results to a file."""
        try:
            # Prepare data for export
            export_data = {
                "timestamp": time.time(),
                "files": [
                    {
                        "file_path": file_path,
                        "metta_atoms": result.get("metta_atoms", []),
                        "structure": result.get("structure", []),
                        "function_calls": result.get("function_calls", {}),
                        "module_relationships": result.get("module_relationships", {})
                    }
                    for file_path, result in results
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.ui.print_success(f"Analysis results exported to {output_file}")
            return True
            
        except Exception as e:
            self.ui.print_error(f"Failed to export analysis results: {e}")
            return False
    
    def get_function_recommendations(self):
        """Get MeTTa recommendations for functions."""
        if not ANALYZER_MODULES_AVAILABLE:
            self.ui.print_error("Analyzer modules not available")
            return []
        
        recommendations = []
        
        # For each file and its functions, query MeTTa for recommendations
        for file_path in self.code_files:
            file_functions = extract_functions(file_path)
            
            for func in file_functions:
                # Check if any recommendations exist
                func_recs = monitor.get_function_recommendations(func['full_name'])
                
                if func_recs:
                    recommendations.append({
                        'file_path': file_path,
                        'function': func,
                        'recommendations': func_recs
                    })
        
        return recommendations