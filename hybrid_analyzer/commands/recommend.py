"""
Recommend Command module for the Hybrid Code Analyzer.
Provides functionality to view and apply MeTTa recommendations.
"""

import os
from typing import List, Dict, Any

# Check for analyzer modules
try:
    from dynamic_monitor import monitor
    MONITOR_MODULE_AVAILABLE = True
except ImportError:
    MONITOR_MODULE_AVAILABLE = False


class RecommendCommand:
    """Command to view and apply MeTTa recommendations."""
    
    def __init__(self, ui, code_manager):
        """Initialize the command."""
        self.ui = ui
        self.code_manager = code_manager
    
    def execute(self):
        """Execute the recommend command."""
        if not MONITOR_MODULE_AVAILABLE:
            self.ui.print_error("Dynamic monitor module is not available")
            return False
        
        self.ui.print_title("View Recommendations")
        
        # Check if we have runtime data
        if not hasattr(monitor, 'function_metrics') or not monitor.function_metrics:
            self.ui.print_warning("No runtime data available yet")
            self.ui.print_info("Run your code with the dynamic monitoring to collect data")
            return False
        
        # Show metrics for monitored functions
        self._show_metrics()
        
        # Show recommendations
        return self._show_recommendations()
    
    def _show_metrics(self):
        """Show metrics for monitored functions."""
        self.ui.print_title("Function Metrics")
        
        if not monitor.function_metrics:
            self.ui.print_warning("No function metrics available")
            return
        
        table = self.ui.create_table(["Function", "Calls", "Success %", "Avg Time (ms)", "Max Time (ms)"])
        
        for func_name, metrics in monitor.function_metrics.items():
            success_pct = (metrics['successes'] / max(1, metrics['calls'])) * 100
            avg_time_ms = metrics['avg_time'] * 1000  # Convert to ms
            max_time_ms = metrics['max_time'] * 1000  # Convert to ms
            
            self.ui.add_row(
                table, 
                func_name, 
                str(metrics['calls']), 
                f"{success_pct:.1f}%", 
                f"{avg_time_ms:.2f}",
                f"{max_time_ms:.2f}"
            )
        
        self.ui.print_table(table)
    
    def _show_recommendations(self):
        """Show recommendations for functions."""
        recs_available = False
        recommendations = []
        
        for func_name in monitor.function_metrics:
            func_recs = monitor.get_function_recommendations(func_name)
            
            if func_recs:
                recs_available = True
                self.ui.print_title(f"Recommendations for {func_name}")
                
                for i, rec in enumerate(func_recs, 1):
                    self.ui.print(f"{i}. {rec['recommendation']}")
                    if 'sample_code' in rec:
                        self.ui.print_code(rec['sample_code'])
                    
                    recommendations.append({
                        'function_name': func_name,
                        'recommendation': rec
                    })
        
        if not recs_available:
            self.ui.print_info("No recommendations available")
            self.ui.print("Run your code more to generate recommendations")
            return False
        
        # Ask if the user wants to apply any recommendations
        if self.ui.confirm("Would you like to apply any of these recommendations?"):
            return self._apply_recommendations(recommendations)
        
        return True
    
    def _apply_recommendations(self, recommendations):
        """Apply selected recommendations."""
        if not recommendations:
            self.ui.print_warning("No recommendations to apply")
            return False
        
        # Create a list of recommendations
        rec_options = []
        for i, rec in enumerate(recommendations):
            rec_options.append(
                f"{rec['function_name']}: {rec['recommendation']['recommendation']}"
            )
        
        rec_options.append("Cancel")
        
        choice = self.ui.select_option("Select a recommendation to apply:", rec_options)
        
        if choice == len(rec_options) - 1:  # Cancel
            return False
        
        selected_rec = recommendations[choice]
        
        # TODO: Implement actual code modification based on recommendations
        self.ui.print_info(f"Applying recommendation to {selected_rec['function_name']}...")
        self.ui.print_warning("Automatic application of recommendations is not implemented yet")
        self.ui.print_info("Please apply the following code manually:")
        self.ui.print_code(selected_rec['recommendation']['sample_code'])
        
        return True
    
    def _find_function_in_files(self, function_name):
        """Find the file containing a function."""
        for file_path in self.code_manager.code_files:
            try:
                from file_processor import extract_functions
                functions = extract_functions(file_path)
                
                for func in functions:
                    if func['full_name'] == function_name:
                        return file_path, func
            except:
                pass
        
        return None, None