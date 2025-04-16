#!/usr/bin/env python3
"""
Hybrid Code Analyzer - A CLI tool for Python-MeTTa code analysis and monitoring.
Main entry point that initializes the application and runs the CLI.
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional

# Import our components
from console_ui import ConsoleUI
from code_manager import CodeManager
from commands.scan_command import ScanCommand
from commands.analyze_command import AnalyzeCommand
from commands.monitor_command import MonitorCommand
from commands.recommend_command import RecommendCommand

# Check for required modules
try:
    from static_analyzer import decompose_file
    from dynamic_monitor import hybrid_transform, monitor
    ANALYZER_MODULES_AVAILABLE = True
except ImportError:
    ANALYZER_MODULES_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hybrid Code Analyzer - Python-MeTTa code analysis and monitoring tool"
    )
    
    parser.add_argument(
        "command", 
        nargs="?",
        choices=["scan", "analyze", "monitor", "recommend", "export"], 
        help="Command to execute (optional, runs interactive mode if omitted)"
    )
    
    parser.add_argument(
        "--dir", "-d", 
        help="Target directory for analysis"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file for analysis results"
    )
    
    parser.add_argument(
        "--context", "-c",
        help="Context for dynamic monitoring"
    )
    
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Enable auto-fix in dynamic monitoring"
    )
    
    return parser.parse_args()


class HybridAnalyzerCLI:
    """
    Command-line interface for the hybrid code analyzer.
    Provides a user-friendly interface for code analysis and monitoring.
    """
    
    def __init__(self):
        """Initialize the CLI."""
        self.ui = ConsoleUI()
        self.code_manager = CodeManager(self.ui)
        self.current_dir = os.getcwd()
        
        # Initialize commands
        self.scan_command = ScanCommand(self.ui, self.code_manager)
        self.analyze_command = AnalyzeCommand(self.ui, self.code_manager)
        self.monitor_command = MonitorCommand(self.ui, self.code_manager)
        self.recommend_command = RecommendCommand(self.ui, self.code_manager)
        
        # Check for analyzer modules
        if not ANALYZER_MODULES_AVAILABLE:
            self.ui.print_error("Analyzer modules not found. Please install them first.")
            self.missing_modules = True
        else:
            self.missing_modules = False
    
    def show_welcome(self):
        """Show the welcome message."""
        self.ui.print_title("Python-MeTTa Hybrid Code Analyzer")
        self.ui.print("A tool for intelligent code analysis and monitoring")
        self.ui.print("")
        
        if self.missing_modules:
            self.ui.print_error("Required modules are missing. Some features will be unavailable.")
            self.ui.print_info("Please ensure static_analyzer.py and dynamic_monitor.py are in your PATH")
            self.ui.print("")
    
    def show_main_menu(self):
        """Show the main menu and handle user selection."""
        options = [
            "Select working directory",
            "Scan for Python files",
            "Run static analysis",
            "Add dynamic monitoring",
            "View recommendations",
            "Export results",
            "Exit"
        ]
        
        while True:
            self.ui.print_title("Main Menu")
            self.ui.print(f"Current directory: {self.current_dir}")
            self.ui.print(f"Files discovered: {len(self.code_manager.code_files)}")
            self.ui.print(f"Files analyzed: {len(self.code_manager.analyzed_files)}")
            self.ui.print(f"Files modified: {len(self.code_manager.modified_files)}")
            self.ui.print("")
            
            choice = self.ui.select_option("Select an option:", options)
            
            if choice == 0:  # Select working directory
                self.select_directory()
            elif choice == 1:  # Scan for Python files
                self.scan_command.execute(self.current_dir)
            elif choice == 2:  # Run static analysis
                self.analyze_command.execute()
            elif choice == 3:  # Add dynamic monitoring
                self.monitor_command.execute()
            elif choice == 4:  # View recommendations
                self.recommend_command.execute()
            elif choice == 5:  # Export results
                self.export_results()
            elif choice == 6:  # Exit
                self.ui.print_info("Exiting...")
                break
    
    def select_directory(self):
        """Let the user select a working directory."""
        self.ui.print_title("Select Working Directory")
        self.ui.print(f"Current directory: {self.current_dir}")
        self.ui.print("")
        
        new_dir = self.ui.prompt("Enter new directory path (or '.' for current)", ".")
        
        if new_dir == ".":
            new_dir = os.getcwd()
        
        if not os.path.isdir(new_dir):
            self.ui.print_error(f"'{new_dir}' is not a valid directory")
            return
        
        self.current_dir = os.path.abspath(new_dir)
        self.ui.print_success(f"Working directory changed to {self.current_dir}")
        
        # Clear previous scan results
        self.code_manager.code_files = []
        self.code_manager.analyzed_files = set()
        self.code_manager.modified_files = set()
    
    def export_results(self):
        """Export analysis results to a file."""
        self.ui.print_title("Export Results")
        
        if not hasattr(self.code_manager, 'last_analysis_results') or not self.code_manager.last_analysis_results:
            self.ui.print_warning("No analysis results to export. Run static analysis first.")
            return
        
        output_file = self.ui.prompt("Enter output file path", "metta_analysis.json")
        self.code_manager.export_analysis_results(self.code_manager.last_analysis_results, output_file)
    
    def run_command(self, command, args):
        """Run a specific command based on command line args."""
        if command == "scan":
            target_dir = args.dir or self.current_dir
            self.scan_command.execute(target_dir)
        
        elif command == "analyze":
            if args.dir:
                self.current_dir = os.path.abspath(args.dir)
                self.scan_command.execute(self.current_dir)
            self.analyze_command.execute(args.output)
        
        elif command == "monitor":
            if args.dir:
                self.current_dir = os.path.abspath(args.dir)
                self.scan_command.execute(self.current_dir)
            self.monitor_command.execute(args.context, args.auto_fix)
        
        elif command == "recommend":
            self.recommend_command.execute()
        
        elif command == "export":
            if not hasattr(self.code_manager, 'last_analysis_results') or not self.code_manager.last_analysis_results:
                self.ui.print_warning("No analysis results to export.")
                return
            
            output_file = args.output or "metta_analysis.json"
            self.code_manager.export_analysis_results(self.code_manager.last_analysis_results, output_file)


def main():
    """Main entry point."""
    args = parse_args()
    cli = HybridAnalyzerCLI()
    
    if args.command:
        # Run in command mode
        cli.run_command(args.command, args)
    else:
        # Run in interactive mode
        cli.show_welcome()
        cli.show_main_menu()


if __name__ == "__main__":
    main()