import time
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from reflectors.static_analyzer import decompose_function, decompose_file, decompose_source

def export_metta_atoms(atoms: List[str], output_path: str, 
                      source_name: str = "analysis", 
                      include_metadata: bool = True) -> bool:
    """
    Simple function to export MeTTa atoms directly to a .metta file.
    
    Args:
        atoms: List of MeTTa atom strings to export
        output_path: Path where to save the .metta file
        source_name: Name/description of the source (for metadata)
        include_metadata: Whether to include metadata comments
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if include_metadata:
                f.write(f"; MeTTa Atoms Export\n")
                f.write(f"; Source: {source_name}\n")
                f.write(f"; Exported: {time.ctime()}\n")
                f.write(f"; Total atoms: {len(atoms)}\n")
                f.write(f";\n")
            
            # Write each atom
            for atom in atoms:
                if atom and atom.strip() and not atom.strip().startswith(';'):
                    f.write(f"{atom.strip()}\n")
        
        return True
        
    except Exception as e:
        print(f"Error exporting atoms to {output_path}: {e}")
        return False


def export_function_atoms(func, output_path: str) -> bool:
    """
    Export a Python function's atoms directly to a .metta file.
    
    Args:
        func: Python function object
        output_path: Path to save the .metta file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        result = decompose_function(func)
        if "error" in result:
            print(f"Error decomposing function: {result['error']}")
            return False
        
        func_name = getattr(func, '__name__', 'unknown_function')
        atoms = result["metta_atoms"]
        
        return export_metta_atoms(atoms, output_path, f"function:{func_name}")
        
    except Exception as e:
        print(f"Error exporting function atoms: {e}")
        return False


def export_file_atoms(file_path: str, output_path: str) -> bool:
    """
    Export atoms from analyzing a Python file.
    
    Args:
        file_path: Path to Python file to analyze
        output_path: Path to save the .metta file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        result = decompose_file(file_path)
        if "error" in result:
            print(f"Error analyzing file {file_path}: {result['error']}")
            return False
        
        atoms = result["metta_atoms"]
        source_name = f"file:{os.path.basename(file_path)}"
        
        return export_metta_atoms(atoms, output_path, source_name)
        
    except Exception as e:
        print(f"Error exporting file atoms: {e}")
        return False


def export_source_atoms(source_code: str, output_path: str, source_name: str = "source") -> bool:
    """
    Export atoms from Python source code string.
    
    Args:
        source_code: Python source code as string
        output_path: Path to save the .metta file
        source_name: Name to identify this source
        
    Returns:
        True if successful, False otherwise
    """
    try:
        result = decompose_source(source_code)
        if "error" in result:
            print(f"Error analyzing source code: {result['error']}")
            return False
        
        atoms = result["metta_atoms"]
        
        return export_metta_atoms(atoms, output_path, source_name)
        
    except Exception as e:
        print(f"Error exporting source atoms: {e}")
        return False


def append_atoms_to_file(atoms: List[str], output_path: str, 
                        source_name: str = "additional") -> bool:
    """
    Append atoms to an existing .metta file (for accumulating from multiple sources).
    
    Args:
        atoms: List of MeTTa atom strings to append
        output_path: Path to the .metta file
        source_name: Name/description of the source
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'a') as f:
            f.write(f"\n; Additional atoms from: {source_name}\n")
            f.write(f"; Added: {time.ctime()}\n")
            f.write(f";\n")
            
            for atom in atoms:
                if atom and atom.strip() and not atom.strip().startswith(';'):
                    f.write(f"{atom.strip()}\n")
        
        return True
        
    except Exception as e:
        print(f"Error appending atoms to {output_path}: {e}")
        return False


# Integration functions for your CLI commands
def export_from_summary_analysis(target_path: str, export_path: str) -> bool:
    """
    Export atoms from summary analysis of a codebase.
    This can be called after your full_analyzer.analyze_codebase() runs.
    
    Args:
        target_path: Path that was analyzed
        export_path: Where to save the atoms
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.isfile(target_path):
            # Single file analysis
            return export_file_atoms(target_path, export_path)
        elif os.path.isdir(target_path):
            # Directory analysis - export all Python files
            python_files = []
            for root, dirs, files in os.walk(target_path):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
            
            if not python_files:
                print(f"No Python files found in {target_path}")
                return False
            
            # Export first file to create the base
            first_file = python_files[0]
            success = export_file_atoms(first_file, export_path)
            if not success:
                return False
            
            # Append remaining files
            for file_path in python_files[1:]:
                result = decompose_file(file_path)
                if "error" not in result:
                    atoms = result["metta_atoms"]
                    source_name = f"file:{os.path.relpath(file_path, target_path)}"
                    append_atoms_to_file(atoms, export_path, source_name)
            
            return True
        else:
            print(f"Invalid target path: {target_path}")
            return False
            
    except Exception as e:
        print(f"Error in summary analysis export: {e}")
        return False


def export_from_complexity_analysis(target_path: str, export_path: str, 
                                   additional_data: Optional[Dict] = None) -> bool:
    """
    Export atoms from complexity analysis.
    This can be called after your complexity analysis runs.
    
    Args:
        target_path: Path that was analyzed
        export_path: Where to save the atoms
        additional_data: Any additional analysis data to include
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Base export from file/directory analysis
        success = export_from_summary_analysis(target_path, export_path)
        
        # If we have additional analysis data, append it as atoms
        if success and additional_data:
            additional_atoms = []
            
            # Convert additional analysis data to MeTTa atoms
            # You can customize this based on what your complexity analyzer produces
            if "complexity_metrics" in additional_data:
                for func_name, metrics in additional_data["complexity_metrics"].items():
                    for metric_name, value in metrics.items():
                        atom = f"(complexity-metric {func_name} {metric_name} {value})"
                        additional_atoms.append(atom)
            
            if "optimization_suggestions" in additional_data:
                for suggestion in additional_data["optimization_suggestions"]:
                    atom = f"(optimization-suggestion {suggestion.get('function', 'unknown')} \"{suggestion.get('description', '')}\")"
                    additional_atoms.append(atom)
            
            if additional_atoms:
                append_atoms_to_file(additional_atoms, export_path, "complexity_analysis")
        
        return success
        
    except Exception as e:
        print(f"Error in complexity analysis export: {e}")
        return False


# Simple function to check if atoms were exported
def import_metta_file(file_path: str, output_path: str, overwrite_conflicts: bool = True) -> Dict:
    """
    Import atoms from one .metta file into another (merge/combine files).
    
    Args:
        file_path: Path to .metta file to import from
        output_path: Path to .metta file to import into (will be created if doesn't exist)
        overwrite_conflicts: Whether to overwrite conflicting atoms
        
    Returns:
        Dictionary with import results
    """
    try:
        if not os.path.exists(file_path):
            return {"success": False, "error": f"Source file not found: {file_path}"}
        
        # Read source atoms
        source_atoms = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(';'):
                    source_atoms.append(line)
        
        if not source_atoms:
            return {"success": True, "imported": 0, "message": "No atoms to import"}
        
        # If output file doesn't exist, just copy all atoms
        if not os.path.exists(output_path):
            success = export_metta_atoms(source_atoms, output_path, f"imported_from_{os.path.basename(file_path)}")
            return {
                "success": success,
                "imported": len(source_atoms) if success else 0,
                "overwritten": 0,
                "skipped": 0,
                "message": f"Imported {len(source_atoms)} atoms to new file"
            }
        
        # Read existing atoms for conflict detection
        existing_atoms = []
        existing_signatures = set()
        
        with open(output_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(';'):
                    existing_atoms.append(line)
                    signature = _get_atom_signature(line)
                    existing_signatures.add(signature)
        
        # Check for conflicts
        new_atoms = []
        conflicts = []
        
        for atom in source_atoms:
            signature = _get_atom_signature(atom)
            if signature in existing_signatures:
                conflicts.append(atom)
            else:
                new_atoms.append(atom)
        
        # Import new atoms
        imported_count = 0
        if new_atoms:
            success = append_atoms_to_file(new_atoms, output_path, f"imported_from_{os.path.basename(file_path)}")
            imported_count = len(new_atoms) if success else 0
        
        # Handle conflicts
        overwritten_count = 0
        skipped_count = 0
        
        if conflicts:
            if overwrite_conflicts:
                # For simplicity, append conflicting atoms with a note
                # In a real implementation, you'd want to actually replace them
                conflict_atoms = [f"; OVERWRITE: {atom}" for atom in conflicts]
                append_atoms_to_file(conflicts, output_path, f"overwritten_from_{os.path.basename(file_path)}")
                overwritten_count = len(conflicts)
            else:
                skipped_count = len(conflicts)
        
        return {
            "success": True,
            "imported": imported_count,
            "overwritten": overwritten_count,
            "skipped": skipped_count,
            "message": f"Imported {imported_count} new atoms, overwritten {overwritten_count}, skipped {skipped_count}"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_atom_signature(atom: str) -> str:
    """Get a signature for an atom to detect conflicts (simple version)."""
    atom = atom.strip()
    if not atom.startswith('('):
        return atom
    
    try:
        # Remove outer parentheses and get first two parts
        inner = atom[1:-1].strip()
        parts = inner.split(' ', 2)
        
        if len(parts) < 2:
            return atom
        
        first_part = parts[0]
        second_part = parts[1]
        
        # Create simple signatures
        if first_part == ':':
            return f"type_def:{second_part}"
        elif first_part == '=':
            if second_part.startswith('('):
                func_expr = second_part[1:].split(' ')[0].split(')')[0]
                return f"function_impl:{func_expr}"
            else:
                return f"assignment:{second_part}"
        else:
            return f"{first_part}:{second_part}"
            
    except Exception:
        return atom


def combine_metta_files(file_paths: List[str], output_path: str, 
                       source_name: str = "combined") -> bool:
    """
    Combine multiple .metta files into one.
    
    Args:
        file_paths: List of .metta files to combine
        output_path: Path for the combined output file
        source_name: Name for the combined source
        
    Returns:
        True if successful, False otherwise
    """
    try:
        all_atoms = []
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    file_atoms = []
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith(';'):
                            file_atoms.append(line)
                    
                    if file_atoms:
                        all_atoms.append(f"; From: {os.path.basename(file_path)}")
                        all_atoms.extend(file_atoms)
                        all_atoms.append("")  # Empty line for separation
        
        if not all_atoms:
            print("No atoms found to combine")
            return False
        
        # Remove empty lines at the end
        clean_atoms = [atom for atom in all_atoms if atom != ""]
        
        return export_metta_atoms(clean_atoms, output_path, source_name)
        
    except Exception as e:
        print(f"Error combining files: {e}")
        return False


def verify_export(export_path: str) -> Dict:
    """
    Verify that atoms were exported correctly.
    
    Args:
        export_path: Path to the exported .metta file
        
    Returns:
        Dictionary with verification results
    """
    try:
        if not os.path.exists(export_path):
            return {"success": False, "error": "Export file does not exist"}
        
        with open(export_path, 'r') as f:
            lines = f.readlines()
        
        atom_count = 0
        comment_count = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith(';'):
                comment_count += 1
            elif line and line.startswith('(') and line.endswith(')'):
                atom_count += 1
        
        return {
            "success": True,
            "total_lines": len(lines),
            "atom_count": atom_count,
            "comment_count": comment_count,
            "file_size": os.path.getsize(export_path)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def export_from_metta_generation(generation_results: Dict[str, Any], output_file: str) -> bool:
    """
    Export MeTTa donor generation results to a .metta file with generator attribution.
    
    Args:
        generation_results: Dictionary containing generation results for each function
        output_file: Path to the output .metta file
        
    Returns:
        bool: True if export successful, False otherwise
    """
    try:
        import time
        
        with open(output_file, 'w') as f:
            # Write header
            f.write(f"; MeTTa Donor Generation Export\n")
            f.write(f"; Generated: {time.ctime()}\n")
            f.write(f"; Functions processed: {len(generation_results)}\n")
            f.write(f";\n\n")
            
            atom_count = 0
            
            # Export generation metadata with generator information
            f.write(f"; Generation Metadata\n")
            for func_name, result in generation_results.items():
                f.write(f"(metta-generation-function {func_name})\n")
                atom_count += 1
                
                if result["generation_success"]:
                    f.write(f"(generation-success {func_name} {len(result['candidates'])})\n")
                    atom_count += 1
                    
                    # Export generator usage for this function
                    generators_used = set(result.get("generators_used", []))
                    for generator in generators_used:
                        f.write(f"(function-generator-used {func_name} {generator})\n")
                        atom_count += 1
                else:
                    f.write(f"(generation-failure {func_name})\n")
                    atom_count += 1
            
            f.write(f"\n; Generated Candidates with Generator Attribution\n")
            
            # Export candidate information with generator details
            for func_name, result in generation_results.items():
                if result["generation_success"] and result["candidates"]:
                    for i, candidate in enumerate(result["candidates"]):
                        candidate_id = f"{func_name}-candidate-{i+1}"
                        generator_used = candidate.get('generator_used', candidate['strategy'])
                        
                        # Basic candidate info
                        f.write(f"(donor-candidate {candidate_id} {candidate['name']})\n")
                        f.write(f"(candidate-generator {candidate_id} {generator_used})\n")
                        f.write(f"(candidate-strategy {candidate_id} {candidate['strategy']})\n")
                        f.write(f"(candidate-score {candidate_id} {candidate['final_score']})\n")
                        f.write(f"(candidate-confidence {candidate_id} {candidate['confidence']})\n")
                        f.write(f"(candidate-pattern-family {candidate_id} {candidate['pattern_family']})\n")
                        f.write(f"(candidate-complexity {candidate_id} {candidate['complexity_estimate']})\n")
                        f.write(f"(candidate-scope {candidate_id} {candidate['applicability_scope']})\n")
                        atom_count += 8
                        
                        # Properties
                        for prop in candidate['properties']:
                            f.write(f"(candidate-property {candidate_id} {prop})\n")
                            atom_count += 1
                        
                        # Data structures used
                        for ds in candidate['data_structures_used']:
                            f.write(f"(candidate-data-structure {candidate_id} {ds})\n")
                            atom_count += 1
                        
                        # Operations used
                        for op in candidate['operations_used']:
                            f.write(f"(candidate-operation {candidate_id} {op})\n")
                            atom_count += 1
                        
                        # MeTTa derivation
                        for derivation in candidate['metta_derivation']:
                            # Clean derivation string for MeTTa format
                            clean_derivation = derivation.replace('"', '\\"')
                            f.write(f"(candidate-derivation {candidate_id} \"{clean_derivation}\")\n")
                            atom_count += 1
                        
                        # Description (escaped)
                        escaped_desc = candidate['description'].replace('"', '\\"')
                        f.write(f"(candidate-description {candidate_id} \"{escaped_desc}\")\n")
                        atom_count += 1
                        
                        f.write(f"\n")
            
            # Export generator statistics
            f.write(f"; Generator Statistics\n")
            generator_stats = {}
            strategy_stats = {}
            total_candidates = 0
            
            for func_name, result in generation_results.items():
                if result["generation_success"]:
                    for candidate in result["candidates"]:
                        total_candidates += 1
                        generator = candidate.get('generator_used', candidate['strategy'])
                        strategy = candidate['strategy']
                        
                        generator_stats[generator] = generator_stats.get(generator, 0) + 1
                        strategy_stats[strategy] = strategy_stats.get(strategy, 0) + 1
            
            # Export generator usage statistics
            for generator, count in generator_stats.items():
                f.write(f"(generator-usage {generator} {count})\n")
                atom_count += 1
            
            # Export strategy usage statistics
            for strategy, count in strategy_stats.items():
                f.write(f"(strategy-usage {strategy} {count})\n")
                atom_count += 1
            
            # Export summary statistics
            f.write(f"\n; Summary Statistics\n")
            successful_count = sum(1 for r in generation_results.values() if r["generation_success"])
            
            f.write(f"(generation-summary total-functions {len(generation_results)})\n")
            f.write(f"(generation-summary successful-functions {successful_count})\n")
            f.write(f"(generation-summary total-candidates {total_candidates})\n")
            f.write(f"(generation-summary total-generators {len(generator_stats)})\n")
            f.write(f"(generation-summary total-strategies {len(strategy_stats)})\n")
            atom_count += 5
            
            f.write(f"\n; Export completed with {atom_count} atoms\n")
        
        return True
        
    except Exception as e:
        print(f"Error exporting MeTTa generation results: {e}")
        import traceback
        traceback.print_exc()
        return False

# Example usage for testing
if __name__ == "__main__":
    # Test function
    def sample_function(x: int, y: str = "test") -> str:
        """Sample function for testing."""
        return f"{x}: {y}"
    
    # Test exports
    print("Testing simple MeTTa export...")
    
    # Test function export
    success = export_function_atoms(sample_function, "./test_exports/function_test.metta")
    print(f"Function export: {success}")
    
    if success:
        verification = verify_export("./test_exports/function_test.metta")
        print(f"Verification: {verification}")
    
    # Test source code export
    source_code = '''
def utility_function(data: list) -> int:
    """Utility function."""
    return len([x for x in data if x > 0])

class DataProcessor:
    def process(self, items: list) -> dict:
        return {"count": len(items)}
'''
    
    success = export_source_atoms(source_code, "./test_exports/source_test.metta", "utility_code")
    print(f"Source export: {success}")
    
    if success:
        verification = verify_export("./test_exports/source_test.metta")
        print(f"Source verification: {verification}")
    
    print("Testing complete!")
