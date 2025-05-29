import time
import os
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

# Import your existing components
from reflectors.static_analyzer import decompose_function
from reflectors.dynamic_monitor import DynamicMonitor


class AtomspaceManager:
    """
    Simple export/import manager for MeTTa atomspaces using only .metta files.
    Handles conflict detection and resolution during imports.
    """
    
    def __init__(self, monitor: Optional[DynamicMonitor] = None):
        self.monitor = monitor or DynamicMonitor()
    
    def export_atomspace(self, output_path: str, include_metadata: bool = True) -> bool:
        """
        Export the current atomspace to a .metta file.
        
        Args:
            output_path: Path to save the .metta file
            include_metadata: Whether to include metadata comments
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Query all atoms from the atomspace
            all_atoms = self._get_all_atoms()
            
            if not all_atoms:
                print("Warning: No atoms found in atomspace")
                return False
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                if include_metadata:
                    f.write(f"; MeTTa Atomspace Export\n")
                    f.write(f"; Exported: {time.ctime()}\n")
                    f.write(f"; Total atoms: {len(all_atoms)}\n")
                    f.write(f";\n")
                
                # Write each atom
                for atom in all_atoms:
                    atom_str = str(atom).strip()
                    if atom_str and not atom_str.startswith(';'):
                        f.write(f"{atom_str}\n")
            
            print(f"Successfully exported {len(all_atoms)} atoms to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting atomspace: {e}")
            return False
    
    def export_function(self, func, output_path: str, include_metadata: bool = True) -> bool:
        """
        Export a single Python function as MeTTa atoms.
        
        Args:
            func: Python function to export
            output_path: Path to save the .metta file
            include_metadata: Whether to include metadata comments
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Decompose the function
            result = decompose_function(func)
            if "error" in result:
                print(f"Error decomposing function: {result['error']}")
                return False
            
            atoms = result["metta_atoms"]
            func_name = getattr(func, '__name__', 'unknown_function')
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                if include_metadata:
                    f.write(f"; MeTTa Function Export: {func_name}\n")
                    f.write(f"; Exported: {time.ctime()}\n")
                    f.write(f"; Total atoms: {len(atoms)}\n")
                    f.write(f";\n")
                
                # Write each atom
                for atom in atoms:
                    if atom.strip():
                        f.write(f"{atom}\n")
            
            print(f"Successfully exported function '{func_name}' with {len(atoms)} atoms to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting function: {e}")
            return False
    
    def import_atomspace(self, file_path: str, overwrite: bool = False, dry_run: bool = False) -> Dict:
        """
        Import atoms from a .metta file with conflict detection.
        
        Args:
            file_path: Path to the .metta file to import
            overwrite: If True, overwrite existing definitions. If False, skip conflicts.
            dry_run: If True, only check for conflicts without actually importing
            
        Returns:
            Dictionary with import results and conflict information
        """
        try:
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File not found: {file_path}"}
            
            # Parse the file to get atoms to import
            incoming_atoms = self._parse_metta_file(file_path)
            if not incoming_atoms:
                return {"success": False, "error": "No valid atoms found in file"}
            
            # Get current atomspace state
            existing_atoms = self._get_all_atoms()
            existing_signatures = self._extract_atom_signatures(existing_atoms)
            
            # Check for conflicts
            conflicts = []
            new_atoms = []
            incoming_signatures = self._extract_atom_signatures(incoming_atoms)
            
            for atom, signature in zip(incoming_atoms, incoming_signatures):
                if signature in existing_signatures:
                    conflicts.append({
                        "signature": signature,
                        "atom": atom,
                        "action": "overwrite" if overwrite else "skip"
                    })
                else:
                    new_atoms.append(atom)
            
            # Prepare results
            result = {
                "success": True,
                "file_path": file_path,
                "total_incoming": len(incoming_atoms),
                "new_atoms": len(new_atoms),
                "conflicts": len(conflicts),
                "conflict_details": conflicts,
                "dry_run": dry_run
            }
            
            if dry_run:
                result["message"] = f"Dry run: Would import {len(new_atoms)} new atoms"
                if conflicts:
                    result["message"] += f", {len(conflicts)} conflicts detected"
                return result
            
            # Perform the actual import
            imported_count = 0
            skipped_count = 0
            overwritten_count = 0
            
            # Import new atoms (no conflicts)
            for atom in new_atoms:
                if self._add_atom_to_space(atom):
                    imported_count += 1
            
            # Handle conflicts
            for conflict in conflicts:
                if overwrite:
                    # Remove existing atom and add new one
                    if self._replace_atom_in_space(conflict["signature"], conflict["atom"]):
                        overwritten_count += 1
                else:
                    skipped_count += 1
            
            result.update({
                "imported": imported_count,
                "overwritten": overwritten_count,
                "skipped": skipped_count,
                "message": f"Imported {imported_count} new atoms"
            })
            
            if overwritten_count > 0:
                result["message"] += f", overwritten {overwritten_count} existing atoms"
            if skipped_count > 0:
                result["message"] += f", skipped {skipped_count} conflicting atoms"
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def check_conflicts(self, file_path: str) -> Dict:
        """
        Check for conflicts without importing (convenience method for dry run).
        """
        return self.import_atomspace(file_path, dry_run=True)
    
    def _get_all_atoms(self) -> List[str]:
        """Get all atoms from the current atomspace."""
        try:
            # Query all atoms
            query = "(match &self $atom $atom)"
            results = self.monitor.query(query)
            
            # Convert to strings and filter out empty results
            atoms = []
            for result in results:
                atom_str = str(result).strip()
                if atom_str and atom_str != "[]":
                    atoms.append(atom_str)
            
            return atoms
            
        except Exception as e:
            print(f"Error getting atoms from atomspace: {e}")
            return []
    
    def _parse_metta_file(self, file_path: str) -> List[str]:
        """Parse a .metta file and extract atoms."""
        try:
            atoms = []
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith(';'):
                        atoms.append(line)
            return atoms
            
        except Exception as e:
            print(f"Error parsing .metta file: {e}")
            return []
    
    def _extract_atom_signatures(self, atoms: List[str]) -> List[str]:
        """
        Extract signatures from atoms to identify duplicates.
        A signature is the atom without specific values that might change.
        """
        signatures = []
        for atom in atoms:
            signature = self._get_atom_signature(atom)
            signatures.append(signature)
        return signatures
    
    def _get_atom_signature(self, atom: str) -> str:
        """
        Get a signature for an atom to detect conflicts.
        
        For example:
        - (: myFunc (-> Number String)) -> "type_def:myFunc"
        - (function-def myFunc global 1 10) -> "function_def:myFunc"
        - (= (myFunc $x) (+ $x 1)) -> "function_impl:myFunc"
        """
        atom = atom.strip()
        if not atom.startswith('('):
            return atom
        
        try:
            # Remove outer parentheses and split
            inner = atom[1:-1].strip()
            parts = inner.split(' ', 2)
            
            if len(parts) < 2:
                return atom
            
            first_part = parts[0]
            second_part = parts[1]
            
            # Handle different atom types
            if first_part == ':':
                # Type definition: (: funcName type)
                return f"type_def:{second_part}"
            elif first_part == '=':
                # Function implementation: (= (funcName ...) ...)
                if second_part.startswith('('):
                    # Extract function name from (funcName ...)
                    func_expr = second_part[1:].split(' ')[0].split(')')[0]
                    return f"function_impl:{func_expr}"
                else:
                    return f"assignment:{second_part}"
            elif first_part == 'function-def':
                # Function definition metadata
                return f"function_def:{second_part}"
            elif first_part == 'class-def':
                # Class definition metadata
                return f"class_def:{second_part}"
            elif first_part == 'variable-assign':
                # Variable assignment
                return f"variable:{second_part}"
            else:
                # Other atoms - use first two parts as signature
                return f"{first_part}:{second_part}"
                
        except Exception:
            # If parsing fails, use the whole atom as signature
            return atom
    
    def _add_atom_to_space(self, atom: str) -> bool:
        """Add a single atom to the atomspace."""
        try:
            return self.monitor.add_atom(atom)
        except Exception as e:
            print(f"Error adding atom: {atom[:50]}... - {e}")
            return False
    
    def _replace_atom_in_space(self, signature: str, new_atom: str) -> bool:
        """
        Replace an existing atom with a new one.
        This is a simplified approach - in practice, MeTTa doesn't have direct
        atom removal, so we just add the new atom (it will overshadow the old one).
        """
        try:
            # For now, just add the new atom
            # MeTTa will use the most recent definition
            return self.monitor.add_atom(new_atom)
        except Exception as e:
            print(f"Error replacing atom with signature {signature}: {e}")
            return False
    
    def get_atomspace_stats(self) -> Dict:
        """Get statistics about the current atomspace."""
        try:
            all_atoms = self._get_all_atoms()
            signatures = self._extract_atom_signatures(all_atoms)
            
            # Count by atom type
            type_counts = {}
            for signature in signatures:
                atom_type = signature.split(':')[0]
                type_counts[atom_type] = type_counts.get(atom_type, 0) + 1
            
            return {
                "total_atoms": len(all_atoms),
                "unique_signatures": len(set(signatures)),
                "type_distribution": type_counts,
                "timestamp": time.ctime()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def list_functions(self) -> List[str]:
        """List all functions in the atomspace."""
        try:
            query = "(match &self (function-def $name $scope $start $end) $name)"
            results = self.monitor.query(query)
            return [str(result) for result in results if result]
        except Exception as e:
            print(f"Error listing functions: {e}")
            return []
    
    def backup_atomspace(self, backup_path: str) -> bool:
        """Create a backup of the current atomspace."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_file = f"{backup_path}_backup_{timestamp}.metta"
        return self.export_atomspace(backup_file, include_metadata=True)


# Convenience functions
def export_atomspace(output_path: str, monitor: Optional[DynamicMonitor] = None) -> bool:
    """Export current atomspace to a .metta file."""
    manager = AtomspaceManager(monitor)
    return manager.export_atomspace(output_path)


def export_function(func, output_path: str, monitor: Optional[DynamicMonitor] = None) -> bool:
    """Export a Python function as MeTTa atoms."""
    manager = AtomspaceManager(monitor)
    return manager.export_function(func, output_path)


def import_atomspace(file_path: str, monitor: Optional[DynamicMonitor] = None, 
                    overwrite: bool = False) -> Dict:
    """Import atoms from a .metta file."""
    manager = AtomspaceManager(monitor)
    return manager.import_atomspace(file_path, overwrite=overwrite)


def check_import_conflicts(file_path: str, monitor: Optional[DynamicMonitor] = None) -> Dict:
    """Check what conflicts would occur when importing a file."""
    manager = AtomspaceManager(monitor)
    return manager.check_conflicts(file_path)


# Example usage and testing
if __name__ == "__main__":
    # Create a test function
    def test_function(x: int, y: str = "default") -> str:
        """A test function."""
        return f"{x}_{y}"
    
    def another_function(a: float) -> float:
        """Another test function."""
        return a * 2.0
    
    # Test export
    print("=== Testing Export ===")
    manager = AtomspaceManager()
    
    # Export individual functions
    export_function(test_function, "./exports/test_function.metta")
    export_function(another_function, "./exports/another_function.metta")
    
    # Export current atomspace
    manager.export_atomspace("./exports/current_atomspace.metta")
    
    # Test import with conflict detection
    print("\n=== Testing Import ===")
    
    # First import
    result1 = manager.import_atomspace("./exports/test_function.metta")
    print(f"First import: {result1['message']}")
    
    # Second import (should detect conflicts)
    result2 = manager.import_atomspace("./exports/test_function.metta", overwrite=False)
    print(f"Second import (no overwrite): {result2['message']}")
    print(f"Conflicts detected: {result2['conflicts']}")
    
    # Third import with overwrite
    result3 = manager.import_atomspace("./exports/test_function.metta", overwrite=True)
    print(f"Third import (with overwrite): {result3['message']}")
    
    # Check conflicts without importing
    print("\n=== Checking Conflicts ===")
    conflicts = manager.check_conflicts("./exports/another_function.metta")
    print(f"Dry run result: {conflicts['message']}")
    
    # Show atomspace stats
    print("\n=== Atomspace Statistics ===")
    stats = manager.get_atomspace_stats()
    print(f"Total atoms: {stats['total_atoms']}")
    print(f"Type distribution: {stats['type_distribution']}")
    
    # List functions
    print(f"Functions in atomspace: {manager.list_functions()}")