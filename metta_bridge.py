from hyperon import MeTTa
import os

class MeTTaBridge:
    """Bridge between Python and MeTTa using Hyperon."""
    
    def __init__(self, metta_file=None):
        """Initialize the bridge with optional MeTTa file."""
        self.metta = MeTTa()
        if metta_file and os.path.exists(metta_file):
            self.load_metta_file(metta_file)
    
    def load_metta_file(self, file_path):
        """Load a MeTTa file into the current space."""
        self.metta.run(f'!(import! &self "{file_path}")')
    
    def add_atoms_to_space(self, atoms_list):
        """Add a list of atom strings to the current space."""
        for atom in atoms_list:
            self.metta.run(atom)
    
    def query(self, query_str):
        """Execute a query and return results."""
        return self.metta.run(query_str)
    
    def find_donor_for_error(self, error_type, message, operation, left_type, right_type):
        """Query the donor system for a fix for the given error."""
        query = f'!(find-donor-for-error ({error_type} "{message}" {operation} {left_type} {right_type}))'
        results = self.query(query)
        
        if results and len(results) > 0:
            # Return the result as a string
            return str(results[0])
        return None
    
    def get_donor_with_context(self, context, operation, left_type, right_type):
        """Get a donor with specific context awareness."""
        query = f'!(get-donor-with-context "{context}" {operation} {left_type} {right_type})'
        results = self.query(query)
        
        if results and len(results) > 0:
            return str(results[0])
        return None
    
    def get_all_donors(self):
        """Get all donors in the system."""
        donors = {
            "code_donors": self.query('!(match &self (= (code-donor $name) $fix) ($name $fix))'),
            "operation_donors": self.query('!(match &self (= (operation-donor $op $left $right) $fix) (($op $left $right) $fix))'),
            "conversion_donors": self.query('!(match &self (= (convert-donor $from $to) $fix) (($from $to) $fix))'),
            "function_donors": self.query('!(match &self (= (function-donor $name) $fix) ($name $fix))')
        }
        return donors

    def analyze_python_errors(self, error_atoms):
        """Analyze a list of error atoms and find fixes."""
        fixes = []
        for error_atom in error_atoms:
            # Parse the error atom string to extract components
            # This is a very basic parser for demonstration - a more robust one would be needed
            if error_atom.startswith("(TypeError"):
                parts = error_atom.strip("()").split(" ", 4)
                if len(parts) >= 5:
                    error_type = parts[0]
                    message = parts[1].strip('"')
                    operation = parts[2]
                    left_type = parts[3]
                    right_type = parts[4]
                    
                    # Find a fix
                    fix = self.find_donor_for_error(error_type, message, operation, left_type, right_type)
                    fixes.append({
                        "error": error_atom,
                        "fix": fix
                    })
        
        return fixes

if __name__ == "__main__":
    # Example usage
    bridge = MeTTaBridge("donor_system.metta")
    
    # Example error
    error = "(TypeError \"Cannot add string and number\" Add String Number)"
    
    # Find a fix
    fix = bridge.find_donor_for_error("TypeError", "Cannot add string and number", "Add", "String", "Number")
    
    print(f"Error: {error}")
    print(f"Fix: {fix}")