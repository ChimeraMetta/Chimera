import os
import sys
from type_extractor import decompose_python_file
from metta_bridge import MeTTaBridge

def main():
    """Main function to analyze a Python file and suggest fixes."""
    # Check for file argument
    if len(sys.argv) < 2:
        print("Usage: python main.py [python_file_to_analyze]")
        return
    
    python_file = sys.argv[1]
    if not os.path.exists(python_file):
        print(f"File not found: {python_file}")
        return
    
    print(f"Analyzing {python_file}...")
    
    # Extract type information and errors
    extracted_info = decompose_python_file(python_file)
    
    # Load the MeTTa donor system
    bridge = MeTTaBridge("donor_system.metta")
    
    # Add the extracted atoms to the space
    bridge.add_atoms_to_space(extracted_info["type_atoms"])
    
    # Analyze errors and find fixes
    if not extracted_info["error_atoms"]:
        print("No type errors detected.")
        return
    
    print(f"Found {len(extracted_info['error_atoms'])} potential type errors:")
    
    # Find fixes for each error
    for i, error_atom in enumerate(extracted_info["error_atoms"]):
        print(f"\nError {i+1}: {error_atom}")
        
        # Parse the error atom to extract components
        parts = error_atom.strip("()").split(" ", 4)
        if len(parts) >= 5:
            error_type = parts[0]
            message = parts[1].strip('"')
            operation = parts[2]
            left_type = parts[3]
            right_type = parts[4]
            
            # Determine context from file name
            context = "general"
            if "receipt" in python_file.lower():
                context = "receipt_formatting"
            
            # Find a fix with context
            fix = bridge.get_donor_with_context(context, operation, left_type, right_type)
            
            if fix:
                print(f"Suggested fix (context: {context}):\n{fix}")
            else:
                # Try without context
                fix = bridge.find_donor_for_error(error_type, message, operation, left_type, right_type)
                if fix:
                    print(f"Suggested fix:\n{fix}")
                else:
                    print("No specific fix found for this error.")

if __name__ == "__main__":
    main()