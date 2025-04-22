import git
from static_analyzer import decompose_source
from tqdm import tqdm
import glob
import hashlib
import time
import os
from hyperon import *
from dynamic_monitor import DynamicMonitor
from typing import Dict, Any, Optional

TEMPORAL_RULE_PATH = "metta/temporal.metta"

def build_historical_knowledge_base(repo_path, output_file):
    """Build a MeTTa knowledge base from Git history."""
    # Initialize MeTTa
    metta = MeTTa()
    space = metta.space()
    
    # Get Git commit history
    repo = git.Repo(repo_path)
    commits = list(repo.iter_commits('master'))
    
    # Process each commit
    for commit in tqdm(commits, desc="Processing commits"):
        commit_id = commit.hexsha
        author = commit.author.name
        timestamp = str(commit.committed_datetime)
        message = commit.message.strip()
        
        # Add commit info to MeTTa - proper atom creation
        commit_info_atom = metta.parse_single(f'(commit-info "{commit_id}" "{author}" "{timestamp}" "{message}")')
        space.add_atom(commit_info_atom)
        
        # Checkout this commit
        repo.git.checkout(commit_id, force=True)
        
        # Analyze Python files at this commit
        for python_file in glob.glob(f"{repo_path}/**/*.py", recursive=True):
            try:
                with open(python_file, 'r') as f:
                    code = f.read()
                
                # Use static analyzer to extract functions
                analysis = decompose_source(code)
                
                # Add function state atoms
                for func in analysis.get("functions", []):
                    func_name = func["name"]
                    signature = func.get("signature", "")
                    body_hash = hashlib.md5(func.get("body", "").encode()).hexdigest()
                    complexity = func.get("complexity", 0)
                    
                    # Create MeTTa atoms - parse first, then add
                    sig_atom = metta.parse_single(
                        f'(function-signature-at "{func_name}" "{commit_id}" "{signature}")')
                    space.add_atom(sig_atom)
                    
                    body_atom = metta.parse_single(
                        f'(function-body-at "{func_name}" "{commit_id}" "{body_hash}")')
                    space.add_atom(body_atom)
                    
                    complexity_atom = metta.parse_single(
                        f'(function-complexity-at "{func_name}" "{commit_id}" {complexity})')
                    space.add_atom(complexity_atom)
                
                # Add dependency information
                for caller, callees in analysis.get("function_calls", {}).items():
                    for callee in callees:
                        dep_atom = metta.parse_single(
                            f'(dependency-at "{caller}" "{callee}" "{commit_id}" 1)')
                        space.add_atom(dep_atom)
            
            except Exception as e:
                print(f"Error processing {python_file} at commit {commit_id}: {e}")
    
    # Return to the original state
    repo.git.checkout('master')
    
    # Save the MeTTa space to file
    with open(output_file, 'w') as f:
        # Export all atoms - get as actual atoms
        atoms = space.query(metta.parse_single("(match &self $x $x)"))
        for atom in atoms:
            f.write(str(atom) + "\n")
    
    return output_file

class TemporalCodeAnalyzer:
    def __init__(self, repo_path: str, monitor: DynamicMonitor):
        """Initialize with repository path and MeTTa monitor."""
        self.repo_path = repo_path
        self.monitor = monitor
        self.repo = None
        self.metta = MeTTa()
        self.metta_space = self.metta.space()
        
        if os.path.exists(os.path.join(repo_path, '.git')):
            self.repo = git.Repo(repo_path)
        else:
            print(f"No git repository found at {repo_path}")
        
        self.load_metta_rules(TEMPORAL_RULE_PATH)
        
        # Merge rules space into monitor space
        for atom in self.metta_space.get_atoms():
            self.monitor.metta_space.add_atom(atom)
    
    def load_metta_rules(self, rules_file: str) -> bool:
        """
        Load MeTTa rules from a file by properly parsing the content first.
        
        This follows the Hyperon MeTTa implementation's expected usage pattern.
        """
        try:
            # Open and read the file
            with open(rules_file, 'r') as f:
                file_content = f.read()
            
            # Parse the content to get actual MeTTa atoms
            parsed_atoms = self.metta.parse_all(file_content)
            
            # Add each parsed atom to our space
            atom_count = 0
            for atom in parsed_atoms:
                try:
                    self.metta_space.add_atom(atom)
                    atom_count += 1
                except Exception as atom_err:
                    print(f"Error adding atom: {atom}")
                    print(f"  Error details: {atom_err}")
            
            print(f"Successfully loaded {atom_count}/{len(parsed_atoms)} rules from {rules_file}")
            return atom_count > 0
        
        except Exception as e:
            print(f"Error loading MeTTa rules: {e}")
            
            # Fallback approach using run and load-ascii
            try:
                print("Trying alternate approach with load-ascii...")
                
                # Create a temporary binding for our space
                space_name = f"&rules_space_{int(time.time())}"
                
                # Bind and load
                self.metta.run(f'''
                    !(bind! {space_name} {self.metta_space})
                    !(load-ascii {space_name} "{rules_file}")
                ''')
                
                print(f"Successfully loaded rules using load-ascii approach")
                return True
            except Exception as e2:
                print(f"Error in alternate approach: {e2}")
                return False
    
    def analyze_history(self, max_commits: Optional[int] = None) -> bool:
        """
        Analyze Git history and populate MeTTa space with temporal information.
        
        Args:
            max_commits: Maximum number of commits to process (None for all)
        
        Returns:
            Success status
        """
        if not self.repo:
            print("No Git repository available for analysis")
            return False
        
        # Get commit history (most recent first)
        commits = list(self.repo.iter_commits('HEAD'))
        if max_commits:
            commits = commits[:max_commits]
        
        print(f"Analyzing {len(commits)} commits in Git history")
        
        # Add commit info to MeTTa space
        for commit in tqdm(commits, desc="Processing commits"):
            commit_id = commit.hexsha
            author = commit.author.name
            timestamp = str(commit.committed_datetime)
            message = commit.message.strip()
            
            # Add commit metadata
            commit_info_atom = self.monitor.metta.parse_single(
                f'(commit-info "{commit_id}" "{author}" "{timestamp}" "{message}")')
            self.monitor.metta_space.add_atom(commit_info_atom)
            
            # Checkout this commit
            original_head = self.repo.head.object.hexsha
            self.repo.git.checkout(commit_id, force=True)
            
            try:
                # Find all Python files at this commit
                python_files = []
                for root, _, files in os.walk(self.repo_path):
                    if '.git' in root:
                        continue
                    for file in files:
                        if file.endswith('.py'):
                            python_files.append(os.path.join(root, file))
                
                # Analyze each Python file
                for py_file in python_files:
                    try:
                        # Static analysis of the file
                        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                            code = f.read()
                        
                        rel_path = os.path.relpath(py_file, self.repo_path)
                        
                        # Get function information
                        analysis = decompose_source(code)
                        
                        # Extract function information
                        self._process_file_analysis(analysis, rel_path, commit_id)
                    except Exception as e:
                        print(f"Error analyzing {py_file} at commit {commit_id}: {e}")
            finally:
                # Restore original HEAD
                self.repo.git.checkout(original_head, force=True)
        
        return True
    
    def _process_file_analysis(self, analysis: Dict[str, Any], file_path: str, commit_id: str):
        """Process static analysis results for a file at a specific commit."""
        # Extract function definitions
        for atom in analysis.get("atoms", []):
            if atom.get("type") == "function_def":
                func_name = f"{file_path}:{atom['name']}"  # Include file path for uniqueness
                signature = self._get_function_signature(atom)
                complexity = self._calculate_complexity(atom)
                body_hash = self._hash_function_body(atom)
                
                # Add function signature at commit
                sig_atom = self.monitor.metta.parse_single(
                    f'(function-signature-at "{func_name}" "{commit_id}" "{signature}")')
                self.monitor.metta_space.add_atom(sig_atom)
                
                # Add function body hash at commit
                body_atom = self.monitor.metta.parse_single(
                    f'(function-body-at "{func_name}" "{commit_id}" "{body_hash}")')
                self.monitor.metta_space.add_atom(body_atom)
                
                # Add function complexity at commit
                complexity_atom = self.monitor.metta.parse_single(
                    f'(function-complexity-at "{func_name}" "{commit_id}" {complexity})')
                self.monitor.metta_space.add_atom(complexity_atom)
        
        # Extract function calls
        for func_dep in analysis.get("function_dependencies", {}).items():
            caller, callees = func_dep
            caller_name = f"{file_path}:{caller}"
            
            for callee in callees:
                # Add function dependency at commit
                dep_atom = self.monitor.metta.parse_single(
                    f'(dependency-at "{caller_name}" "{callee}" "{commit_id}" 1)')
                self.monitor.metta_space.add_atom(dep_atom)
    
    def _get_function_signature(self, func_atom: Dict[str, Any]) -> str:
        """Generate function signature from function atom."""
        params = func_atom.get("params", [])
        param_types = []
        for param in params:
            param_name, param_type = param
            param_types.append(f"{param_name}: {param_type or 'Any'}")
        
        return_type = func_atom.get("return_type", "Any")
        
        return f"({', '.join(param_types)}) -> {return_type}"

    def _calculate_complexity(self, func_atom: Dict[str, Any]) -> int:
        """Calculate cyclomatic complexity from function atom."""
        # Simplified complexity calculation
        complexity = 1  # Base complexity
        
        # Add complexity for each control flow branch
        # This is a simplified approximation
        lineno_start = func_atom.get("line_start", 0)
        lineno_end = func_atom.get("line_end", 0)
        
        # In a real implementation, we would count conditionals, loops, etc.
        # Here we use line count as a crude approximation
        complexity += (lineno_end - lineno_start) // 5
        
        return max(1, complexity)
    
    def _hash_function_body(self, func_atom: Dict[str, Any]) -> str:
        """Generate a hash of function body for change detection."""
        # In a real implementation, we would extract the function body
        # Here we use a simplified approach with line ranges
        lineno_start = func_atom.get("line_start", 0)
        lineno_end = func_atom.get("line_end", 0)
        name = func_atom.get("name", "")
        
        # Create a unique hash from available information
        hash_input = f"{name}:{lineno_start}:{lineno_end}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def function_history(self, func_name):
        """Get complete history of a function's evolution."""
        # Create query atom
        query_atom = self.metta.parse_single(f'(match &self (function-signature-at "{func_name}" $commit $_) $commit)')
        commits = self.monitor.metta_space.query(query_atom)
        
        history = []
        for commit in commits:
            commit_id = str(commit).strip('"')
            
            # Create atoms for querying signature and complexity
            sig_query = self.metta.parse_single(f'(match &self (function-signature-at "{func_name}" "{commit_id}" $sig) $sig)')
            signature_result = self.monitor.metta_space.query(sig_query)
            signature = signature_result[0] if signature_result else None
            
            complex_query = self.metta.parse_single(f'(match &self (function-complexity-at "{func_name}" "{commit_id}" $c) $c)')
            complexity_result = self.monitor.metta_space.query(complex_query)
            complexity = complexity_result[0] if complexity_result else None
            
            # Get commit info
            info_query = self.metta.parse_single(f'(match &self (commit-info "{commit_id}" $author $timestamp $msg) ($timestamp $author $msg))')
            commit_info_result = self.monitor.metta_space.query(info_query)
            commit_info = commit_info_result[0] if commit_info_result else None
            
            if commit_info and signature is not None and complexity is not None:
                timestamp, author, message = str(commit_info).strip('()').split(' ', 2)
                history.append({
                    'commit_id': commit_id,
                    'timestamp': timestamp.strip('"'),
                    'author': author.strip('"'),
                    'message': message.strip('"'),
                    'signature': str(signature).strip('"'),
                    'complexity': int(str(complexity))
                })
        
        # Sort by timestamp
        history.sort(key=lambda x: x['timestamp'])
        return history
    
    def identify_hotspots(self):
        """Identify code hotspots based on change frequency and complexity."""
        # Create query atom for hotspots
        hotspot_query = self.metta.parse_single('(match &self (function-hotspot $func $confidence) ($func $confidence))')
        hotspots = self.monitor.metta_space.query(hotspot_query)
        
        results = []
        for hotspot in hotspots:
            func, confidence = str(hotspot).strip('()').split(' ')
            
            # Get function summary
            summary = self.function_evolution_summary(func.strip('"'))
            results.append({
                "function": func.strip('"'),
                "confidence": confidence.strip('"'),
                "total_changes": summary.get("total_changes", 0),
                "complexity_change": summary.get("complexity_change", 0)
            })
        
        return sorted(results, key=lambda x: x['total_changes'], reverse=True)