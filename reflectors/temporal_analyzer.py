import git
from reflectors.static_analyzer import decompose_source
from tqdm import tqdm
import glob
import hashlib
import time
import os
from hyperon import *
from reflectors.dynamic_monitor import DynamicMonitor
from typing import Dict, Any, Optional
import logging

# Added logger instance
logger = logging.getLogger(__name__)

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
                logger.error(f"Error processing {python_file} at commit {commit_id}: {e}")
    
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
            logger.info(f"No git repository found at {repo_path}")
        
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
                    logger.error(f"Error adding atom: {atom}")
                    logger.error(f"  Error details: {atom_err}")
            
            logger.info(f"Successfully loaded {atom_count}/{len(parsed_atoms)} rules from {rules_file}")
            return atom_count > 0
        
        except Exception as e:
            logger.error(f"Error loading MeTTa rules: {e}")
            
            # Fallback approach using run and load-ascii
            try:
                logger.info("Trying alternate approach with load-ascii...")
                
                # Create a temporary binding for our space
                space_name = f"&rules_space_{int(time.time())}"
                
                # Bind and load
                self.metta.run(f'''
                    !(bind! {space_name} {self.metta_space})
                    !(load-ascii {space_name} "{rules_file}")
                ''')
                
                logger.info(f"Successfully loaded rules using load-ascii approach")
                return True
            except Exception as e2:
                logger.error(f"Error in alternate approach: {e2}")
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
            logger.info("No Git repository available for analysis")
            return False
        
        # Get commit history (most recent first)
        commits = list(self.repo.iter_commits('HEAD'))
        if max_commits:
            commits = commits[:max_commits]
        
        logger.info(f"Analyzing {len(commits)} commits in Git history")
        
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
                        logger.error(f"Error analyzing {py_file} at commit {commit_id}: {e}")
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
        """Get complete history of a function's evolution with improved error handling."""
        # Create query atom for finding all commits with this function
        query_atom = self.metta.parse_single(f'(function-commits "{func_name}" $commits)')
        commits_result = self.monitor.metta_space.query(query_atom)
        
        if not commits_result or not commits_result[0]:
            logger.info(f"No history found for function {func_name}")
            return []
        
        # Try to get the sorted commits using MeTTa's sorting
        commits_atom = commits_result[0]
        sort_atom = self.metta.parse_single(f'(sort-by-timestamp {commits_atom} $sorted)')
        sorted_result = self.monitor.metta_space.query(sort_atom)
        
        # If MeTTa sorting fails, use a fallback approach
        if not sorted_result or not sorted_result[0]:
            logger.warning(f"Warning: Could not sort commits in MeTTa. Using Python fallback.")
            # Extract commit IDs using atom iteration
            commit_ids = []
            for c in commits_atom.iterate():
                commit_ids.append(str(c))
            sorted_commits = self._sort_commits_by_timestamp(commit_ids)
        else:
            # Process the sorted result from MeTTa
            sorted_commits = []
            for c in sorted_result[0].iterate():
                sorted_commits.append(str(c))
        
        history = []
        for commit_id in sorted_commits:
            try:
                # Skip if not a valid commit ID
                if not commit_id or commit_id == "Empty":
                    continue
                    
                # Create atoms for querying signature and complexity
                sig_query = self.metta.parse_single(f'(function-signature-at "{func_name}" {commit_id} $sig)')
                signature_result = self.monitor.metta_space.query(sig_query)
                
                signature = None
                if signature_result and signature_result[0]:
                    signature_atom = signature_result[0]
                    signature = str(signature_atom)
                else:
                    signature = "Unknown"
                
                # Query complexity
                complex_query = self.metta.parse_single(f'(get-complexity "{func_name}" {commit_id} $c)')
                complexity_result = self.monitor.metta_space.query(complex_query)
                
                # Extract complexity value
                complexity = 0
                if complexity_result and complexity_result[0]:
                    complexity_atom = complexity_result[0]
                    try:
                        complexity = int(str(complexity_atom))
                    except ValueError:
                        pass
                
                # Get commit info
                info_query = self.metta.parse_single(f'(match &self (commit-info {commit_id} $author $timestamp $msg) ($author $timestamp $msg))')
                commit_info_result = self.monitor.metta_space.query(info_query)
                
                timestamp = "Unknown"
                author = "Unknown"
                message = "Unknown"
                
                if commit_info_result and commit_info_result[0]:
                    # Extract parts from the result atom
                    info_atom = commit_info_result[0]
                    parts = []
                    
                    # Extract the three parts using atom iteration
                    for part in info_atom.iterate():
                        parts.append(str(part))
                    
                    if len(parts) >= 3:
                        author = parts[0]
                        timestamp = parts[1]
                        message = parts[2]
                
                history.append({
                    'commit_id': commit_id,
                    'timestamp': timestamp,
                    'author': author,
                    'message': message,
                    'signature': signature,
                    'complexity': complexity
                })
            except Exception as e:
                logger.error(f"Error processing commit {commit_id}: {e}")
                continue
        
        return history

    def _sort_commits_by_timestamp(self, commit_ids):
        """Sort commit IDs by timestamp (fallback Python implementation)."""
        commit_data = []
        
        for commit_id in commit_ids:
            try:
                # Query timestamp using MeTTa
                timestamp_query = self.metta.parse_single(f'(get-commit-timestamp {commit_id} $ts)')
                ts_result = self.monitor.metta_space.query(timestamp_query)
                
                timestamp = 0
                if ts_result and ts_result[0]:
                    timestamp_atom = ts_result[0]
                    try:
                        timestamp = int(str(timestamp_atom))
                    except ValueError:
                        # If can't parse as int, use string comparison
                        timestamp = str(timestamp_atom)
                
                commit_data.append((commit_id, timestamp))
            except Exception as e:
                # If we can't get timestamp, put at the end
                logger.error(f"Error getting timestamp for {commit_id}: {e}")
                commit_data.append((commit_id, float('inf') if isinstance(timestamp, (int, float)) else "ZZZZZZ"))
        
        # Sort by timestamp
        try:
            commit_data.sort(key=lambda x: x[1])
        except Exception as e:
            # If sorting fails, return unsorted
            logger.warning(f"Warning: Failed to sort commits by timestamp: {e}")
        
        return [c[0] for c in commit_data]

    def identify_hotspots(self):
        """Identify code hotspots with improved error handling."""
        # Use the improved MeTTa rules
        hotspots = []
        
        # First get functions with high change frequency
        frequency_query = self.metta.parse_single('(match &self (function-change-frequency $func $freq) (> $freq 3) ($func $freq))')
        frequency_result = self.monitor.metta_space.query(frequency_query)
        
        for result in frequency_result:
            try:
                # Extract function name and frequency by iterating through the atom
                parts = []
                for part in result.iterate():
                    parts.append(str(part))
                
                if len(parts) >= 2:
                    func_name, freq = parts[0], parts[1]
                    try:
                        freq_val = int(freq)
                    except ValueError:
                        freq_val = 0
                    
                    hotspots.append({
                        "function": func_name,
                        "metric": "change_frequency",
                        "value": freq_val,
                        "confidence": "high" if freq_val > 5 else "medium"
                    })
            except Exception as e:
                logger.error(f"Error processing frequency result: {e}")
        
        # Then get functions with high complexity
        complexity_query = self.metta.parse_single('(match &self (complexity-hotspot $func High) $func)')
        complexity_result = self.monitor.metta_space.query(complexity_query)
        
        for func_atom in complexity_result:
            func_name = str(func_atom)
            hotspots.append({
                "function": func_name,
                "metric": "complexity",
                "value": self._get_max_complexity(func_name),
                "confidence": "high"
            })
        
        # Sort by confidence and value
        hotspots.sort(key=lambda x: (0 if x["confidence"] == "high" else 1, -x["value"]))
        
        return hotspots

    def _get_max_complexity(self, func_name):
        """Get maximum complexity for a function across all commits."""
        query = self.metta.parse_single(f'(match &self (function-complexity-at "{func_name}" $commit $complexity) $complexity)')
        complexity_results = self.monitor.metta_space.query(query)
        
        max_complexity = 0
        for c_atom in complexity_results:
            try:
                complexity = int(str(c_atom))
                max_complexity = max(max_complexity, complexity)
            except ValueError:
                pass
        
        return max_complexity

    def function_evolution_summary(self, func_name):
        """Get summary metrics for function evolution."""
        history = self.function_history(func_name)
        
        if not history:
            return {"total_changes": 0, "complexity_change": 0}
        
        # Count total versions
        total_changes = len(history)
        
        # Calculate complexity change from first to last version
        try:
            first_complexity = history[0].get('complexity', 0)
            last_complexity = history[-1].get('complexity', 0)
            complexity_change = last_complexity - first_complexity
        except (IndexError, KeyError):
            complexity_change = 0
        
        return {
            "total_changes": total_changes,
            "complexity_change": complexity_change,
            "first_commit": history[0].get('commit_id') if history else None,
            "last_commit": history[-1].get('commit_id') if history else None
        }

    # For full_analyzer.py - Helper function to process MeTTa results properly
    def process_temporal_result(result_atom):
        """Extract data from MeTTa result atoms properly."""
        try:
            # For atoms that implement iterate
            if hasattr(result_atom, 'iterate'):
                parts = []
                for part in result_atom.iterate():
                    parts.append(str(part))
                
                if len(parts) == 1:
                    return parts[0], None
                elif len(parts) >= 2:
                    return parts[0], parts[1]
            
            # Fallback for other types of atoms
            result_str = str(result_atom)
            if "(" in result_str and ")" in result_str:
                # This is a less reliable fallback
                return None, None
            
            return result_str, None
        except Exception as e:
            logger.error(f"Error processing result {result_atom}: {e}")
            return None, None

    # Modify analyze_temporal_evolution in full_analyzer.py to use this method
    def get_temporal_functions_with_changes(monitor):
        """Get functions with frequent changes using proper atom processing."""
        try:
            # Query for functions with change frequency > 3
            query = monitor.metta.parse_single('(match &self (function-change-frequency $func $freq) (> $freq 3) ($func $freq))')
            results = monitor.metta_space.query(query)
            
            functions = []
            for result in results:
                try:
                    # Process each result using atom iteration
                    parts = []
                    for part in result.iterate():
                        parts.append(str(part))
                    
                    if len(parts) >= 2:
                        func_name, freq = parts[0], parts[1]
                        try:
                            freq_val = int(freq)
                            functions.append((func_name, freq_val))
                        except ValueError:
                            pass
                except Exception as e:
                    logger.error(f"Error processing result: {e}")
            
            return functions
        except Exception as e:
            logger.error(f"Error querying functions with changes: {e}")
            return []