import re
from typing import List, Tuple

class PatternMapper:
    """
    Maps expression patterns in code to specific MeTTa atoms.
    Uses descriptive names that match the MeTTa ontology.
    """
    
    def __init__(self):
        """Initialize pattern dictionaries with regex patterns."""
        # Bounds checking patterns
        self.bounds_patterns = {
            r'(?:index|i|j|k)\s*<\s*(?:length|len|size|count)': 'index-less-than-length',
            r'(?:index|i|j|k)\s*<=\s*(?:length|len|size|count)': 'index-less-equal-length',
            r'(?:index|i|j|k)\s*>=\s*0': 'index-greater-equal-zero',
            r'0\s*<=\s*(?:index|i|j|k)\s*<\s*(?:length|len|size|count)': 'index-within-bounds'
        }
        
        # Ordering patterns
        self.ordering_patterns = {
            r'(?:array|list|collection)\s*(?:is|are)\s*sorted': 'array-is-sorted',
            r'ascending\s*order': 'ascending-order',
            r'descending\s*order': 'descending-order',
            r'preserves\s*(?:the\s*)?order': 'preserves-element-order'
        }
        
        # Null checking patterns
        self.null_patterns = {
            r'(?:value|var|x)\s*!=\s*(?:null|None)': 'value-not-null',
            r'if\s*(?:.*?)\s*(?:is|==)\s*(?:null|None)': 'checks-for-null',
            r'(?:empty|length\s*==\s*0)': 'handles-empty-collection'
        }
        
        # Termination patterns
        self.termination_patterns = {
            r'(?:i|index|counter)\s*(?:-=|=\s*.*?\s*-|decreases)': 'decreasing-loop-variable',
            r'(?:i|index|counter)\s*(?:\+=|=\s*.*?\s*\+|increases)': 'increasing-towards-bound',
            r'invariant.*?(?:progress|decreases|increases)': 'loop-invariant-progress',
            r'(?:for|iterations)\s*.*?\s*(?:times|count)': 'finite-iteration-count'
        }
        
        # Error handling patterns
        self.error_patterns = {
            r'(?:return|=)\s*(?:-1|NOT_FOUND)': 'checks-for-not-found',
            r'(?:validate|check|verify).*?(?:input|parameter)': 'validates-input',
            r'(?:edge|special|corner)\s*case': 'handles-edge-cases',
            r'(?:error|status)\s*code': 'error-code-return'
        }
        
        # Dictionary mapping pattern types to their regex dictionaries
        self.pattern_types = {
            'bound-check': self.bounds_patterns,
            'ordering-check': self.ordering_patterns,
            'null-check': self.null_patterns,
            'termination-guarantee': self.termination_patterns,
            'error-handling': self.error_patterns
        }
    
    def identify_patterns(self, expression: str, description: str = None) -> List[Tuple[str, str]]:
        """
        Identify which patterns are present in an expression and description.
        
        Args:
            expression: The logical expression to analyze
            description: Optional natural language description
            
        Returns:
            List of (pattern_atom, property_type) tuples
        """
        found_patterns = []
        text_to_analyze = (expression + " " + (description or "")).lower()
        
        # Check each pattern type
        for property_type, patterns in self.pattern_types.items():
            for pattern_regex, pattern_atom in patterns.items():
                if re.search(pattern_regex, text_to_analyze):
                    found_patterns.append((pattern_atom, property_type))
        
        return found_patterns
    
    def generate_metta_atoms(self, expr_id: str, expression: str, description: str = None) -> List[str]:
        """
        Generate MeTTa atoms for the patterns in an expression.
        
        Args:
            expr_id: Expression identifier in MeTTa
            expression: The logical expression
            description: Optional natural language description
            
        Returns:
            List of MeTTa atom strings
        """
        metta_atoms = []
        
        # Identify patterns
        patterns = self.identify_patterns(expression, description)
        
        # Generate Expression-Property and pattern-property atoms
        for pattern_atom, property_type in patterns:
            # Relate the expression to the property
            metta_atoms.append(f"(Expression-Property {expr_id} {property_type})")
            
            # Relate the pattern to the property (if not already done in ontology)
            # metta_atoms.append(f"(pattern-property {pattern_atom} {property_type})")
        
        return metta_atoms
    
    def map_requirement_to_property(self, requirement: str) -> str:
        """
        Map a donor requirement description to a property atom.
        
        Args:
            requirement: Description of the required property
            
        Returns:
            MeTTa property atom
        """
        requirement_lower = requirement.lower()
        
        # Map common requirement terms to property atoms
        if any(term in requirement_lower for term in ['bound', 'index', 'range']):
            return 'bound-check'
        
        if any(term in requirement_lower for term in ['order', 'sort', 'ascend', 'descend']):
            return 'ordering-check'
        
        if any(term in requirement_lower for term in ['null', 'none', 'empty']):
            return 'null-check'
        
        if any(term in requirement_lower for term in ['terminat', 'halt', 'stop', 'loop']):
            return 'termination-guarantee'
        
        if any(term in requirement_lower for term in ['error', 'exception', 'not found', 'valid']):
            return 'error-handling'
        
        # Default to a custom property if no match
        safe_name = requirement_lower.replace(" ", "-").replace(":", "")
        return f"property-{safe_name}"