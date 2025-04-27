import json
import re
from typing import Dict, List, Any

class ProofVerificationHelper:
    """Helper class for manual verification of binary search proofs."""
    
    @staticmethod
    def analyze_proof_structure(proof_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the structure and completeness of a proof.
        
        Args:
            proof_json: The JSON IR of the proof
            
        Returns:
            Dictionary with analysis results
        """
        # Extract components
        components = proof_json.get("proof_components", [])
        
        # Count component types
        component_counts = {}
        for component in components:
            comp_type = component.get("type", "unknown")
            component_counts[comp_type] = component_counts.get(comp_type, 0) + 1
        
        # Check for required component types
        required_types = {"loop_invariant", "precondition", "assertion"}
        missing_types = required_types - set(component_counts.keys())
        
        # Extract expressions from components
        expressions = [component.get("expression", "") for component in components]
        
        # Check for key invariants
        key_invariants = {
            "loop_condition": r"left\s*<=\s*right",
            "mid_calculation": r"mid\s*=",
            "array_access": r"arr\s*\[",
            "target_reference": r"target",
            "sorted_assumption": r"sorted|order"
        }
        
        # Count matches for each invariant
        invariant_matches = {}
        for name, pattern in key_invariants.items():
            invariant_matches[name] = sum(1 for expr in expressions if re.search(pattern, expr, re.IGNORECASE))
        
        # Extract verification strategy
        strategy = proof_json.get("verification_strategy", {})
        approach = strategy.get("approach", "")
        key_lemmas = strategy.get("key_lemmas", [])
        
        # Check lemmas for important concepts
        lemma_concepts = {
            "termination": r"terminat|halt|loop",
            "correctness": r"correct|accurate|sound",
            "sorted_array": r"sorted|order",
            "binary_search": r"binary|search|divide"
        }
        
        lemma_coverage = {}
        for concept, pattern in lemma_concepts.items():
            lemma_coverage[concept] = any(re.search(pattern, lemma, re.IGNORECASE) for lemma in key_lemmas)
        
        # Assess overall completeness
        completeness_score = 0
        max_score = 10
        
        # 1. All required component types present (3 points)
        completeness_score += 3 * (1 - len(missing_types) / len(required_types))
        
        # 2. At least one invariant of each key type (3 points)
        completeness_score += 3 * sum(1 for count in invariant_matches.values() if count > 0) / len(key_invariants)
        
        # 3. Strategy has approach and lemmas (2 points)
        if approach:
            completeness_score += 1
        if key_lemmas:
            completeness_score += 1
        
        # 4. Lemma coverage (2 points)
        completeness_score += 2 * sum(1 for covered in lemma_coverage.values() if covered) / len(lemma_concepts)
        
        # Round to nearest tenth
        completeness_score = round(completeness_score, 1)
        
        # Prepare result
        result = {
            "component_counts": component_counts,
            "missing_component_types": list(missing_types),
            "invariant_coverage": invariant_matches,
            "strategy_present": bool(approach),
            "lemma_count": len(key_lemmas),
            "lemma_concept_coverage": lemma_coverage,
            "completeness_score": completeness_score,
            "max_score": max_score
        }
        
        return result
    
    @staticmethod
    def check_binary_search_correctness(proof_components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check if proof components correctly prove binary search properties.
        
        Args:
            proof_components: List of proof component dictionaries
            
        Returns:
            List of issues or empty list if all checks pass
        """
        issues = []
        
        # Define critical properties to check
        critical_properties = {
            "loop_termination": False,
            "bounds_checking": False,
            "return_correctness": False,
            "sorted_array_assumption": False
        }
        
        # Check each component for critical properties
        for component in proof_components:
            comp_type = component.get("type", "")
            expr = component.get("expression", "").lower()
            desc = component.get("natural_language", "").lower()
            text = f"{expr} {desc}"
            
            # Check loop termination
            if ("termination" in text or "progress" in text or 
                re.search(r"(left|right|mid).*?(increase|decrease)", text)):
                critical_properties["loop_termination"] = True
            
            # Check bounds checking
            if (re.search(r"(index|mid).*?(bounds|range|length)", text) or
                "out of bounds" in text or "within array" in text):
                critical_properties["bounds_checking"] = True
            
            # Check return correctness
            if (("target" in text and "found" in text) or
                ("return" in text and ("mid" in text or "index" in text)) or
                "correct index" in text):
                critical_properties["return_correctness"] = True
            
            # Check sorted array assumption
            if "sorted" in text or "order" in text or "monotonic" in text:
                critical_properties["sorted_array_assumption"] = True
        
        # Report missing properties
        for property_name, is_present in critical_properties.items():
            if not is_present:
                issues.append({
                    "issue_type": "missing_property",
                    "property": property_name,
                    "description": f"No proof component addresses {property_name.replace('_', ' ')}"
                })
        
        return issues
    
    @staticmethod
    def format_proof_report(proof_json: Dict[str, Any]) -> str:
        """
        Format a human-readable report of the proof.
        
        Args:
            proof_json: The JSON IR of the proof
            
        Returns:
            Formatted report string
        """
        components = proof_json.get("proof_components", [])
        strategy = proof_json.get("verification_strategy", {})
        
        # Start building report
        report = "## Binary Search Proof Report\n\n"
        
        # Verification strategy
        report += "### Verification Strategy\n\n"
        report += f"**Approach**: {strategy.get('approach', 'Not specified')}\n\n"
        
        # Key lemmas
        lemmas = strategy.get("key_lemmas", [])
        if lemmas:
            report += "**Key Lemmas**:\n"
            for lemma in lemmas:
                report += f"- {lemma}\n"
            report += "\n"
        else:
            report += "**Key Lemmas**: None specified\n\n"
        
        # Proof components by type
        report += "### Proof Components\n\n"
        
        # Group components by type
        component_groups = {}
        for component in components:
            comp_type = component.get("type", "unknown")
            if comp_type not in component_groups:
                component_groups[comp_type] = []
            component_groups[comp_type].append(component)
        
        # Format each group
        for comp_type, comps in component_groups.items():
            report += f"#### {comp_type.replace('_', ' ').title()} ({len(comps)})\n\n"
            
            for i, component in enumerate(comps):
                expr = component.get("expression", "")
                desc = component.get("natural_language", "")
                loc = component.get("location", "")
                
                report += f"**{i+1}. "
                if loc:
                    report += f"At {loc}: "
                report += "**\n\n"
                
                report += f"Expression: `{expr}`\n\n"
                report += f"Explanation: {desc}\n\n"
        
        # Analysis
        analysis = ProofVerificationHelper.analyze_proof_structure(proof_json)
        report += "### Proof Analysis\n\n"
        report += f"**Completeness Score**: {analysis['completeness_score']}/{analysis['max_score']}\n\n"
        
        # Missing components
        if analysis["missing_component_types"]:
            report += "**Missing Component Types**:\n"
            for missing in analysis["missing_component_types"]:
                report += f"- {missing.replace('_', ' ').title()}\n"
            report += "\n"
        
        # Issues
        issues = ProofVerificationHelper.check_binary_search_correctness(components)
        if issues:
            report += "**Issues**:\n"
            for issue in issues:
                report += f"- {issue['description']}\n"
            report += "\n"
        else:
            report += "**Issues**: None detected\n\n"
        
        return report