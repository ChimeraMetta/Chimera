#!/bin/bash
# Script to run binary search proof generation tests with detailed reporting

# Setup colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -k, --key API_KEY    OpenAI API key to use"
    echo "  -f, --file KEY_FILE  File containing the OpenAI API key"
    echo "  -t, --test TEST_NAME Run specific test only"
    echo "  -s, --save           Save generated proofs to files"
    echo "  -v, --verbose        Run tests with verbose output"
    echo "  -h, --help           Display this help message"
    echo ""
    echo "Available tests:"
    echo "  test_basic_proof_generation"
    echo "  test_proof_component_types"
    echo "  test_detailed_proof_analysis"
    echo "  test_metta_integration"
    echo "  test_proof_robustness"
    echo ""
    echo "Examples:"
    echo "  $0 -k sk-your-api-key-here"
    echo "  $0 -f .openai_key -s"
    echo "  $0 -t test_basic_proof_generation"
}

# Parse command line arguments
API_KEY="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"
KEY_FILE=""
TEST_NAME=""
VERBOSE=""
SAVE_PROOFS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--key)
            API_KEY="$2"
            shift 2
            ;;
        -f|--file)
            KEY_FILE="$2"
            shift 2
            ;;
        -t|--test)
            TEST_NAME="$2"
            shift 2
            ;;
        -s|--save)
            SAVE_PROOFS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# If key file specified, read key from file
if [ -n "$KEY_FILE" ]; then
    if [ -f "$KEY_FILE" ]; then
        API_KEY=$(cat "$KEY_FILE" | tr -d '\n')
        echo -e "${GREEN}Read API key from $KEY_FILE${NC}"
    else
        echo -e "${RED}Error: Key file $KEY_FILE not found${NC}"
        exit 1
    fi
fi

# Check if API key is provided
if [ -z "$API_KEY" ]; then
    echo -e "${YELLOW}Warning: No API key provided${NC}"
    echo -e "${YELLOW}Checking environment variable OPENAI_API_KEY...${NC}"
    
    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${RED}Error: No API key found. Please provide an API key.${NC}"
        usage
        exit 1
    else
        echo -e "${GREEN}Using API key from environment variable OPENAI_API_KEY${NC}"
    fi
else
    # Export API key to environment variable
    export OPENAI_API_KEY="$API_KEY"
    echo -e "${GREEN}API key set from command line argument${NC}"
fi

# Create output directory for saved proofs
if [ "$SAVE_PROOFS" = true ]; then
    PROOF_DIR="generated_proofs"
    mkdir -p "$PROOF_DIR"
    echo -e "${BLUE}Proofs will be saved to ${PROOF_DIR}/${NC}"
fi

# Check if proof verification helper exists
HELPER_FILE="proof_verification_helper.py"
if [ ! -f "$HELPER_FILE" ]; then
    echo -e "${YELLOW}Proof verification helper not found, creating it...${NC}"
    
    # Create helper script
    cat > "$HELPER_FILE" << 'EOF'
import json
import re
import sys
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
        
    @staticmethod
    def main():
        """Command line interface for proof verification."""
        if len(sys.argv) < 2:
            print("Usage: python proof_verification_helper.py <proof_json_file> [report_output_file]")
            sys.exit(1)
        
        try:
            # Read proof JSON
            with open(sys.argv[1], 'r') as f:
                proof_json = json.load(f)
            
            # Analyze proof
            analysis = ProofVerificationHelper.analyze_proof_structure(proof_json)
            issues = ProofVerificationHelper.check_binary_search_correctness(proof_json.get("proof_components", []))
            
            # Generate report
            report = ProofVerificationHelper.format_proof_report(proof_json)
            
            # Print or save report
            if len(sys.argv) >= 3:
                with open(sys.argv[2], 'w') as f:
                    f.write(report)
                print(f"Report saved to {sys.argv[2]}")
            else:
                print(report)
            
            # Print summary
            print(f"\nCompleteness score: {analysis['completeness_score']}/{analysis['max_score']}")
            if issues:
                print(f"Issues found: {len(issues)}")
                for issue in issues:
                    print(f"- {issue['description']}")
            else:
                print("No issues found")
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    ProofVerificationHelper.main()
EOF

    chmod +x "$HELPER_FILE"
    echo -e "${GREEN}Created proof verification helper${NC}"
fi

# Check if test file exists
TEST_FILE="binary_search_proof_generation_test.py"
if [ ! -f "$TEST_FILE" ]; then
    echo -e "${RED}Error: Test file $TEST_FILE not found${NC}"
    echo -e "${YELLOW}Please make sure you have created the test file first${NC}"
    exit 1
fi

# Create proof extraction utility
EXTRACTOR_FILE="extract_proof.py"
cat > "$EXTRACTOR_FILE" << 'EOF'
#!/usr/bin/env python3
"""
Utility to extract and format proofs from JSON to Markdown.
"""
import json
import sys
import os

def format_proof(proof_json):
    """Format a proof in markdown."""
    components = proof_json.get("proof_components", [])
    strategy = proof_json.get("verification_strategy", {})
    
    # Start building report
    report = "# Binary Search Proof\n\n"
    
    # Verification strategy
    report += "## Verification Strategy\n\n"
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
    report += "## Proof Components\n\n"
    
    # Group components by type
    component_groups = {}
    for component in components:
        comp_type = component.get("type", "unknown")
        if comp_type not in component_groups:
            component_groups[comp_type] = []
        component_groups[comp_type].append(component)
    
    # Format each group
    for comp_type, comps in component_groups.items():
        report += f"### {comp_type.replace('_', ' ').title()} ({len(comps)})\n\n"
        
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
    
    return report

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.json> [output.md]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}.md"
    
    try:
        with open(input_file, 'r') as f:
            proof_json = json.load(f)
        
        report = format_proof(proof_json)
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Successfully extracted proof from {input_file} to {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x "$EXTRACTOR_FILE"
echo -e "${GREEN}Created proof extraction utility${NC}"

# Run the tests
if [ -n "$TEST_NAME" ]; then
    echo -e "${BLUE}Running test: ${TEST_NAME}${NC}"
    if [ -x "$(command -v python3)" ]; then
        PYTHONPATH=. python3 -m unittest $VERBOSE binary_search_proof_generation_test.BinarySearchProofGenerationTest.$TEST_NAME
    else
        PYTHONPATH=. python -m unittest $VERBOSE binary_search_proof_generation_test.BinarySearchProofGenerationTest.$TEST_NAME
    fi
else
    echo -e "${BLUE}Running all binary search proof generation tests...${NC}"
    if [ -x "$(command -v python3)" ]; then
        PYTHONPATH=. python3 -m unittest $VERBOSE binary_search_proof_generation_test.BinarySearchProofGenerationTest
    else
        PYTHONPATH=. python -m unittest $VERBOSE binary_search_proof_generation_test.BinarySearchProofGenerationTest
    fi
fi

TEST_RESULT=$?

# If tests passed and saving is enabled, try to extract proofs
if [ $TEST_RESULT -eq 0 ] && [ "$SAVE_PROOFS" = true ]; then
    echo -e "${GREEN}Tests passed! Extracting proofs...${NC}"
    
    # Create the output directory if it doesn't exist
    mkdir -p "$PROOF_DIR"
    
    # Run a simple script to extract proof from MeTTa space
    EXTRACT_SCRIPT="extract_from_metta.py"
    cat > "$EXTRACT_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
Extract proof components from MeTTa space after test run.
"""
import sys
import os
import json
from binary_search_proof_generation_test import BinarySearchProofGenerationTest

def extract_proofs(output_dir):
    # Create a test instance
    test = BinarySearchProofGenerationTest('test_basic_proof_generation')
    test.setUp()
    
    try:
        # Run the test to generate proof
        test.test_basic_proof_generation()
        
        # Extract components from MeTTa space
        loop_invariants = test.monitor.query("(match &self (LoopInvariant $loc $expr) ($loc $expr))")
        preconditions = test.monitor.query("(match &self (Precondition $expr) $expr)")
        assertions = test.monitor.query("(match &self (Assertion $loc $expr) ($loc $expr))")
        
        print(f"Found {len(loop_invariants)} loop invariants")
        print(f"Found {len(preconditions)} preconditions")
        print(f"Found {len(assertions)} assertions")
        
        # Create a JSON representation
        proof_json = {
            "proof_components": [],
            "verification_strategy": {
                "approach": "Binary search verification approach",
                "key_lemmas": [
                    "Array is sorted",
                    "Loop invariant: target in current range if it exists",
                    "Binary search terminates"
                ]
            }
        }
        
        # Add loop invariants
        for loc, expr in loop_invariants:
            proof_json["proof_components"].append({
                "type": "loop_invariant",
                "location": str(loc),
                "expression": str(expr),
                "natural_language": "Loop invariant extracted from MeTTa"
            })
        
        # Add preconditions
        for expr in preconditions:
            proof_json["proof_components"].append({
                "type": "precondition",
                "expression": str(expr),
                "natural_language": "Precondition extracted from MeTTa"
            })
        
        # Add assertions
        for loc, expr in assertions:
            proof_json["proof_components"].append({
                "type": "assertion",
                "location": str(loc),
                "expression": str(expr),
                "natural_language": "Assertion extracted from MeTTa"
            })
        
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        json_file = os.path.join(output_dir, "extracted_proof.json")
        with open(json_file, 'w') as f:
            json.dump(proof_json, f, indent=2)
        
        print(f"Saved extracted proof to {json_file}")
        return json_file
        
    except Exception as e:
        print(f"Error extracting proof: {e}")
        return None
    finally:
        test.tearDown()

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        output_dir = sys.argv[1]
    else:
        output_dir = "generated_proofs"
    
    extracted_file = extract_proofs(output_dir)
    
    if extracted_file:
        # Convert to markdown
        md_file = os.path.splitext(extracted_file)[0] + ".md"
        os.system(f"python extract_proof.py {extracted_file} {md_file}")
EOF

    chmod +x "$EXTRACT_SCRIPT"
    
    # Run the extraction script
    if [ -x "$(command -v python3)" ]; then
        python3 "$EXTRACT_SCRIPT" "$PROOF_DIR"
    else
        python "$EXTRACT_SCRIPT" "$PROOF_DIR"
    fi
    
    # Show the generated files
    echo ""
    echo -e "${BLUE}Generated proof files:${NC}"
    ls -la "$PROOF_DIR"
    
    # Provide instructions for viewing
    if [ -f "$PROOF_DIR/extracted_proof.md" ]; then
        echo ""
        echo -e "${GREEN}You can view the extracted proof with:${NC}"
        echo -e "  cat $PROOF_DIR/extracted_proof.md"
    fi
fi

# Final status
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Tests failed!${NC}"
    exit 1
fi