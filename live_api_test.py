#!/usr/bin/env python3
"""
Debug script for testing OpenAI API connectivity and MeTTa integration.
Run this before running the full test suite to identify issues.
"""

import os
import sys
import json
import logging
import traceback

from hyperon import *
import requests
from dynamic_monitor import DynamicMonitor
from proofs.generator import MettaProofGenerator, OpenAIRequests
from proofs.pattern_mapper import PatternMapper
from proofs.processor import ProofProcessorWithPatterns
from proofs.analyzer import ImmuneSystemProofAnalyzer

API_KEY = "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"

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("debug_api_test")

def check_api_key():
    """Check that OpenAI API key is available and properly formatted."""
    logger.info("Checking OpenAI API key...")
    
    # Try multiple sources for the API key
    api_key = API_KEY
    
    if not api_key.startswith(("sk-", "sk-proj-")):
        logger.warning("API key format doesn't match expected pattern")
    
    return api_key

def test_api_connectivity(api_key):
    """Test connectivity to OpenAI API."""
    logger.info("Testing OpenAI API connectivity...")
    
    try:
        # Use gpt-3.5-turbo as it's cheaper
        client = OpenAIRequests(api_key, "gpt-3.5-turbo")
        
        # Make a simple request
        response = client.get_completion_text([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Return the word 'SUCCESS' and nothing else."}
        ])
        
        logger.info(f"API response: {response}")
        
        if "SUCCESS" in response or "success" in response.lower():
            logger.info("✓ API connectivity test successful")
            return True
        else:
            logger.warning("API response doesn't contain expected keyword")
            return False
        
    except Exception as e:
        logger.error(f"✗ API connectivity test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_metta_setup():
    """Test MeTTa setup and basic operations."""
    logger.info("Testing MeTTa setup...")
    
    try:
        # Create monitor
        monitor = DynamicMonitor()
        
        # Test adding a simple atom
        test_atom = "(: test-atom Type)"
        success = monitor.add_atom(test_atom)
        
        if success:
            logger.info("✓ Successfully added atom to MeTTa space")
        else:
            logger.warning("Failed to add atom to MeTTa space")
        
        # Test querying
        result = monitor.query("(match &self (: test-atom $type) $type)")
        
        if result and result[0] == "Type":
            logger.info("✓ Successfully queried MeTTa space")
            return True
        else:
            logger.warning(f"Query returned unexpected result: {result}")
            return False
        
    except Exception as e:
        logger.error(f"✗ MeTTa setup test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_pattern_mapper():
    """Test PatternMapper functionality."""
    logger.info("Testing PatternMapper...")
    
    try:
        # Create pattern mapper
        pattern_mapper = PatternMapper()
        
        # Test mapping a property
        property_str = "maintains array bounds"
        result = pattern_mapper.map_requirement_to_property(property_str)
        
        logger.info(f"Mapped '{property_str}' to '{result}'")
        
        if result == "bound-check":
            logger.info("✓ PatternMapper correctly mapped property")
            return True
        else:
            logger.warning(f"PatternMapper returned unexpected mapping: {result}")
            return False
        
    except Exception as e:
        logger.error(f"✗ PatternMapper test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_simplified_json_to_metta():
    """Test simplified JSON to MeTTa conversion."""
    logger.info("Testing simplified JSON to MeTTa conversion...")
    
    try:
        # Create test data
        test_json = {
            "proof_components": [
                {
                    "type": "loop_invariant",
                    "location": "while_loop",
                    "expression": "left <= right",
                    "natural_language": "Search range is valid"
                }
            ],
            "verification_strategy": {
                "approach": "Prove invariants are maintained",
                "key_lemmas": ["Binary search terminates"]
            }
        }
        
        # Create monitor and processor
        monitor = DynamicMonitor()
        processor = ProofProcessorWithPatterns(monitor)
        processor.pattern_mapper = PatternMapper()
        
        # Add basic type definitions
        monitor.add_atom("(: Expression Type)")
        monitor.add_atom("(: Property Type)")
        monitor.add_atom("(: bound-check Property)")
        
        # Convert to MeTTa atoms
        metta_atoms = processor._json_to_metta_proof(test_json)
        
        logger.info(f"Generated {len(metta_atoms)} MeTTa atoms")
        
        # Add atoms to MeTTa space
        for atom in metta_atoms:
            monitor.add_atom(atom)
        
        # Check for loop invariant
        result = monitor.query("(match &self (LoopInvariant $loc $expr) ($loc $expr))")
        
        if result and len(result) > 0:
            logger.info("✓ Successfully converted JSON to MeTTa atoms")
            return True
        else:
            logger.warning(f"Query returned unexpected result: {result}")
            return False
        
    except Exception as e:
        logger.error(f"✗ JSON to MeTTa conversion test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all debug tests."""
    logger.info("=== Starting API Test Debug Script ===")
    
    # Track overall success
    all_tests_passed = True
    
    # Check API key
    api_key = check_api_key()
    if not api_key:
        logger.error("API key check failed. Set OPENAI_API_KEY environment variable.")
        all_tests_passed = False
    
    # Test API connectivity
    if api_key and not test_api_connectivity(api_key):
        logger.error("API connectivity test failed.")
        all_tests_passed = False
    
    # Test MeTTa setup
    if not test_metta_setup():
        logger.error("MeTTa setup test failed.")
        all_tests_passed = False
    
    # Test pattern mapper
    if not test_pattern_mapper():
        logger.error("PatternMapper test failed.")
        all_tests_passed = False
    
    # Test JSON to MeTTa conversion
    if not test_simplified_json_to_metta():
        logger.error("JSON to MeTTa conversion test failed.")
        all_tests_passed = False
    
    # Final summary
    if all_tests_passed:
        logger.info("=== All debug tests passed! ===")
        logger.info("You should be able to run the full test suite now.")
        return True
    else:
        logger.error("=== One or more debug tests failed ===")
        logger.error("Fix the issues before running the full test suite.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)