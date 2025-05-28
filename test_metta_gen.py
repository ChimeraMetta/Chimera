#!/usr/bin/env python3
"""
End-to-End Testing Code for MeTTa Donor Generator
Tests the complete pipeline from function analysis to donor generation.
"""

import sys
import os
import traceback
from typing import List, Dict, Any

# Add the project root to the path so we can import our modules
# Adjust this path based on your project structure
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our components
from reflectors.static_analyzer import decompose_function, convert_to_metta_atoms, CodeDecomposer
from reflectors.dynamic_monitor import monitor
from executors.metta_generator import MettaDonorGenerator, integrate_metta_generation


def find_max_in_range(numbers, start_idx, end_idx):
    """Find the maximum value in a list within a specific range."""
    if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
        return None
        
    max_val = numbers[start_idx]
    for i in range(start_idx + 1, end_idx):
        if numbers[i] > max_val:
            max_val = numbers[i]
        
    return max_val

def test_static_analysis():
    """Test the static analysis pipeline."""
    print("  TESTING STATIC ANALYSIS PIPELINE")
    print("=" * 50)
    
    func = find_max_in_range
    
    try:
        # Test decompose_function
        print("1. Testing decompose_function()...")
        result = decompose_function(func)
        
        if "error" in result:
            print(f" Error in decompose_function: {result['error']}")
            return False
        
        print(f" decompose_function succeeded")
        print(f" Generated {len(result.get('metta_atoms', []))} MeTTa atoms")
        print(f" Found {len(result.get('structure', []))} structural elements")
        print(f" Detected {len(result.get('function_calls', {}))} function call patterns")
        print(f" Mapped {len(result.get('variables', {}))} variable scopes")
        
        # Show sample atoms
        atoms = result.get('metta_atoms', [])
        if atoms:
            print("\n   Sample MeTTa atoms:")
            for i, atom in enumerate(atoms[:5]):
                print(f"     {i+1}. {atom}")
            if len(atoms) > 5:
                print(f"     ... and {len(atoms) - 5} more")
        
        return True
        
    except Exception as e:
        print(f" Static analysis failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_metta_integration():
    """Test MeTTa space integration."""
    print("\nTESTING METTA INTEGRATION")
    print("=" * 50)
    
    func = find_max_in_range
    
    try:
        # FIRST: Debug the monitor itself
        print("0. Debugging monitor state...")
        print("   Monitor type: {}".format(type(monitor)))
        print("   Monitor id: {}".format(id(monitor)))
        print("   Monitor metta_space type: {}".format(type(monitor.metta_space)))
        print("   Monitor metta_space id: {}".format(id(monitor.metta_space)))
        
        # Check if monitor has the expected attributes
        monitor_attrs = [attr for attr in dir(monitor) if not attr.startswith('_')]
        print("   Monitor attributes: {}".format(monitor_attrs))
        
        # Check MeTTa space attributes
        space_attrs = [attr for attr in dir(monitor.metta_space) if not attr.startswith('_')]
        print("   MeTTa space attributes: {}".format(space_attrs[:10]))
        
        # Test basic monitor functionality
        test_atom = "(test-atom simple)"
        add_result = monitor.add_atom(test_atom)
        print("   Test atom add result: {}".format(add_result))
        
        # Test MeTTa space integration
        print("1. Testing MeTTa atom loading...")
        
        # Get analysis result
        result = decompose_function(func)
        atoms = result.get('metta_atoms', [])
        
        print("   LOADING: Loading {} atoms into MeTTa space...".format(len(atoms)))
        
        # DEBUG: Show the first few atoms being loaded
        print("\n   DEBUG: First 5 atoms being loaded:")
        for i, atom in enumerate(atoms[:5]):
            print("     {}: {}".format(i+1, atom))
        
        # Load atoms and track what happens
        loaded_count = 0
        failed_count = 0
        failed_atoms = []
        
        for i, atom in enumerate(atoms):
            try:
                result = monitor.add_atom(atom)
                if result:
                    loaded_count += 1
                    if i < 3:  # Debug first few
                        print("   DEBUG: Successfully loaded atom {}: {}".format(i+1, atom))
                else:
                    failed_count += 1
                    failed_atoms.append(atom)
                    if i < 3:  # Debug first few failures
                        print("   DEBUG: Failed to load atom {}: {}".format(i+1, atom))
            except Exception as e:
                failed_count += 1
                failed_atoms.append(atom)
                print("   DEBUG: Exception loading atom {}: {} - Error: {}".format(i+1, atom, e))
        
        print("\n   SUCCESS: Loaded {}/{} atoms successfully".format(loaded_count, len(atoms)))
        if failed_count > 0:
            print("   WARNING: {} atoms failed to load".format(failed_count))
            print("   FAILED ATOMS (first 3):")
            for atom in failed_atoms[:3]:
                print("     - {}".format(atom))
        
        # Test querying with detailed space inspection
        print("\n2. Testing MeTTa space inspection...")
        
        # Method 1: String representation
        try:
            metta_space_str = str(monitor.metta_space)
            print("   MeTTa space string length: {}".format(len(metta_space_str)))
            print("   MeTTa space string (first 200 chars): '{}'".format(metta_space_str[:200]))
            
            if metta_space_str.strip():
                # If not empty, look for our patterns
                contains_function_def = "function-def" in metta_space_str
                contains_bin_op = "bin-op" in metta_space_str
                contains_loop = "loop-pattern" in metta_space_str
                
                print("   String search results:")
                print("     function-def: {}".format(contains_function_def))
                print("     bin-op: {}".format(contains_bin_op))
                print("     loop-pattern: {}".format(contains_loop))
            else:
                print("   WARNING: MeTTa space string is empty!")
                
        except Exception as e:
            print("   ERROR: Could not get MeTTa space string: {}".format(e))
        
        # Method 2: Try different space access methods
        try:
            if hasattr(monitor.metta_space, 'get_atoms'):
                space_atoms = monitor.metta_space.get_atoms()
                print("   get_atoms() returned: {} items".format(len(space_atoms) if space_atoms else 0))
            elif hasattr(monitor.metta_space, 'atoms'):
                space_atoms = monitor.metta_space.atoms
                print("   .atoms attribute: {} items".format(len(space_atoms) if space_atoms else 0))
            else:
                print("   No obvious way to get atoms from space")
                
        except Exception as e:
            print("   ERROR: Could not access space atoms: {}".format(e))
        
        # Test querying with proper MeTTa queries
        print("\n2. Testing MeTTa querying with proper query syntax...")
        
        # Test queries using the monitor.query() method
        test_queries = [
            ("function-def", "(match &self (function-def $name $scope $start $end) $name)"),
            ("bin-op", "(match &self (bin-op $op $left $right $scope $line) $op)"),
            ("loop-pattern", "(match &self (loop-pattern $id $type $scope $line) $type)"),
            ("function-return", "(match &self (function-return $func $type $line) $func)"),
            ("variable-assign", "(match &self (variable-assign $name $scope $line) $name)"),
            ("function-call", "(match &self (function-call $name $args $scope $line) $name)")
        ]
        
        for query_name, query_pattern in test_queries:
            try:
                print("   Testing query: {}".format(query_name))
                print("     Pattern: {}".format(query_pattern))
                
                # Use the proper monitor.query() method
                results = monitor.query(query_pattern)
                
                if results and len(results) > 0:
                    print("     RESULT: FOUND {} matches".format(len(results)))
                    # Show first few results
                    for i, result in enumerate(results[:3]):
                        print("       {}: {}".format(i+1, result))
                    if len(results) > 3:
                        print("       ... and {} more".format(len(results) - 3))
                else:
                    print("     RESULT: NOT FOUND (empty results)")
                    
            except Exception as e:
                print("     ERROR: Query failed - {}".format(e))
        
        # Also test some simpler existence queries
        print("\n   Testing simple existence queries...")
        simple_queries = [
            ("any-function-def", "(match &self (function-def $x $y $z $w) True)"),
            ("any-bin-op", "(match &self (bin-op $x $y $z $w $v) True)"),
            ("any-loop", "(match &self (loop-pattern $x $y $z $w) True)")
        ]
        
        for query_name, query_pattern in simple_queries:
            try:
                results = monitor.query(query_pattern)
                found = results and len(results) > 0
                print("   {}: {}".format(query_name, "FOUND" if found else "NOT FOUND"))
                
            except Exception as e:
                print("   {}: ERROR - {}".format(query_name, e))
        
        # Test the basic space string method for comparison
        print("\n3. String-based search for comparison...")
        try:
            metta_space_str = str(monitor.metta_space)
            print("   MeTTa space string length: {}".format(len(metta_space_str)))
            
            if metta_space_str.strip():
                string_results = {
                    "function-def": "function-def" in metta_space_str,
                    "bin-op": "bin-op" in metta_space_str,
                    "loop-pattern": "loop-pattern" in metta_space_str,
                    "variable-assign": "variable-assign" in metta_space_str
                }
                
                print("   String search results:")
                for key, found in string_results.items():
                    count = metta_space_str.count(key)
                    print("     {}: {} (count: {})".format(key, "FOUND" if found else "NOT FOUND", count))
                
                # Show a sample of the space content
                print("   Space content sample (first 300 chars):")
                print("   '{}'".format(metta_space_str[:300]))
            else:
                print("   WARNING: MeTTa space string representation is empty!")
                
        except Exception as e:
            print("   ERROR: String-based search failed - {}".format(e))
        
        return True
        
    except Exception as e:
        print("ERROR: MeTTa integration failed: {}".format(e))
        print("   Traceback: {}".format(traceback.format_exc()))
        return False

def test_pattern_detection():
    """Test pattern detection capabilities."""
    print("\nTESTING PATTERN DETECTION")
    print("=" * 50)
    
    func = find_max_in_range
    
    try:
        # Create generator but use the global monitor's space (same as integration test)
        generator = MettaDonorGenerator(metta_space=monitor.metta_space)
        
        print("1. Setting up generator with existing MeTTa space...")
        print("   Generator metta_space id: {}".format(id(generator.metta_space)))
        print("   Global monitor space id: {}".format(id(monitor.metta_space)))
        print("   Spaces are same: {}".format(generator.metta_space is monitor.metta_space))
        
        # Extract source and analyze
        import inspect
        source_code = inspect.getsource(func)
        
        # Parse and analyze
        import ast
        tree = ast.parse(source_code)
        decomposer = CodeDecomposer()
        decomposer.visit(tree)
        
        # Get atoms from analysis
        atoms = convert_to_metta_atoms(decomposer)
        print("   Generated {} atoms from analysis".format(len(atoms)))
        
        # Set up the generator state
        generator.function_name = func.__name__
        generator.original_code = source_code
        generator.metta_atoms = atoms  # Store atoms for reference
        
        # Since we're using the same space as integration test, atoms should already be loaded
        # But let's verify by loading them again (this should be safe)
        print("\n2. Loading atoms into generator's MeTTa space...")
        generator._load_atoms_to_metta(atoms)
        
        # Test the atom summary method specifically
        print("\n3. Testing atom summary generation...")
        summary = generator._get_atoms_summary()
        print("   Atom summary result: {}".format(summary))
        
        if not summary:
            print("   DEBUG: Empty summary - investigating...")
            
            # Check if generator has stored atoms
            if hasattr(generator, 'metta_atoms'):
                print("   Generator has stored atoms: {} items".format(len(generator.metta_atoms)))
                for i, atom in enumerate(generator.metta_atoms[:3]):
                    print("     {}: {}".format(i+1, atom))
            else:
                print("   Generator has no stored atoms")
            
            # Check if space has content using queries
            try:
                func_def_query = "(match &self (function-def $name $scope $start $end) $name)"
                func_results = monitor.query(func_def_query)
                print("   Query for function-def: {} results".format(len(func_results) if func_results else 0))
                
                bin_op_query = "(match &self (bin-op $op $left $right $scope $line) $op)"
                bin_results = monitor.query(bin_op_query)
                print("   Query for bin-op: {} results".format(len(bin_results) if bin_results else 0))
                
            except Exception as e:
                print("   Query test failed: {}".format(e))
        
        # Test strategy applicability
        print("\n4. Testing strategy applicability...")
        strategies = generator._get_applicable_strategies_from_metta(None)
        
        print("   STRATEGIES: Found {} applicable strategies:".format(len(strategies)))
        for i, strategy in enumerate(strategies, 1):
            print("     {}. {}".format(i, strategy))
        
        return True
        
    except Exception as e:
        print("ERROR: Pattern detection failed: {}".format(e))
        print("   Traceback: {}".format(traceback.format_exc()))
        return False

def test_donor_generation():
    """Test the complete donor generation pipeline."""
    print("\nTESTING DONOR GENERATION")
    print("=" * 50)
    
    func = find_max_in_range
    
    try:
        print("1. Running complete donor generation pipeline...")
        
        # Use the integration function
        candidates = integrate_metta_generation(func)
        
        print(f"    Generated {len(candidates)} donor candidates")
        
        if not candidates:
            print("No candidates generated - checking for issues...")
            return False
        
        # Analyze candidates
        print("\n2. Analyzing generated candidates...")
        
        strategies_used = set()
        total_score = 0
        
        for candidate in candidates:
            strategies_used.add(candidate['strategy'])
            total_score += candidate['final_score']
        
        avg_score = total_score / len(candidates) if candidates else 0
        
        print(f"   Strategies used: {', '.join(strategies_used)}")
        print(f"   Average confidence: {avg_score:.3f}")
        print(f"   Best candidate score: {candidates[0]['final_score']:.3f}")
        
        # Show top 3 candidates
        print("\n3. Top candidates:")
        for i, candidate in enumerate(candidates[:3], 1):
            print(f"\n   {i}. {candidate['name']}")
            print(f"      {candidate['description']}")
            print(f"      Strategy: {candidate['strategy']}")
            print(f"      Score: {candidate['final_score']:.3f}")
            print(f"      Properties: {', '.join(candidate['properties'])}")
            
            # Show code preview
            code_lines = candidate['code'].split('\n')
            print(f"      Code preview (first 4 lines):")
            for line in code_lines[:4]:
                print(f"         {line}")
            if len(code_lines) > 4:
                print(f"         ... ({len(code_lines)-4} more lines)")
        
        return True
        
    except Exception as e:
        print(f" Donor generation failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_code_execution():
    """Test that generated donor code actually executes."""
    print("\n  TESTING CODE EXECUTION")
    print("=" * 50)
    
    func = find_max_in_range
    
    try:
        # Generate candidates
        candidates = integrate_metta_generation(func)
        
        if not candidates:
            print(" No candidates to test")
            return False
        
        print(f"1. Testing execution of {len(candidates)} candidates...")
        
        # Test data
        test_data = [1, 5, 3, 9, 2, 7, 4]
        test_cases = [
            (test_data, 1, 4),  # Should work
            (test_data, 0, 3),  # Should work  
            (test_data, -1, 2), # Should handle bounds
            (test_data, 5, 10), # Should handle bounds
        ]
        
        execution_results = []
        
        for i, candidate in enumerate(candidates):
            print(f"\n   Testing candidate {i+1}: {candidate['name']}")
            
            try:
                # Execute the generated code
                exec_namespace = {}
                exec(candidate['code'], exec_namespace)
                
                # Get the function from the namespace
                func_name = candidate['name']
                if func_name in exec_namespace:
                    generated_func = exec_namespace[func_name]
                    
                    # Test with different inputs
                    test_results = []
                    for test_input in test_cases:
                        try:
                            result = generated_func(*test_input)
                            test_results.append(f" {test_input} → {result}")
                        except Exception as e:
                            test_results.append(f" {test_input} → Error: {e}")
                    
                    execution_results.append({
                        'name': func_name,
                        'success': True,
                        'results': test_results
                    })
                    
                    print(f"       Execution successful")
                    for result in test_results[:2]:  # Show first 2 test results
                        print(f"         {result}")
                
                else:
                    execution_results.append({
                        'name': func_name,
                        'success': False,
                        'error': f"Function {func_name} not found in executed code"
                    })
                    print(f"       Function not found after execution")
                    
            except Exception as e:
                execution_results.append({
                    'name': candidate['name'],
                    'success': False,
                    'error': str(e)
                })
                print(f"       Execution failed: {e}")
        
        # Summary
        successful = sum(1 for r in execution_results if r['success'])
        print(f"\n2. Execution Summary:")
        print(f"    {successful}/{len(execution_results)} candidates executed successfully")
        
        return successful > 0
        
    except Exception as e:
        print(f" Code execution testing failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def run_comprehensive_test():
    """Run all tests in sequence."""
    print("COMPREHENSIVE METTA DONOR GENERATOR TEST")
    print("=" * 60)
    
    test_results = []
    
    # Run all test phases
    tests = [
        ("Static Analysis", test_static_analysis),
        ("MeTTa Integration", test_metta_integration),
        ("Pattern Detection", test_pattern_detection),
        ("Donor Generation", test_donor_generation),
        ("Code Execution", test_code_execution),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name.upper()} {'='*20}")
        try:
            result = test_func()
            test_results.append((test_name, result))
            
            if result:
                print(f"\n {test_name} PASSED")
            else:
                print(f"\n {test_name} FAILED")
                
        except Exception as e:
            print(f"\n {test_name} CRASHED: {e}")
            test_results.append((test_name, False))
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = " PASS" if result else " FAIL"
        print(f"   {test_name:<20} {status}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n  Overall Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    
    if passed == total:
        print("ALL TESTS PASSED! The MeTTa donor generator is working correctly.")
    elif passed > total // 2:
        print("Most tests passed, but some issues need attention.")
    else:
        print(" Multiple test failures - significant issues need to be resolved.")
    
    return success_rate >= 80  # Consider 80%+ a success

def quick_demo():
    """Quick demonstration of the system working."""
    print("\nQUICK DEMO")
    print("=" * 30)
    
    func = find_max_in_range
    print(f"Input: {func.__name__}")
    
    try:
        candidates = integrate_metta_generation(func)
        
        if candidates:
            best = candidates[0]
            print(f"Best candidate: {best['name']}")
            print(f"  Confidence: {best['final_score']:.3f}")
            print(f"  Strategy: {best['strategy']}")
            
            # Test the code
            exec_namespace = {}
            exec(best['code'], exec_namespace)
            
            if best['name'] in exec_namespace:
                generated_func = exec_namespace[best['name']]
                test_result = generated_func([1, 5, 3, 9, 2], 1, 4)
                print(f"  Test run: {test_result}")
                print(" Demo successful!")
            else:
                print(" Generated function not executable")
        else:
            print(" No candidates generated")
            
    except Exception as e:
        print(f" Demo failed: {e}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            quick_demo()
        elif sys.argv[1] == "--demo":
            quick_demo()
        else:
            print("Usage: python end_to_end_test.py [--quick|--demo]")
    else:
        # Run comprehensive test
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)