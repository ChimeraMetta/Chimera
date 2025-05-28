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


def test_function():
    """Our test function for donor generation."""
    def find_max_in_range(numbers, start_idx, end_idx):
        """Find the maximum value in a list within a specific range."""
        if start_idx < 0 or end_idx > len(numbers) or start_idx >= end_idx:
            return None
        
        max_val = numbers[start_idx]
        for i in range(start_idx + 1, end_idx):
            if numbers[i] > max_val:
                max_val = numbers[i]
        
        return max_val
    
    return find_max_in_range

def test_static_analysis():
    """Test the static analysis pipeline."""
    print("🔍 TESTING STATIC ANALYSIS PIPELINE")
    print("=" * 50)
    
    func = test_function()
    
    try:
        # Test decompose_function
        print("1. Testing decompose_function()...")
        result = decompose_function(func)
        
        if "error" in result:
            print(f"❌ Error in decompose_function: {result['error']}")
            return False
        
        print(f"✅ decompose_function succeeded")
        print(f"   📊 Generated {len(result.get('metta_atoms', []))} MeTTa atoms")
        print(f"   📦 Found {len(result.get('structure', []))} structural elements")
        print(f"   🔗 Detected {len(result.get('function_calls', {}))} function call patterns")
        print(f"   📝 Mapped {len(result.get('variables', {}))} variable scopes")
        
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
        print(f"❌ Static analysis failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_metta_integration():
    """Test MeTTa space integration."""
    print("\n🧠 TESTING METTA INTEGRATION")
    print("=" * 50)
    
    func = test_function()
    
    try:
        # Test MeTTa atom loading
        print("1. Testing MeTTa atom loading...")
        
        # Get analysis result
        result = decompose_function(func)
        atoms = result.get('metta_atoms', [])
        
        print(f"   📥 Loading {len(atoms)} atoms into MeTTa space...")
        
        # Load atoms
        loaded_count = 0
        failed_count = 0
        
        for atom in atoms:
            if monitor.add_atom(atom):
                loaded_count += 1
            else:
                failed_count += 1
        
        print(f"   ✅ Loaded {loaded_count}/{len(atoms)} atoms successfully")
        if failed_count > 0:
            print(f"   ⚠️  {failed_count} atoms failed to load")
        
        # Test querying
        print("\n2. Testing MeTTa querying...")
        
        # Try some basic queries
        test_queries = [
            "function-def",
            "bin-op",
            "loop-pattern",
            "function-return"
        ]
        
        for query_type in test_queries:
            try:
                # Simple existence check
                atoms_str = str(monitor.metta_space)
                has_evidence = query_type in atoms_str
                print(f"   🔍 {query_type}: {'✅ Found' if has_evidence else '❌ Not found'}")
            except Exception as e:
                print(f"   ⚠️  Query for {query_type} failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ MeTTa integration failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_pattern_detection():
    """Test pattern detection capabilities."""
    print("\n🔍 TESTING PATTERN DETECTION")
    print("=" * 50)
    
    func = test_function()
    
    try:
        # Create generator and load function
        generator = MettaDonorGenerator()
        
        # Extract source and analyze
        import inspect
        source_code = inspect.getsource(func)
        
        # Parse and analyze
        import ast
        tree = ast.parse(source_code)
        decomposer = CodeDecomposer()
        decomposer.visit(tree)
        
        # Load atoms
        atoms = convert_to_metta_atoms(decomposer)
        generator._load_atoms_to_metta(atoms)
        generator.function_name = func.__name__
        generator.original_code = source_code
        
        # Test pattern detection
        print("1. Testing pattern detection...")
        patterns = generator._detect_patterns_with_metta()
        
        print(f"   🎯 Detected {len(patterns)} patterns:")
        for i, pattern in enumerate(patterns, 1):
            print(f"     {i}. {pattern.pattern_type} (confidence: {pattern.confidence:.2f})")
            print(f"        Properties: {', '.join(pattern.properties)}")
        
        # Test strategy applicability
        print("\n2. Testing strategy applicability...")
        strategies = generator._get_applicable_strategies_from_metta(None)
        
        print(f"   🎯 Found {len(strategies)} applicable strategies:")
        for i, strategy in enumerate(strategies, 1):
            print(f"     {i}. {strategy}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pattern detection failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_donor_generation():
    """Test the complete donor generation pipeline."""
    print("\n⚙️  TESTING DONOR GENERATION")
    print("=" * 50)
    
    func = test_function()
    
    try:
        print("1. Running complete donor generation pipeline...")
        
        # Use the integration function
        candidates = integrate_metta_generation(func)
        
        print(f"   ✅ Generated {len(candidates)} donor candidates")
        
        if not candidates:
            print("   ⚠️  No candidates generated - checking for issues...")
            return False
        
        # Analyze candidates
        print("\n2. Analyzing generated candidates...")
        
        strategies_used = set()
        total_score = 0
        
        for candidate in candidates:
            strategies_used.add(candidate['strategy'])
            total_score += candidate['final_score']
        
        avg_score = total_score / len(candidates) if candidates else 0
        
        print(f"   📊 Strategies used: {', '.join(strategies_used)}")
        print(f"   📈 Average confidence: {avg_score:.3f}")
        print(f"   🏆 Best candidate score: {candidates[0]['final_score']:.3f}")
        
        # Show top 3 candidates
        print("\n3. Top candidates:")
        for i, candidate in enumerate(candidates[:3], 1):
            print(f"\n   {i}. {candidate['name']}")
            print(f"      📝 {candidate['description']}")
            print(f"      🎯 Strategy: {candidate['strategy']}")
            print(f"      📊 Score: {candidate['final_score']:.3f}")
            print(f"      🏷️  Properties: {', '.join(candidate['properties'])}")
            
            # Show code preview
            code_lines = candidate['code'].split('\n')
            print(f"      📄 Code preview (first 4 lines):")
            for line in code_lines[:4]:
                print(f"         {line}")
            if len(code_lines) > 4:
                print(f"         ... ({len(code_lines)-4} more lines)")
        
        return True
        
    except Exception as e:
        print(f"❌ Donor generation failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_code_execution():
    """Test that generated donor code actually executes."""
    print("\n🚀 TESTING CODE EXECUTION")
    print("=" * 50)
    
    func = test_function()
    
    try:
        # Generate candidates
        candidates = integrate_metta_generation(func)
        
        if not candidates:
            print("❌ No candidates to test")
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
                            test_results.append(f"✅ {test_input} → {result}")
                        except Exception as e:
                            test_results.append(f"❌ {test_input} → Error: {e}")
                    
                    execution_results.append({
                        'name': func_name,
                        'success': True,
                        'results': test_results
                    })
                    
                    print(f"      ✅ Execution successful")
                    for result in test_results[:2]:  # Show first 2 test results
                        print(f"         {result}")
                
                else:
                    execution_results.append({
                        'name': func_name,
                        'success': False,
                        'error': f"Function {func_name} not found in executed code"
                    })
                    print(f"      ❌ Function not found after execution")
                    
            except Exception as e:
                execution_results.append({
                    'name': candidate['name'],
                    'success': False,
                    'error': str(e)
                })
                print(f"      ❌ Execution failed: {e}")
        
        # Summary
        successful = sum(1 for r in execution_results if r['success'])
        print(f"\n2. Execution Summary:")
        print(f"   ✅ {successful}/{len(execution_results)} candidates executed successfully")
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ Code execution testing failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def run_comprehensive_test():
    """Run all tests in sequence."""
    print("🧪 COMPREHENSIVE METTA DONOR GENERATOR TEST")
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
                print(f"\n✅ {test_name} PASSED")
            else:
                print(f"\n❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"\n💥 {test_name} CRASHED: {e}")
            test_results.append((test_name, False))
    
    # Final summary
    print("\n" + "="*60)
    print("🏁 FINAL TEST RESULTS")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<20} {status}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n📊 Overall Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The MeTTa donor generator is working correctly.")
    elif passed > total // 2:
        print("⚠️  Most tests passed, but some issues need attention.")
    else:
        print("❌ Multiple test failures - significant issues need to be resolved.")
    
    return success_rate >= 80  # Consider 80%+ a success

def quick_demo():
    """Quick demonstration of the system working."""
    print("\n🚀 QUICK DEMO")
    print("=" * 30)
    
    func = test_function()
    print(f"📝 Input: {func.__name__}")
    
    try:
        candidates = integrate_metta_generation(func)
        
        if candidates:
            best = candidates[0]
            print(f"🏆 Best candidate: {best['name']}")
            print(f"📊 Confidence: {best['final_score']:.3f}")
            print(f"🎯 Strategy: {best['strategy']}")
            
            # Test the code
            exec_namespace = {}
            exec(best['code'], exec_namespace)
            
            if best['name'] in exec_namespace:
                generated_func = exec_namespace[best['name']]
                test_result = generated_func([1, 5, 3, 9, 2], 1, 4)
                print(f"🚀 Test run: {test_result}")
                print("✅ Demo successful!")
            else:
                print("❌ Generated function not executable")
        else:
            print("❌ No candidates generated")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")

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