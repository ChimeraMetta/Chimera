import os
import logging
import json
from typing import Dict, List, Any

# Import the example-driven proof generator
from proofs.example_generator import ExampleDrivenProofGenerator

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("proof_demo")

def run_proof_demo(api_key: str):
    """
    Run a demonstration of the example-driven proof generation.
    
    Args:
        api_key: OpenAI API key for LLM access
    """
    # Create the proof generator
    generator = ExampleDrivenProofGenerator(model_name="gpt-4o-mini", api_key=api_key)
    
    # Test functions of different types
    test_functions = [
        {
            "name": "binary_search",
            "code": """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
        },
        {
            "name": "merge_sort",
            "code": """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
        
    # Divide array into two halves
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Merge the two sorted halves
    return merge(left, right)
    
def merge(left, right):
    result = []
    i = j = 0
    
    # Compare elements from both arrays
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"""
        },
        {
            "name": "graph_bfs",
            "code": """
def bfs(graph, start):
    visited = set([start])
    queue = [start]
    result = []
    
    while queue:
        vertex = queue.pop(0)
        result.append(vertex)
        
        # Visit adjacent nodes
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                
    return result
"""
        },
        {
            "name": "data_processor",
            "code": """
def process_data(data_list, threshold):
    results = []
    error_count = 0
    
    for item in data_list:
        try:
            if item['value'] > threshold:
                processed = transform_item(item)
                if processed is not None:
                    results.append(processed)
        except KeyError:
            error_count += 1
            continue
            
    return results, error_count
"""
        }
    ]
    
    # Process each test function
    results = {}
    for func in test_functions:
        logger.info(f"Generating proof for: {func['name']}")
        
        try:
            # Generate proof
            proof_result = generator.generate_proof(
                function_code=func["code"],
                function_name=func["name"]
            )
            
            # Store result
            success = proof_result.get("success", False)
            results[func["name"]] = {
                "success": success,
                "component_count": len(proof_result.get("proof", [])),
                "is_fallback": proof_result.get("fallback", False),
                "error": proof_result.get("error", None)
            }
            
            # Save the detailed JSON IR
            output_dir = "proof_outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            with open(f"{output_dir}/{func['name']}_proof.json", "w") as f:
                json.dump(proof_result.get("json_ir", {}), f, indent=2)
                
            # Save MeTTa atoms
            with open(f"{output_dir}/{func['name']}_metta.txt", "w") as f:
                for atom in proof_result.get("proof", []):
                    f.write(atom + "\n")
            
            logger.info(f"Proof generation for {func['name']}: {'SUCCESS' if success else 'FAILED'}")
            if not success:
                logger.error(f"Error: {proof_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error processing {func['name']}: {str(e)}")
            results[func["name"]] = {
                "success": False,
                "error": str(e)
            }
    
    # Report summary
    logger.info("\n======= PROOF GENERATION SUMMARY =======")
    success_count = sum(1 for r in results.values() if r["success"])
    logger.info(f"Total functions: {len(test_functions)}")
    logger.info(f"Successful proofs: {success_count}/{len(test_functions)} ({success_count/len(test_functions)*100:.1f}%)")
    logger.info(f"Fallback proofs: {sum(1 for r in results.values() if r.get('is_fallback', False))}")
    
    for name, result in results.items():
        status = "SUCCESS" if result["success"] else "FAILED"
        fallback = " (FALLBACK)" if result.get("is_fallback", False) else ""
        components = result.get("component_count", 0)
        logger.info(f"  - {name}: {status}{fallback} - {components} components")
    
    return results

if __name__ == "__main__":
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
        
    run_proof_demo(api_key)