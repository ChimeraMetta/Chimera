"""Executor for the cybersecurity query command.

Follows the executors/full_analyzer.py pattern: initializes MeTTa space,
loads the cybersecurity ontology, and orchestrates query execution.
"""

import os
from common.logging_utils import get_logger
from reflectors.dynamic_monitor import DynamicMonitor

logger = get_logger("query_executor")

CYBERSECURITY_ONTOLOGY_PATH = "metta/cybersecurity_ontology.metta"


def run_query(query_text, workspace_root, interactive=False, show_metrics=False):
    """Execute a cybersecurity NL query.

    Args:
        query_text: The natural language query string.
        workspace_root: Project root directory.
        interactive: If True, prompt for rating after query.
        show_metrics: If True, display accuracy metrics.
    """
    # Initialize MeTTa space and load ontology
    logger.info("Initializing MeTTa space for cybersecurity query...")
    metta_instance = None
    metta_space = None

    try:
        local_monitor = DynamicMonitor()
        ontology_path = os.path.join(workspace_root, CYBERSECURITY_ONTOLOGY_PATH)

        if os.path.exists(ontology_path):
            local_monitor.load_metta_rules(ontology_path)
            logger.info(f"Loaded cybersecurity ontology from {ontology_path}")
        else:
            logger.warning(f"Cybersecurity ontology not found at {ontology_path}. Using fallback mode.")

        metta_instance = local_monitor.metta
        metta_space = local_monitor.metta_space
    except Exception as e:
        logger.warning(f"MeTTa initialization failed: {e}. Using Python fallback.")

    # Create query engine
    from cybersecurity_query.engine import CyberSecurityQueryEngine
    engine = CyberSecurityQueryEngine(
        metta_instance=metta_instance,
        metta_space=metta_space,
    )

    # Execute query
    logger.info(f"Processing query: \"{query_text}\"")
    result = engine.query(query_text)

    # Display formatted response
    print(result.formatted_response)

    # Interactive rating
    if interactive:
        logger.info("Interactive mode: collecting rating...")
        rating = engine.rate_response(result)
        if rating:
            logger.info(f"Rating recorded: {rating.rating}/5")
        else:
            logger.info("No rating provided.")

    # Show metrics
    if show_metrics:
        metrics = engine.get_metrics()
        print("\n" + "=" * 60)
        print("  Query System Metrics")
        print("=" * 60)
        print(f"  Total queries rated: {metrics.rated_queries}")
        print(f"  Average rating: {metrics.average_rating:.2f}/5")
        print(f"  Intent accuracy: {metrics.intent_accuracy:.0%}")
        print(f"  Translation accuracy: {metrics.translation_accuracy:.0%}")
        print(f"  Retrieval accuracy: {metrics.retrieval_accuracy:.0%}")
        print(f"  Low-rated queries: {metrics.low_rated_queries}")

    return result
