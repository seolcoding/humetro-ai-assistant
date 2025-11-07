#!/usr/bin/env python3
"""
End-to-End Knowledge Graph Builder for Seoul Traffic Data

This script orchestrates the complete KG building pipeline:
1. User Intent Agent - defines the KG goal
2. NER Agent - identifies entity types
3. Fact Extraction Agent - identifies relationships
4. KG Construction - builds the graph from markdown files
"""

import asyncio
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.kg_agent.agents.user_intent_agent import create_user_intent_agent, APPROVED_USER_GOAL
from src.kg_agent.agents.ner_agent import create_ner_agent, APPROVED_ENTITIES
from src.kg_agent.agents.fact_extraction_agent import create_fact_extraction_agent, APPROVED_FACTS
from src.kg_agent.kg_construction import construct_kg_from_files
from src.kg_agent.utils.agent_caller import make_agent_caller
from src.kg_agent.utils.tools import get_data_import_dir
from src.kg_agent.config.llm_config import DEFAULT_KG_MODEL

logger = logging.getLogger(__name__)


async def build_seoul_traffic_kg(
    num_files: int = 10,
    model_name: str = DEFAULT_KG_MODEL,
    user_goal_description: str = None,
    verbose: bool = False
):
    """
    Build Knowledge Graph for Seoul Traffic Information

    Args:
        num_files: Number of markdown files to process
        model_name: LLM model to use (default from config)
        user_goal_description: Custom user goal description (optional)

    Returns:
        Dict with pipeline results
    """
    print("=" * 70)
    print("Seoul Traffic Knowledge Graph Builder")
    print("=" * 70)

    results = {}

    # ========================================================================
    # Stage 1: User Intent
    # ========================================================================
    print("\n[Stage 1/4] User Intent Agent")
    print("-" * 70)

    user_intent_agent = create_user_intent_agent(model_name)
    user_intent_caller = await make_agent_caller(user_intent_agent)

    # Use custom description or default
    if user_goal_description is None:
        user_goal_description = """I want to build a knowledge graph for Seoul metropolitan traffic information.
        The graph should include subway, bus, and public transportation data to support
        information retrieval and question answering about Seoul's transportation system."""

    print(f"User goal: {user_goal_description[:100]}...")

    await user_intent_caller.call(user_goal_description, verbose=verbose)
    session = await user_intent_caller.get_session()

    # Auto-approve - try multiple times if needed
    max_attempts = 3
    for attempt in range(max_attempts):
        if APPROVED_USER_GOAL in session.state:
            break

        if "perceived_user_goal" in session.state:
            await user_intent_caller.call("Yes, that's correct. Please approve it.", verbose=verbose)
            session = await user_intent_caller.get_session()
        else:
            # Agent is asking for clarification - provide confirmation
            await user_intent_caller.call("Yes, that's exactly what I want. Please set and approve the perceived goal.", verbose=verbose)
            session = await user_intent_caller.get_session()

    if APPROVED_USER_GOAL not in session.state:
        raise RuntimeError("Failed to get approved user goal")

    approved_user_goal = session.state[APPROVED_USER_GOAL]
    results['user_goal'] = approved_user_goal
    print(f"✅ Goal approved: {approved_user_goal['kind_of_graph']}")

    # ========================================================================
    # Stage 2: File Selection
    # ========================================================================
    print("\n[Stage 2/4] File Selection")
    print("-" * 70)

    import_dir = get_data_import_dir()
    all_md_files = sorted([str(f.relative_to(import_dir)) for f in import_dir.rglob("*.md")])

    if num_files > 0:
        approved_files = all_md_files[:num_files]
    else:
        approved_files = all_md_files

    results['approved_files'] = approved_files
    print(f"✅ Selected {len(approved_files)} files for processing")
    print(f"   Sample files: {approved_files[:3]}")

    # ========================================================================
    # Stage 3: Named Entity Recognition
    # ========================================================================
    print("\n[Stage 3/4] Named Entity Recognition")
    print("-" * 70)

    # Prepare initial state with construction plan (well-known entities)
    ner_initial_state = {
        "approved_user_goal": approved_user_goal,
        "approved_files": approved_files,
        "approved_construction_plan": {
            "Station": {
                "construction_type": "node",
                "label": "Station",
            },
            "Line": {
                "construction_type": "node",
                "label": "Line",
            },
            "Route": {
                "construction_type": "node",
                "label": "Route",
            }
        }
    }

    ner_agent = create_ner_agent(model_name)
    ner_caller = await make_agent_caller(ner_agent, ner_initial_state)

    # Give clear step-by-step instructions to the agent
    ner_message = """
    Please analyze the approved files and propose entity types for Seoul traffic information.

    Follow these steps:
    1. Use get_approved_user_goal to understand the goal
    2. Use get_approved_files to get the file list
    3. Use get_well_known_types to see existing entity types
    4. Use sample_file to examine 2-3 files from the approved files list
    5. Based on the content, use set_proposed_entities with a list of relevant entity types
    6. Present the proposed entities for approval

    The entity types should cover transportation-related entities like stations, routes, lines, services, etc.
    """

    await ner_caller.call(ner_message, verbose=verbose)
    session = await ner_caller.get_session()

    # Auto-approve if proposed
    if "proposed_entity_types" in session.state:
        await ner_caller.call("Yes, approve the proposed entities.", verbose=verbose)
        session = await ner_caller.get_session()

    if APPROVED_ENTITIES not in session.state:
        logger.error(f"Session state keys: {list(session.state.keys())}")
        logger.error(f"Proposed entities in state: {'proposed_entity_types' in session.state}")
        raise RuntimeError("Failed to get approved entities")

    approved_entities = session.state[APPROVED_ENTITIES]
    results['approved_entities'] = approved_entities
    print(f"✅ Approved {len(approved_entities)} entity types:")
    print(f"   {', '.join(approved_entities)}")

    # ========================================================================
    # Stage 4: Fact Extraction
    # ========================================================================
    print("\n[Stage 4/4] Fact Extraction")
    print("-" * 70)

    fact_initial_state = {
        "approved_user_goal": approved_user_goal,
        "approved_files": approved_files,
        "approved_entity_types": approved_entities
    }

    fact_agent = create_fact_extraction_agent(model_name)
    fact_caller = await make_agent_caller(fact_agent, fact_initial_state)

    # Give clear instructions with available entity types
    fact_message = f"""
    Please propose relationship types (fact triples) between the approved entity types for Seoul traffic data.

    The approved entity types are: {', '.join(approved_entities)}

    Follow these steps:
    1. Use get_approved_user_goal to understand the context
    2. Consider what relationships exist between these entity types
    3. For each relationship, use add_proposed_fact with:
       - subject_label: one of the approved entity types
       - predicate_label: a descriptive relationship name (e.g., "operates_on", "connects_to")
       - object_label: another approved entity type
    4. Use get_proposed_facts to review all proposed relationships
    5. Present the relationships for approval

    Example relationships for transportation:
    - (TransportationMode) --[operates_on]--> (Line)
    - (Station) --[is_part_of]--> (Line)
    - (Route) --[connects]--> (Station)
    """

    await fact_caller.call(fact_message, verbose=verbose)
    session = await fact_caller.get_session()

    # Auto-approve if proposed
    if "proposed_fact_types" in session.state:
        await fact_caller.call("Yes, approve all the proposed fact types.", verbose=verbose)
        session = await fact_caller.get_session()

    if APPROVED_FACTS not in session.state:
        logger.error(f"Session state keys: {list(session.state.keys())}")
        logger.error(f"Proposed facts in state: {'proposed_fact_types' in session.state}")
        raise RuntimeError("Failed to get approved facts")

    approved_fact_types = session.state[APPROVED_FACTS]
    results['approved_fact_types'] = approved_fact_types
    print(f"✅ Approved {len(approved_fact_types)} relationship types:")
    for predicate, fact in list(approved_fact_types.items())[:5]:
        print(f"   ({fact['subject_label']}) --[{fact['predicate_label']}]--> ({fact['object_label']})")
    if len(approved_fact_types) > 5:
        print(f"   ... and {len(approved_fact_types) - 5} more")

    # ========================================================================
    # Stage 5: Knowledge Graph Construction
    # ========================================================================
    print("\n[Stage 5/5] Knowledge Graph Construction")
    print("-" * 70)
    print(f"Processing {len(approved_files)} markdown files...")

    construction_results = await construct_kg_from_files(
        approved_files,
        approved_entities,
        approved_fact_types,
        model_name
    )

    results['construction'] = construction_results
    print(f"\n✅ Construction complete!")
    print(f"   Total files: {construction_results['total_files']}")
    print(f"   Processed: {construction_results['processed']}")
    print(f"   Failed: {construction_results['failed']}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print(f"✅ User Goal: {approved_user_goal['kind_of_graph']}")
    print(f"✅ Entity Types: {len(approved_entities)}")
    print(f"✅ Relationship Types: {len(approved_fact_types)}")
    print(f"✅ Files Processed: {construction_results['processed']}/{construction_results['total_files']}")
    print("=" * 70)

    return results


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build Knowledge Graph for Seoul Traffic Information"
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=10,
        help="Number of markdown files to process (default: 10, 0 = all)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_KG_MODEL,
        help=f"LLM model to use (default: {DEFAULT_KG_MODEL})"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # Run pipeline
    try:
        results = asyncio.run(build_seoul_traffic_kg(
            num_files=args.num_files,
            model_name=args.model,
            verbose=args.verbose
        ))
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
