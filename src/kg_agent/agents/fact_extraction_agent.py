#!/usr/bin/env python3
"""
Fact Extraction Agent for Unstructured Data
Based on agentic_kg_build/6_schema_proposal_unstructured.ipynb

Proposes fact types (relationships) that can be extracted from unstructured text files.
"""

from google.adk.agents import Agent
from google.adk.tools import ToolContext
from typing import Dict, Any
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.kg_agent.config.llm_config import get_llm, DEFAULT_KG_MODEL
from src.kg_agent.utils.tools import (
    get_approved_user_goal,
    get_data_import_dir,
    tool_success,
    tool_error
)
import logging

logger = logging.getLogger(__name__)

# State keys
APPROVED_ENTITIES = "approved_entity_types"
PROPOSED_FACTS = "proposed_fact_types"
APPROVED_FACTS = "approved_fact_types"


# ============================================================================
# Agent Instructions (from notebook)
# ============================================================================

fact_agent_role_and_goal = """
  You are a top-tier algorithm designed for analyzing text files and proposing
  the type of facts that could be extracted from text that would be relevant
  for a user's goal.
"""

fact_agent_hints = """
  Do not propose specific individual facts, but instead propose the general type
  of facts that would be relevant for the user's goal.
  For example, do not propose "ABK likes coffee" but the general type of fact "Person likes Beverage".

  Facts are triplets of (subject, predicate, object) where the subject and object are
  approved entity types, and the proposed predicate provides information about
  how they are related. For example, a fact type could be (Person, likes, Beverage).

  Design rules for facts:
  - only use approved entity types as subjects or objects. Do not propose new types of entities
  - the proposed predicate should describe the relationship between the approved subject and object
  - the predicate should optimize for information that is relevant to the user's goal
  - the predicate must appear in the source text. Do not guess.
  - use the 'add_proposed_fact' tool to record each proposed fact type
"""

fact_agent_chain_of_thought_directions = """
    Prepare for the task:
    - use the 'get_approved_user_goal' tool to get the user goal
    - use the 'get_approved_files' tool to get the list of approved files
    - use the 'get_approved_entities' tool to get the list of approved entity types

    Think step by step:
    1. Use the 'get_approved_user_goal' tool to get the user goal
    2. Sample some of the approved files using the 'sample_file' tool to understand the content
    3. Consider how subjects and objects are related in the text
    4. Call the 'add_proposed_fact' tool for each type of fact you propose
    5. Use the 'get_proposed_facts' tool to retrieve all the proposed facts
    6. Present the proposed types of facts to the user, along with an explanation
"""

fact_agent_instruction = f"""
{fact_agent_role_and_goal}
{fact_agent_hints}
{fact_agent_chain_of_thought_directions}
"""


# ============================================================================
# Tool Definitions (from notebook)
# ============================================================================

def get_approved_files(tool_context: ToolContext) -> Dict[str, Any]:
    """Get the approved list of files to process."""
    approved_files = tool_context.state.get("approved_files", [])
    return tool_success("approved_files", approved_files)


def get_approved_entities(tool_context: ToolContext) -> dict:
    """Get the approved list of entity types to extract from unstructured text."""
    approved_entities = tool_context.state.get(APPROVED_ENTITIES, [])
    return tool_success(APPROVED_ENTITIES, approved_entities)


def sample_file(file_path: str, tool_context: ToolContext = None) -> dict:
    """Samples a file by reading its content as text.

    Treats any file as text and reads up to a maximum of 100 lines.

    Args:
      file_path: file to sample, relative to the import directory

    Returns:
        dict: A dictionary containing metadata about the content.
    """
    from itertools import islice

    # Trust, but verify. The agent may invent absolute file paths.
    if Path(file_path).is_absolute():
        return tool_error("File path must be relative to the import directory. Make sure the file is from the list of approved files.")

    import_dir = Path(get_data_import_dir())

    # create the full path by extending from the import_dir
    full_path_to_file = import_dir / file_path

    # of course, _that_ may not exist
    if not full_path_to_file.exists():
        return tool_error(f"File does not exist in import directory. Make sure {file_path} is from the list of approved files.")

    try:
        # Treat all files as text
        with open(full_path_to_file, 'r', encoding='utf-8') as file:
            # Read up to 100 lines
            lines = list(islice(file, 100))
            content = ''.join(lines)
            return tool_success("content", content)

    except Exception as e:
        return tool_error(f"Error reading or processing file {file_path}: {e}")


def add_proposed_fact(approved_subject_label: str,
                      proposed_predicate_label: str,
                      approved_object_label: str,
                      tool_context: ToolContext) -> dict:
    """Add a proposed type of fact that could be extracted from the files.

    A proposed fact type is a tuple of (subject, predicate, object) where
    the subject and object are approved entity types and the predicate
    is a proposed relationship label.

    Args:
      approved_subject_label: approved label of the subject entity type
      proposed_predicate_label: label of the predicate
      approved_object_label: approved label of the object entity type
    """
    # Guard against invalid labels
    approved_entities = tool_context.state.get(APPROVED_ENTITIES, [])

    if approved_subject_label not in approved_entities:
        return tool_error(f"Approved subject label {approved_subject_label} not found. Try again.")
    if approved_object_label not in approved_entities:
        return tool_error(f"Approved object label {approved_object_label} not found. Try again.")

    current_predicates = tool_context.state.get(PROPOSED_FACTS, {})
    current_predicates[proposed_predicate_label] = {
        "subject_label": approved_subject_label,
        "predicate_label": proposed_predicate_label,
        "object_label": approved_object_label
    }
    tool_context.state[PROPOSED_FACTS] = current_predicates
    return tool_success(PROPOSED_FACTS, current_predicates)


def get_proposed_facts(tool_context: ToolContext) -> dict:
    """Get the proposed types of facts that could be extracted from the files."""
    proposed_facts = tool_context.state.get(PROPOSED_FACTS, {})
    return tool_success(PROPOSED_FACTS, proposed_facts)


def approve_proposed_facts(tool_context: ToolContext) -> dict:
    """Upon user approval, records the proposed fact types as approved fact types

    Only call this tool if the user has explicitly approved the proposed fact types.
    """
    if PROPOSED_FACTS not in tool_context.state:
        return tool_error("No proposed fact types to approve. Please set proposed facts first, ask for user approval, then call this tool.")
    tool_context.state[APPROVED_FACTS] = tool_context.state.get(PROPOSED_FACTS)
    return tool_success(APPROVED_FACTS, tool_context.state[APPROVED_FACTS])


# Tool list
fact_agent_tools = [
    get_approved_user_goal,
    get_approved_files,
    get_approved_entities,
    sample_file,
    add_proposed_fact,
    get_proposed_facts,
    approve_proposed_facts
]


# ============================================================================
# Agent Factory Function
# ============================================================================

def create_fact_extraction_agent(model_name: str = DEFAULT_KG_MODEL) -> Agent:
    """
    Create and return a Fact Extraction Agent

    Args:
        model_name: LLM model to use (default from config)

    Returns:
        Agent instance configured for fact extraction
    """
    llm = get_llm(model_name)

    agent = Agent(
        name="fact_type_extraction_agent_v1",
        model=llm,
        description="Proposes the kind of relevant facts that could be extracted from text files.",
        instruction=fact_agent_instruction,
        tools=fact_agent_tools,
    )

    logger.info(f"✅ Agent '{agent.name}' created with model '{model_name}'")
    return agent


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from src.kg_agent.utils.agent_caller import make_agent_caller

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    async def test_fact_extraction_agent():
        """Test the Fact Extraction agent with Seoul traffic data"""

        print("=" * 60)
        print("Fact Extraction Agent Test")
        print("=" * 60)

        # Create agent
        print("\n1. Creating agent...")
        agent = create_fact_extraction_agent()

        # Prepare initial state with approved entities from NER agent
        print("2. Preparing initial state with approved entities...")

        # Get first 5 markdown files
        import_dir = get_data_import_dir()
        md_files = sorted([str(f.relative_to(import_dir)) for f in import_dir.rglob("*.md")])[:5]

        initial_state = {
            "approved_user_goal": {
                "kind_of_graph": "Seoul Traffic Information",
                "graph_description": "A knowledge graph for Seoul metropolitan traffic information including subway, bus, and other public transportation data to support information retrieval and question answering."
            },
            "approved_files": md_files,
            # These would come from NER agent output
            "approved_entity_types": [
                'Station', 'Line', 'Route',
                'TransportationMode', 'Service', 'Event',
                'Landmark', 'Facility', 'SafetyMeasure', 'Schedule'
            ]
        }

        print(f"   Using {len(md_files)} markdown files")
        print(f"   With {len(initial_state['approved_entity_types'])} approved entities")

        # Create agent caller
        print("\n3. Creating agent caller...")
        caller = await make_agent_caller(agent, initial_state)

        # Check initial session state
        session = await caller.get_session()
        print(f"   Initial state has approved_entity_types: {APPROVED_ENTITIES in session.state}")

        # Start conversation
        print("\n4. Starting fact extraction analysis...")
        response1 = await caller.call(
            "Propose fact types that can be found in the text.",
            verbose=True
        )
        print(f"\n>>> Agent Response 1:\n{response1}\n")

        # Check proposed facts
        session = await caller.get_session()
        if PROPOSED_FACTS in session.state:
            print(f"\nProposed facts count: {len(session.state[PROPOSED_FACTS])}")
            print("Sample proposed facts:")
            for i, (predicate, fact) in enumerate(list(session.state[PROPOSED_FACTS].items())[:3]):
                print(f"  {i+1}. ({fact['subject_label']}) --[{fact['predicate_label']}]--> ({fact['object_label']})")

            # Approve them
            print("\n5. Approving proposed facts...")
            response2 = await caller.call("Approve the proposed fact types.", verbose=True)
            print(f"\n>>> Agent Response 2:\n{response2}\n")

        # Check final session state
        session = await caller.get_session()
        print("\n" + "=" * 60)
        print("Final Session State:")
        print("=" * 60)
        if PROPOSED_FACTS in session.state:
            print(f"Proposed facts count: {len(session.state[PROPOSED_FACTS])}")
        if APPROVED_FACTS in session.state:
            print(f"Approved facts count: {len(session.state[APPROVED_FACTS])}")
            print("\nAll approved fact types:")
            for predicate, fact in session.state[APPROVED_FACTS].items():
                print(f"  ({fact['subject_label']}) --[{fact['predicate_label']}]--> ({fact['object_label']})")
        print("=" * 60)

        # Verify success
        if APPROVED_FACTS in session.state:
            print("\n✅ Test PASSED: Facts successfully proposed and approved!")
            return True
        else:
            print("\n❌ Test FAILED: Approved facts not found in session state")
            return False

    # Run the test
    asyncio.run(test_fact_extraction_agent())
