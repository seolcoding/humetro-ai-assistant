#!/usr/bin/env python3
"""
Named Entity Recognition (NER) Agent for Unstructured Data
Based on agentic_kg_build/6_schema_proposal_unstructured.ipynb

Proposes entity types that can be extracted from unstructured text files.
"""

from google.adk.agents import Agent
from google.adk.tools import ToolContext
from typing import Dict, Any, List
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
PROPOSED_ENTITIES = "proposed_entity_types"
APPROVED_ENTITIES = "approved_entity_types"


# ============================================================================
# Agent Instructions (from notebook)
# ============================================================================

ner_agent_role_and_goal = """
  You are a top-tier algorithm designed for analyzing text files and proposing
  the kind of named entities that could be extracted which would be relevant
  for a user's goal.
  """

ner_agent_hints = """
  Entities are people, places, things and qualities, but not quantities.
  Your goal is to propose a list of the type of entities, not the actual instances
  of entities.

  There are two general approaches to identifying types of entities:
  - well-known entities: these closely correlate with approved node labels in an existing graph schema
  - discovered entities: these may not exist in the graph schema, but appear consistently in the source text

  Design rules for well-known entities:
  - always use existing well-known entity types. For example, if there is a well-known type "Person", and people appear in the text, then propose "Person" as the type of entity.
  - prefer reusing existing entity types rather than creating new ones

  Design rules for discovered entities:
  - discovered entities are consistently mentioned in the text and are highly relevant to the user's goal
  - always look for entities that would provide more depth or breadth to the existing graph
  - for example, if the user goal is to represent social communities and the graph has "Person" nodes, look through the text to discover entities that are relevant like "Hobby" or "Event"
  - avoid quantitative types that may be better represented as a property on an existing entity or relationship.
  - for example, do not propose "Age" as a type of entity. That is better represented as an additional property "age" on a "Person".
"""

ner_agent_chain_of_thought_directions = """
  Prepare for the task:
  - use the 'get_approved_user_goal' tool to get the user goal
  - use the 'get_approved_files' tool to get the list of approved files
  - use the 'get_well_known_types' tool to get the approved node labels

  Think step by step:
  1. Sample some of the files using the 'sample_file' tool to understand the content
  2. Consider what well-known entities are mentioned in the text
  3. Discover entities that are frequently mentioned in the text that support the user's goal
  4. Use the 'set_proposed_entities' tool to save the list of well-known and discovered entity types
  5. Use the 'get_proposed_entities' tool to retrieve the proposed entities and present them to the user for their approval
  6. If the user approves, use the 'approve_proposed_entities' tool to finalize the entity types
  7. If the user does not approve, consider their feedback and iterate on the proposal
"""

ner_agent_instruction = f"""
{ner_agent_role_and_goal}
{ner_agent_hints}
{ner_agent_chain_of_thought_directions}
"""


# ============================================================================
# Tool Definitions (from notebook)
# ============================================================================

def get_approved_files(tool_context: ToolContext) -> Dict[str, Any]:
    """Get the approved list of files to process."""
    approved_files = tool_context.state.get("approved_files", [])
    return tool_success("approved_files", approved_files)


def get_well_known_types(tool_context: ToolContext) -> dict:
    """Gets the approved labels that represent well-known entity types in the graph schema."""
    construction_plan = tool_context.state.get("approved_construction_plan", {})
    # approved labels are the keys for each construction plan entry where `construction_type` is "node"
    approved_labels = {entry["label"] for entry in construction_plan.values() if entry["construction_type"] == "node"}
    return tool_success("approved_labels", list(approved_labels))


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


def set_proposed_entities(proposed_entity_types: List[str], tool_context: ToolContext) -> dict:
    """Sets the list proposed entity types to extract from unstructured text."""
    tool_context.state[PROPOSED_ENTITIES] = proposed_entity_types
    return tool_success(PROPOSED_ENTITIES, proposed_entity_types)


def get_proposed_entities(tool_context: ToolContext) -> dict:
    """Gets the list of proposed entity types to extract from unstructured text."""
    proposed = tool_context.state.get(PROPOSED_ENTITIES, [])
    return tool_success(PROPOSED_ENTITIES, proposed)


def approve_proposed_entities(tool_context: ToolContext) -> dict:
    """Upon approval from user, records the proposed entity types as an approved list of entity types

    Only call this tool if the user has explicitly approved the suggested files.
    """
    if PROPOSED_ENTITIES not in tool_context.state:
        return tool_error("No proposed entity types to approve. Please set proposed entities first, ask for user approval, then call this tool.")
    tool_context.state[APPROVED_ENTITIES] = tool_context.state.get(PROPOSED_ENTITIES)
    return tool_success(APPROVED_ENTITIES, tool_context.state[APPROVED_ENTITIES])


# Tool list
ner_agent_tools = [
    get_approved_user_goal,
    get_approved_files,
    sample_file,
    get_well_known_types,
    set_proposed_entities,
    get_proposed_entities,
    approve_proposed_entities
]


# ============================================================================
# Agent Factory Function
# ============================================================================

def create_ner_agent(model_name: str = DEFAULT_KG_MODEL) -> Agent:
    """
    Create and return a Named Entity Recognition Agent

    Args:
        model_name: LLM model to use (default from config)

    Returns:
        Agent instance configured for NER
    """
    llm = get_llm(model_name)

    agent = Agent(
        name="ner_schema_agent_v1",
        model=llm,
        description="Proposes the kind of named entities that could be extracted from text files.",
        instruction=ner_agent_instruction,
        tools=ner_agent_tools,
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

    async def test_ner_agent():
        """Test the NER agent with Seoul traffic data"""

        print("=" * 60)
        print("NER Agent Test")
        print("=" * 60)

        # Create agent
        print("\n1. Creating agent...")
        agent = create_ner_agent()

        # Prepare initial state for Seoul traffic data
        print("2. Preparing initial state...")

        # Get first 5 markdown files
        import_dir = get_data_import_dir()
        md_files = sorted([str(f.relative_to(import_dir)) for f in import_dir.rglob("*.md")])[:5]

        initial_state = {
            "approved_user_goal": {
                "kind_of_graph": "Seoul Traffic Information",
                "graph_description": "A knowledge graph for Seoul metropolitan traffic information including subway, bus, and other public transportation data to support information retrieval and question answering."
            },
            "approved_files": md_files,
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

        print(f"   Using {len(md_files)} markdown files")
        print(f"   Files: {md_files[:3]}...")

        # Create agent caller
        print("\n3. Creating agent caller...")
        caller = await make_agent_caller(agent, initial_state)

        # Check initial session state
        session = await caller.get_session()
        print(f"   Initial state has approved_user_goal: {'approved_user_goal' in session.state}")

        # Start conversation
        print("\n4. Starting entity extraction analysis...")
        response1 = await caller.call(
            "Add Seoul traffic information to the knowledge graph. Extract relevant entities from the markdown files.",
            verbose=True
        )
        print(f"\n>>> Agent Response 1:\n{response1}\n")

        # Check proposed entities
        session = await caller.get_session()
        if PROPOSED_ENTITIES in session.state:
            print(f"\nProposed entities: {session.state[PROPOSED_ENTITIES]}")

            # Approve them
            print("\n5. Approving proposed entities...")
            response2 = await caller.call("Approve the proposed entities.", verbose=True)
            print(f"\n>>> Agent Response 2:\n{response2}\n")

        # Check final session state
        session = await caller.get_session()
        print("\n" + "=" * 60)
        print("Final Session State:")
        print("=" * 60)
        if PROPOSED_ENTITIES in session.state:
            print(f"Proposed entities: {session.state[PROPOSED_ENTITIES]}")
        if APPROVED_ENTITIES in session.state:
            print(f"Approved entities: {session.state[APPROVED_ENTITIES]}")
        print("=" * 60)

        # Verify success
        if APPROVED_ENTITIES in session.state:
            print("\n✅ Test PASSED: Entities successfully proposed and approved!")
            return True
        else:
            print("\n❌ Test FAILED: Approved entities not found in session state")
            return False

    # Run the test
    asyncio.run(test_ner_agent())
