#!/usr/bin/env python3
"""
File Suggestion Agent
Based on agentic_kg_build/4_file_suggestion.ipynb

Suggests relevant files for knowledge graph construction based on user goals.
"""

from google.adk.agents import Agent
from google.adk.tools import ToolContext
from typing import Dict, Any, List
from pathlib import Path
from itertools import islice
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
ALL_AVAILABLE_FILES = "all_available_files"
SUGGESTED_FILES = "suggested_files"
APPROVED_FILES = "approved_files"


# ============================================================================
# Agent Instructions (from notebook)
# ============================================================================

file_suggestion_agent_instruction = """
You are a constructive critic AI reviewing a list of files. Your goal is to suggest relevant files
for constructing a knowledge graph.

**Task:**
Review the file list for relevance to the kind of graph and description specified in the approved user goal.

For any file that you're not sure about, use the 'sample_file' tool to get
a better understanding of the file contents.

Only consider structured data files like CSV or JSON.

Prepare for the task:
- use the 'get_approved_user_goal' tool to get the approved user goal

Think carefully, repeating these steps until finished:
1. list available files using the 'list_available_files' tool
2. evaluate the relevance of each file, then record the list of suggested files using the 'set_suggested_files' tool
3. use the 'get_suggested_files' tool to get the list of suggested files
4. ask the user to approve the set of suggested files
5. If the user has feedback, go back to step 1 with that feedback in mind
6. If approved, use the 'approve_suggested_files' tool to record the approval
"""


# ============================================================================
# Tool Definitions (from notebook)
# ============================================================================

def list_available_files(tool_context: ToolContext) -> dict:
    f"""Lists files available for knowledge graph construction.
    All files are relative to the import directory.

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a {ALL_AVAILABLE_FILES} key with list of file names.
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    # get the import dir using the helper function
    import_dir = Path(get_data_import_dir())

    # get a list of relative file names, so files must be rooted at the import dir
    file_names = [str(x.relative_to(import_dir))
                 for x in import_dir.rglob("*")
                 if x.is_file()]

    # save the list to state so we can inspect it later
    tool_context.state[ALL_AVAILABLE_FILES] = file_names

    return tool_success(ALL_AVAILABLE_FILES, file_names)


def sample_file(file_path: str, tool_context: ToolContext) -> dict:
    """Samples a file by reading its content as text.

    Treats any file as text and reads up to a maximum of 100 lines.

    Args:
      file_path: file to sample, relative to the import directory

    Returns:
        dict: A dictionary containing metadata about the content,
            along with a sampling of the file.
            Includes a 'status' key ('success' or 'error').
            If 'success', includes a 'content' key with textual file content.
            If 'error', includes an 'error_message' key.
            The 'error_message' may have instructions about how to handle the error.
    """
    # Trust, but verify. The agent may invent absolute file paths.
    if Path(file_path).is_absolute():
        return tool_error("File path must be relative to the import directory. Make sure the file is from the list of available files.")

    import_dir = Path(get_data_import_dir())

    # create the full path by extending from the import_dir
    full_path_to_file = import_dir / file_path

    # of course, _that_ may not exist
    if not full_path_to_file.exists():
        return tool_error(f"File does not exist in import directory. Make sure {file_path} is from the list of available files.")

    try:
        # Treat all files as text
        with open(full_path_to_file, 'r', encoding='utf-8') as file:
            # Read up to 100 lines
            lines = list(islice(file, 100))
            content = ''.join(lines)
            return tool_success("content", content)

    except Exception as e:
        return tool_error(f"Error reading or processing file {file_path}: {e}")


def set_suggested_files(suggest_files: List[str], tool_context: ToolContext) -> Dict[str, Any]:
    """Set the suggested files to be used for data import.

    Args:
        suggest_files (List[str]): List of file paths to suggest

    Returns:
        Dict[str, Any]: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a {SUGGESTED_FILES} key with list of file names.
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    tool_context.state[SUGGESTED_FILES] = suggest_files
    return tool_success(SUGGESTED_FILES, suggest_files)


def get_suggested_files(tool_context: ToolContext) -> Dict[str, Any]:
    """Get the files to be used for data import.

    Returns:
        Dict[str, Any]: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a {SUGGESTED_FILES} key with list of file names.
                If 'error', includes an 'error_message' key.
    """
    return tool_success(SUGGESTED_FILES, tool_context.state[SUGGESTED_FILES])


def approve_suggested_files(tool_context: ToolContext) -> Dict[str, Any]:
    """Approves the {SUGGESTED_FILES} in state for further processing as {APPROVED_FILES}.

    If {SUGGESTED_FILES} is not in state, return an error.
    """
    if SUGGESTED_FILES not in tool_context.state:
        return tool_error("Current files have not been set. Take no action other than to inform user.")

    tool_context.state[APPROVED_FILES] = tool_context.state[SUGGESTED_FILES]
    return tool_success(APPROVED_FILES, tool_context.state[APPROVED_FILES])


# Tool list
file_suggestion_agent_tools = [
    get_approved_user_goal,
    list_available_files,
    sample_file,
    set_suggested_files,
    get_suggested_files,
    approve_suggested_files
]


# ============================================================================
# Agent Factory Function
# ============================================================================

def create_file_suggestion_agent(model_name: str = DEFAULT_KG_MODEL) -> Agent:
    """
    Create and return a File Suggestion Agent

    Args:
        model_name: LLM model to use (default from config)

    Returns:
        Agent instance configured for file suggestion
    """
    llm = get_llm(model_name)

    agent = Agent(
        name="file_suggestion_agent_v1",
        model=llm,
        description="Helps the user select files to import.",
        instruction=file_suggestion_agent_instruction,
        tools=file_suggestion_agent_tools,
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

    async def test_file_suggestion_agent():
        """Test the file suggestion agent with a sample conversation"""

        print("=" * 60)
        print("File Suggestion Agent Test")
        print("=" * 60)

        # Create agent
        print("\n1. Creating agent...")
        agent = create_file_suggestion_agent()

        # Create agent caller with initial state (approved user goal)
        print("2. Creating agent caller with approved user goal...")
        initial_state = {
            "approved_user_goal": {
                "kind_of_graph": "Seoul Traffic Information",
                "graph_description": "A knowledge graph for Seoul metropolitan traffic information including subway, bus, and other public transportation data to support information retrieval and question answering."
            }
        }
        caller = await make_agent_caller(agent, initial_state)

        # Check initial session state
        session = await caller.get_session()
        print(f"\n3. Initial session state has approved_user_goal: {'approved_user_goal' in session.state}")

        # Start conversation
        print("\n4. Starting conversation...")
        response1 = await caller.call(
            "What files can we use for import?",
            verbose=True
        )
        print(f"\n>>> Agent Response 1:\n{response1}\n")

        # Check suggested files
        session = await caller.get_session()
        if SUGGESTED_FILES in session.state:
            print(f"Suggested files: {session.state[SUGGESTED_FILES][:5]}...")  # Show first 5

        # Approve the suggestions
        print("\n5. Approving the suggestions...")
        response2 = await caller.call("Yes, let's do it", verbose=True)
        print(f"\n>>> Agent Response 2:\n{response2}\n")

        # Check final session state
        session = await caller.get_session()
        print("\n" + "=" * 60)
        print("Final Session State:")
        print("=" * 60)
        if ALL_AVAILABLE_FILES in session.state:
            print(f"Available files count: {len(session.state[ALL_AVAILABLE_FILES])}")
        if SUGGESTED_FILES in session.state:
            print(f"Suggested files count: {len(session.state[SUGGESTED_FILES])}")
        if APPROVED_FILES in session.state:
            print(f"Approved files count: {len(session.state[APPROVED_FILES])}")
            print(f"First 5 approved files: {session.state[APPROVED_FILES][:5]}")
        print("=" * 60)

        # Verify success
        if APPROVED_FILES in session.state:
            print("\n✅ Test PASSED: Files successfully suggested and approved!")
            return True
        else:
            print("\n❌ Test FAILED: Approved files not found in session state")
            return False

    # Run the test
    asyncio.run(test_file_suggestion_agent())
