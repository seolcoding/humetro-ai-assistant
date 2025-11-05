"""
Common tools shared across KG building agents
Based on agentic_kg_build notebooks
"""

from google.adk.tools import ToolContext
from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# State keys from user_intent_agent
APPROVED_USER_GOAL = "approved_user_goal"


def tool_success(key: str, data: Any) -> Dict[str, Any]:
    """Format successful tool result"""
    return {
        "status": "success",
        "key": key,
        "data": data
    }


def tool_error(message: str) -> Dict[str, Any]:
    """Format tool error result"""
    return {
        "status": "error",
        "error_message": message
    }


def get_approved_user_goal(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Retrieve the approved user goal from the session state.

    Returns:
        Dict containing the approved user goal with 'kind_of_graph' and 'graph_description'
    """
    if APPROVED_USER_GOAL not in tool_context.state:
        return tool_error("approved_user_goal not found in state. User intent must be established first.")

    return tool_success(APPROVED_USER_GOAL, tool_context.state[APPROVED_USER_GOAL])


def get_data_import_dir() -> Path:
    """
    Get the directory containing data files for import.

    In our case, this is the markdown_filtered directory with crawled Seoul traffic data
    filtered by date range (2021-2025) and including documents without date info.

    Returns:
        Path to the import directory
    """
    project_root = Path(__file__).parent.parent.parent.parent
    import_dir = project_root / "data" / "crawled" / "seoul_traffic" / "markdown_filtered"

    if not import_dir.exists():
        raise FileNotFoundError(f"Import directory not found: {import_dir}")

    return import_dir
