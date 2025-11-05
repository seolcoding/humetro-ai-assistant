#!/usr/bin/env python3
"""
User Intent Agent
Based on agentic_kg_build/3_user_intent.ipynb

Helps users ideate on knowledge graph use cases through conversational interaction.
"""

from google.adk.agents import Agent
from google.adk.tools import ToolContext
from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.kg_agent.config.llm_config import get_llm, DEFAULT_KG_MODEL
from src.kg_agent.utils.neo4j_for_adk import graphdb
import logging

logger = logging.getLogger(__name__)

# State keys
PERCEIVED_USER_GOAL = "perceived_user_goal"
APPROVED_USER_GOAL = "approved_user_goal"


# ============================================================================
# Tool Helper Functions (from neo4j_for_adk.py pattern)
# ============================================================================

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


# ============================================================================
# Agent Instructions (from notebook)
# ============================================================================

agent_role_and_goal = """
    You are an expert at knowledge graph use cases.
    Your primary goal is to help the user come up with a knowledge graph use case.
"""

agent_conversational_hints = """
    If the user is unsure what to do, make some suggestions based on classic use cases like:
    - social network involving friends, family, or professional relationships
    - logistics network with suppliers, customers, and partners
    - recommendation system with customers, products, and purchase patterns
    - fraud detection over multiple accounts with suspicious patterns of transactions
    - pop-culture graphs with movies, books, or music
"""

agent_output_definition = """
    A user goal has two components:
    - kind_of_graph: at most 3 words describing the graph, for example "social network" or "USA freight logistics"
    - description: a few sentences about the intention of the graph, for example "A dynamic routing and delivery system for cargo." or "Analysis of product dependencies and supplier alternatives."
"""

agent_chain_of_thought_directions = """
    Think carefully and collaborate with the user:
    1. Understand the user's goal, which is a kind_of_graph with description
    2. Ask clarifying questions as needed
    3. When you think you understand their goal, use the 'set_perceived_user_goal' tool to record your perception
    4. Present the perceived user goal to the user for confirmation
    5. If the user agrees, use the 'approve_perceived_user_goal' tool to approve the user goal. This will save the goal in state under the 'approved_user_goal' key.
"""

complete_agent_instruction = f"""
{agent_role_and_goal}
{agent_conversational_hints}
{agent_output_definition}
{agent_chain_of_thought_directions}
"""


# ============================================================================
# Tool Definitions (from notebook)
# ============================================================================

def set_perceived_user_goal(kind_of_graph: str, graph_description: str, tool_context: ToolContext):
    """Sets the perceived user's goal, including the kind of graph and its description.

    Args:
        kind_of_graph: 2-3 word definition of the kind of graph, for example "recent US patents"
        graph_description: a single paragraph description of the graph, summarizing the user's intent
    """
    user_goal_data = {"kind_of_graph": kind_of_graph, "graph_description": graph_description}
    tool_context.state[PERCEIVED_USER_GOAL] = user_goal_data
    return tool_success(PERCEIVED_USER_GOAL, user_goal_data)


def approve_perceived_user_goal(tool_context: ToolContext):
    """Upon approval from user, will record the perceived user goal as the approved user goal.

    Only call this tool if the user has explicitly approved the perceived user goal.
    """
    # Trust, but verify.
    # Require that the perceived goal was set before approving it.
    if PERCEIVED_USER_GOAL not in tool_context.state:
        return tool_error("perceived_user_goal not set. Set perceived user goal first, or ask clarifying questions if you are unsure.")

    tool_context.state[APPROVED_USER_GOAL] = tool_context.state[PERCEIVED_USER_GOAL]

    return tool_success(APPROVED_USER_GOAL, tool_context.state[APPROVED_USER_GOAL])


# Tool list
user_intent_agent_tools = [set_perceived_user_goal, approve_perceived_user_goal]


# ============================================================================
# Agent Factory Function
# ============================================================================

def create_user_intent_agent(model_name: str = DEFAULT_KG_MODEL) -> Agent:
    """
    Create and return a User Intent Agent

    Args:
        model_name: LLM model to use (default from config)

    Returns:
        Agent instance configured for user intent understanding
    """
    llm = get_llm(model_name)

    agent = Agent(
        name="user_intent_agent_v1",
        model=llm,
        description="Helps the user ideate on a knowledge graph use case.",
        instruction=complete_agent_instruction,
        tools=user_intent_agent_tools,
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

    async def test_user_intent_agent():
        """Test the user intent agent with a sample conversation"""

        print("=" * 60)
        print("User Intent Agent Test")
        print("=" * 60)

        # Create agent
        print("\n1. Creating agent...")
        agent = create_user_intent_agent()

        # Create agent caller
        print("2. Creating agent caller...")
        caller = await make_agent_caller(agent)

        # Check initial session state
        session = await caller.get_session()
        print(f"\n3. Initial session state: {session.state}")

        # Start conversation
        print("\n4. Starting conversation...")
        response1 = await caller.call(
            """I'd like a bill of materials graph (BOM graph) which includes all levels from suppliers to finished product,
            which can support root-cause analysis.""",
            verbose=True
        )
        print(f"\n>>> Agent Response 1:\n{response1}\n")

        # Check if perceived goal was set
        session = await caller.get_session()
        if PERCEIVED_USER_GOAL not in session.state:
            print("5. Perceived goal not set yet, providing more details...")
            response2 = await caller.call(
                "I'm concerned about possible manufacturing or supplier issues.",
                verbose=True
            )
            print(f"\n>>> Agent Response 2:\n{response2}\n")

        # Approve the goal
        print("6. Approving the goal...")
        response3 = await caller.call("Approve that goal.", verbose=True)
        print(f"\n>>> Agent Response 3:\n{response3}\n")

        # Check final session state
        session = await caller.get_session()
        print("\n" + "=" * 60)
        print("Final Session State:")
        print("=" * 60)
        print(f"Perceived User Goal: {session.state.get(PERCEIVED_USER_GOAL)}")
        print(f"Approved User Goal: {session.state.get(APPROVED_USER_GOAL)}")
        print("=" * 60)

        # Verify success
        if APPROVED_USER_GOAL in session.state:
            print("\n✅ Test PASSED: User intent successfully captured and approved!")
            return True
        else:
            print("\n❌ Test FAILED: Approved user goal not found in session state")
            return False

    # Run the test
    asyncio.run(test_user_intent_agent())
