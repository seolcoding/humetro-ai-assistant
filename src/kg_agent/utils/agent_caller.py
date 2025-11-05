"""
AgentCaller helper class
Based on agentic_kg_build/1_intro_to_adk_1.ipynb
"""

from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AgentCaller:
    """A simple wrapper class for interacting with an ADK agent."""

    def __init__(self, agent: Agent, runner: Runner,
                 user_id: str, session_id: str):
        """Initialize the AgentCaller with required components."""
        self.agent = agent
        self.runner = runner
        self.user_id = user_id
        self.session_id = session_id

    async def get_session(self):
        """Get current session"""
        return await self.runner.session_service.get_session(
            app_name=self.runner.app_name,
            user_id=self.user_id,
            session_id=self.session_id
        )

    async def call(self, user_message: str, verbose: bool = False) -> str:
        """
        Call the agent with a query and return the response.

        Args:
            user_message: User's input message
            verbose: If True, print all events during execution

        Returns:
            Agent's final response as string
        """
        logger.info(f"\n>>> User Message: {user_message}")

        # Prepare the user's message in ADK format
        content = types.Content(
            role='user',
            parts=[types.Part(text=user_message)]
        )

        final_response_text = "Agent did not produce a final response."

        # Iterate through events to find the final answer
        async for event in self.runner.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            new_message=content
        ):
            if verbose:
                logger.info(
                    f"  [Event] Author: {event.author}, "
                    f"Type: {type(event).__name__}, "
                    f"Final: {event.is_final_response()}"
                )
                # Show tool calls if present
                if hasattr(event, 'actions') and event.actions:
                    if hasattr(event.actions, 'tool_calls') and event.actions.tool_calls:
                        for tool_call in event.actions.tool_calls:
                            logger.info(f"    [Tool Call] {tool_call.name}")
                    if hasattr(event.actions, 'tool_responses') and event.actions.tool_responses:
                        for tool_response in event.actions.tool_responses:
                            logger.info(f"    [Tool Response] {tool_response}")

            # Check for final response
            if event.is_final_response():
                if event.content and event.content.parts:
                    # Assuming text response in the first part
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    # Handle potential errors/escalations
                    final_response_text = (
                        f"Agent escalated: "
                        f"{event.error_message or 'No specific message.'}"
                    )
                break  # Stop processing events once final response is found

        logger.info(f"<<< Agent Response: {final_response_text}")
        return final_response_text


async def make_agent_caller(
    agent: Agent,
    initial_state: Optional[Dict[str, Any]] = None
) -> AgentCaller:
    """
    Create and return an AgentCaller instance for the given agent.

    This is a factory method that handles async initialization.

    Args:
        agent: The Agent instance to wrap
        initial_state: Optional initial state for the agent's memory

    Returns:
        AgentCaller instance ready to use
    """
    if initial_state is None:
        initial_state = {}

    app_name = agent.name + "_app"
    user_id = agent.name + "_user"
    session_id = agent.name + "_session_01"

    # Initialize a session service and a session
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=initial_state
    )

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service
    )

    return AgentCaller(agent, runner, user_id, session_id)
