# code from https://python.langchain.com/docs/integrations/callbacks/streamlit?ref=blog.langchain.dev#installation-and-setup
from tabnanny import verbose
import openai
import os

from llm_tools.GoogleRoutes import GoogleRouteTool
from llm_tools.HumetroFareTool import HumetroFareTool
from llm_tools.HumetroWebSearchTool import HumetroWebSearchTool
from llm_tools.TrainScheduleTool import TrainScheduleTool
from llm_tools.HumetroWikiSearchTool import HumetroWikiSearchTool

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.messages import SystemMessage

from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import  OpenAIFunctionsAgent, AgentExecutor, OpenAIMultiFunctionsAgent, AgentType, initialize_agent

from langchain.tools.render import format_tool_to_openai_function

from langchain.chat_models import ChatOpenAI
from langchain.prompts  import ChatPromptTemplate, MessagesPlaceholder, ChatMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from dotenv import load_dotenv, find_dotenv

from llm_tools.prompts import humetro_system_prompt
from capturing_callback_handler import playback_callbacks
from clear_results import with_clear_container

import langchain

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

tools = [
    HumetroWebSearchTool(),
    HumetroFareTool(),
    GoogleRouteTool(),
    TrainScheduleTool(),
    HumetroWikiSearchTool(),
    # TrainScheduleTool(),
    # StationDistanceTool(),
]

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0, streaming=True)


prompt = ChatPromptTemplate.from_messages([
    ("system", humetro_system_prompt),
    ("user", "{input}"),
    # add below to use memory
    MessagesPlaceholder(variable_name='agent_scratchpad')
])

haa_agent = OpenAIMultiFunctionsAgent(
    llm = llm,
    tools=tools,
    prompt=prompt,
)
haa_agent = initialize_agent(
        tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_error=True,
)
print(haa_agent.agent.llm_chain.prompt) # type ChatPrompt Template


functions = [format_tool_to_openai_function(t) for t in tools]

llm = ChatOpenAI(
    temperature=0, model="gpt-3.5-turbo-16k").bind(functions=functions)

prompt = ChatPromptTemplate.from_messages([
    ("system", humetro_system_prompt),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name='agent_scratchpad')
])

chain = prompt | llm | OpenAIFunctionsAgentOutputParser()

agent_chain = RunnablePassthrough.assign(
    agent_scratchpad=lambda x: format_to_openai_functions(
        x['intermediate_steps'])
) | chain