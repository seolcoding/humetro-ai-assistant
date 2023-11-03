# code from https://python.langchain.com/docs/integrations/callbacks/streamlit?ref=blog.langchain.dev#installation-and-setup
import warnings
import langchain
import openai
import os

from llm_tools.prompts import humetro_system_prompt
from llm_tools.GoogleRoutes import GoogleRouteTool
from llm_tools.HumetroFareTool import HumetroFareTool
from llm_tools.HumetroWebSearchTool import HumetroWebSearchTool
from mongo_db import save_to_mongo

from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents import AgentType, initialize_agent, load_tools, AgentExecutor
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools.render import format_tool_to_openai_function
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms.openai import OpenAI
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import sys


from capturing_callback_handler import playback_callbacks
from clear_results import with_clear_container

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

tools = [
    HumetroWebSearchTool(),
    HumetroFareTool(),
    # HumetroWikiSearchTool(),
    GoogleRouteTool(),
    # TrainScheduleTool(),
    # StationDistanceTool(),
]

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0, streaming=True)

haa = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)


st.set_page_config(page_title="Humetro-AI-Assistant",
                   page_icon="🤖")
st.write("# 🚇 Humetro AI Assistant")
st.write("### 🤖 인공지능 어시스턴트에게 질문해보세요!")

# with st.form(key="form"):
#     user_input = ""
#     if not user_input:
#         user_input = "질문을 입력하세요."
#     submit_clicked = st.form_submit_button("Submit Question")

output_container = st.empty()
answer_container = st.empty()
if user_input := st.chat_input():
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant")
    st_callback = StreamlitCallbackHandler(answer_container)

    answer = haa.run(user_input, callbacks=[st_callback])
    answer_container.write(answer)

    save_to_mongo(user_input, answer)
