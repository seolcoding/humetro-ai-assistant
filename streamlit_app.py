# code from https://python.langchain.com/docs/integrations/callbacks/streamlit?ref=blog.langchain.dev#installation-and-setup
from tabnanny import verbose
from click import prompt
from httpx import stream
import openai
import os

from llm_tools.GoogleRoutes import GoogleRouteTool
from llm_tools.HumetroFareTool import HumetroFareTool
from llm_tools.HumetroWebSearchTool import HumetroWebSearchTool
from mongo_db import save_to_mongo

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.messages import SystemMessage

from langchain.chat_models import ChatOpenAI
from langchain.prompts  import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import initialize_agent, AgentType, OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from dotenv import load_dotenv, find_dotenv

import streamlit as st
from llm_tools.prompts import humetro_system_prompt
# from audiorecorder import AudioRecorder, audiorecorder
# from gtts import gTTS

from capturing_callback_handler import playback_callbacks
from clear_results import with_clear_container

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

tools = [
    HumetroWebSearchTool(),
    HumetroFareTool(),
    GoogleRouteTool(),
    # HumetroWikiSearchTool(),
    # TrainScheduleTool(),
    # StationDistanceTool(),
]

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0, streaming=True)

st.set_page_config(page_title="Humetro-AI-Assistant",
                   page_icon="🤖")
if 'langchain_messages' not in st.session_state:
    st.session_state.langchain_messages = []

msgs = StreamlitChatMessageHistory(key="langchain_message")

memory = ConversationBufferMemory(memory_key='history', chat_memory=msgs)
if len(msgs.messages)  == 0:
    msgs.add_ai_message("저는 하단역의 인공지능 어시스턴트입니다. 무엇을 도와드릴까요?")

view_messages = st.expander("세션에 저장됨 메시지 목록을 봅니다.")

prompt = ChatPromptTemplate.from_messages([
    ("system", humetro_system_prompt),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name='agent_scratchpad')
])

haa_agent = OpenAIFunctionsAgent(
    llm = llm,
    tools=tools,
    prompt=prompt,
)

haa_executor = AgentExecutor.from_agent_and_tools(agent=haa_agent, tools=tools, verbose=True)

st.write("# 🚇 Humetro AI Assistant")
st.write("### 🤖 인공지능 어시스턴트에게 질문해보세요!")

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

output_container = st.empty()
answer_container = st.empty()
if user_input := st.chat_input():
    output_container = output_container.container()
    for msg in st.session_state.langchain_messages:
        output_container.chat_message("user").write(msg["user"])
        output_container.chat_message("assistant").write(msg["assistant"])
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant")
    st_callback = StreamlitCallbackHandler(answer_container)

    # answer = haa.run(user_input, callbacks=[st_callback])
    answer = haa_executor.run({"input":user_input}, callbacks=[st_callback])
    answer_container.write(answer)
    st.session_state.langchain_messages.append({"user": user_input, "assistant": answer})

    save_to_mongo(user_input, answer)

# if audio_input := st.audio_recorder("audio.wav"):
#     output_container = output_container.container()
#     output_container.chat_message("user").write(audio_input)

#     answer_container = output_container.chat_message("assistant")
#     st_callback = StreamlitCallbackHandler(answer_container)

#     answer = haa.run(audio_input, callbacks=[st_callback])
#     answer_container.write(answer)

#     save_to_mongo(audio_input, answer)

with view_messages:
    view_messages.json(st.session_state.langchain_messages)