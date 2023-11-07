# code from https://python.langchain.com/docs/integrations/callbacks/streamlit?ref=blog.langchain.dev#installation-and-setup
import openai
import os

from mongo_db import save_to_mongo

from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema.messages import SystemMessage

from dotenv import load_dotenv, find_dotenv

import streamlit as st
from haa_agent import agent_executor_lcel, agent_chain, tools, function_agent, multi_function_agent

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

st.set_page_config(page_title="Humetro-AI-Assistant",
                   page_icon="🤖")

if 'langchain_messages' not in st.session_state:
    st.session_state.langchain_messages = []
    
if 'output_language' not in st.session_state:
    st.session_state.output_language = 'KOREAN'

chat_history = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(memory_key='chat_history', k=3, return_messages=True, chat_memory=chat_history)

welcome_messages = {
    'KOREAN': '하단역의 인공지능 어시스턴트입니다. 궁금한 것이 있으시면 물어보세요!',
    'ENGLISH': 'I am the AI assistant at Hadan station. If you have any questions, please ask!',
    'CHINESE': '我是哈丹站的人工智能助手。如果您有任何问题，请提问！',
    'JAPANESE': '私はハダン駅のAIアシスタントです。ご質問があれば、どうぞお尋ねください'
}

if len(chat_history.messages) == 0:
    chat_history.add_ai_message(welcome_messages[st.session_state.output_language])
else:
    chat_history.messages[0].content = welcome_messages[st.session_state.output_language]

with st.sidebar:
    st.image('resized_avatar.png', use_column_width=True)
    output_language = st.selectbox(
        "Output Language",
        ("KOREAN", "ENGLISH", "CHINESE", "JAPANESE"),
        key="output_language",
        on_change=chat_history.clear,
        help="인공지능의 답변 언어를 선택하세요."
    )
    station = st.selectbox(
        "Choose Target Stations",
        ("하단역", "서면역", "연산역", "수영역"),
        disabled=True,
        help="현재는 하단역만 지원합니다."
    )
    st.write('세션 변수  확인하기(디버그)')
    st.json(st.session_state, expanded=False)


st.write("# 🚇 Humetro AI Assistant")
st.info('**OUTPUT LANGUAGE : ' + st.session_state.output_language + '**')

output_container = st.empty()
answer_container = st.empty()

output_container = output_container.container()

for msg in st.session_state.langchain_messages:
    output_container.chat_message(msg.type).write(msg.content)

if user_input := st.chat_input('이곳에 질문을 입력하세요.'):

# 중복을 배제하기 위해 마지막 두개의 메시지를 제외한 메시지를 출력한다.

    output_container.chat_message("user").write(user_input)
    answer_container = output_container.chat_message("assistant")
    st_callback = StreamlitCallbackHandler(answer_container)
    
    output_language = st.session_state.output_language
    stream = True
    
    import langchain
    langchain.debug = True
    if stream: # 스트리밍 지원하는 코드
        answer=function_agent.run({"input":user_input,
                                   "chat_history": chat_history.messages,
                                   "output_language":[SystemMessage(content=f"!!!IMPORTANT : You MUST ANSWER IN {output_language}")]},
                                  callbacks=[st_callback])
        answer_container.write(answer)
        # 메시지를 메시지를 보관하는 세션에 저장한다.
        chat_history.add_user_message(user_input)
        chat_history.add_ai_message(answer)
        save_to_mongo(user_input, answer)
    else: #lcel 사용한 코드
        answer = agent_executor_lcel.invoke({"input":user_input})
        answer_container.write(answer['input'])
        chat_history.add_user_message(user_input)
        chat_history.add_ai_message(answer['input'])
        save_to_mongo(user_input, answer['input'])
