# code from https://python.langchain.com/docs/integrations/callbacks/streamlit?ref=blog.langchain.dev#installation-and-setup

from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms.openai import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

llm = OpenAI(temperature=0.9, streaming=True)
wrapper= DuckDuckGoSearchAPIWrapper(region="ko-KR", time="d", max_results=10)
tools = [DuckDuckGoSearchResults(api_wrapper=wrapper, backend='news')]
tools = load_tools(['ddg-search'])

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container(), collapse_completed_thoughts=False)
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)