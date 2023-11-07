# 노션에 저장된 Humetro wiki를 대상으로 semantic search를 수행한다.

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.tools import BaseTool
from langchain.agents import tool
from pydantic.v1 import BaseModel, Field

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file


class HumetroWikiSearchInput(BaseModel):
    """Input for HumetroWikiSearchTool Check"""
    query: str = Field(...,
                       description="Entire user input sentence for semantic search")


@tool(args_schema=HumetroWikiSearchInput)
def get_wiki(query:str) -> str:
    """You should use this tool to answer following topics, 환승, 전화번호, 정기승차권, 다자녀 안내, 이중개표,  운임반환, 재개표 , 단체승차권, 부가운임, 현금영수증, 휴대금지, 금지행위, 캐시비 및 티머니 카드, 고객응대, 유실물, 영업배상, 물품보관함, 자전거, 봉사활동, 교통카드, 복자교통카드, 동백패스"""
    # TODO implement hybrid ensemble search
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory="./chroma/wiki", embedding_function=embedding)
    result = vectordb.similarity_search(query, k=3)
    print("wiki search result: ", result)
    return result