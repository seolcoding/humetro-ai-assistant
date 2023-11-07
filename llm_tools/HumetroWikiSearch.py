# 노션에 저장된 Humetro wiki를 대상으로 semantic search를 수행한다.
import pickle

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
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

# To load the list of documents from the binary file elsewhere
    with open('documents_export.pkl', 'rb') as f:
        loaded_documents = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(loaded_documents)
    bm25_retriever.k = 2
    
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory="./chroma/wiki", embedding_function=embedding)
    chroma_retriever = vectordb.as_retriever(search_kwargs={"k":2})
    
    
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5])  
    result_docs = ensemble_retriever.get_relevant_documents(query)
    return '\n'.join([doc.page_content for doc in result_docs])

if __name__ == "__main__":
    get_wiki('보관함')