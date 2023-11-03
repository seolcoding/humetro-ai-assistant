import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from bs4 import BeautifulSoup

from duckduckgo_search import ddg
from langchain.document_loaders import WebBaseLoader
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

# this supresses the warning for using https without certificate
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


class HumetroWebSearchInputChecker(BaseModel):
    """Input for humetro web search check"""
    query: str = Field(...,
                       description="Query to search on http://www.humetro.busan.kr/")


class HumetroWebSearchTool(BaseTool):
    name = "get_humetro_web_search"
    description = "If you don't find adequete tools, use this tool to Get the search result for a given query on http://www.humetro.busan.kr/"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, sdch, br',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    }
    docs = []
    splits = []
    vectordb: Chroma | None = None
    embedding = OpenAIEmbeddings()

    def split_docs(self):
        chunk_size = 1000
        chunk_overlap = 200
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap)
        self.splits = text_splitter.split_documents(self.docs)

    def gen_vectordb(self):
        self.vectordb = Chroma.from_documents(
            documents=self.splits,
            embedding=self.embedding,
            persist_directory='./chroma',
        )

    def query_vectordb(self, query):
        docs = self.vectordb.similarity_search(query, k=3)
        print('vectordb query results: ', docs)
        return docs

    def _run(self, query):
        prefix = "site:http://www.humetro.busan.kr/homepage/default/ "
        query = prefix + query
        search_results = [i['href'] for i in ddg(query, max_results=5)]
        target_urls = set()

        for url in search_results:
            if "main.do" in url or "login.do" in url:
                continue
            target_urls.add(url)

        for url in target_urls:
            res = self.parse_response(url)
            if res:
                self.docs.append(res)

        if len(self.docs) == 0:
            return ""

        self.split_docs()
        self.gen_vectordb()
        result = self.query_vectordb(query)
        return result

    def _arun(self, query):
        raise NotImplementedError(
            "HumetroWebSearchTool does not support async")

    def clean_whitespace(self, text):
        text = text.replace(
            "\n현재 열람하신 페이지의 내용이나 사용 편의성에 만족하십니까?\n\n\xa0\xa0매우만족(5점)\n\n\n\xa0\xa0만족(4점)\n\n\n\xa0\xa0보통(3점)\n\n\n\xa0\xa0불만족(2점)\n\n\n\xa0\xa0매우불만족(1점)\n\n\n평가하기\n",
            ""
        ).replace("\n글자크기\n글자크기 확대하기\n글자크기 축소하기\n", "").replace('\xa0', ' ').replace('\t', ' ')

        # Split text into lines and remove empty lines
        lines = [line for line in text.split('\n') if line.strip()]

        # Join the lines back into a string
        cleaned_text = '\n'.join(lines)
        return cleaned_text

    def parse_response(self, url):
        res = requests.get(url, headers=self.headers, verify=False)
        if res.status_code != 200:
            return False

        soup = BeautifulSoup(res.content, 'html.parser')
        if '올바른 메뉴가 아닙니다.' in soup.text:
            return False

        content_tag = soup.select_one('#contents')
        if content_tag is None:
            return False
        content_text = content_tag.text
        content_text = self.clean_whitespace(content_text)

        doc = Document(
            page_content=content_text,
            metadata={
                "url": url,
                "title": soup.title.text.strip(),
                "description": soup.find("meta", {"name": "description"})["content"]
            }
        )
        return doc

    args_schema: Optional[Type[BaseModel]] = HumetroWebSearchInputChecker


if __name__ == "__main__":
    tool = HumetroWebSearchTool()
    tool.run("도시철도 견학정보")
