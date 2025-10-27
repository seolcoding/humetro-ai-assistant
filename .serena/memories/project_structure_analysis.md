# Humetro AI Assistant - Project Structure Analysis

## Project Overview
- **Purpose**: AI-powered assistant for Busan Metro (Humetro) using RAG technology
- **Main Technologies**: Python 3.12, LangChain, OpenAI GPT, AutoRAG
- **Architecture**: RAG (Retrieval-Augmented Generation) with web crawling and real-time search

## Core Components

### 1. LLM Tools (src/llm_tools/)
- **HumetroWikiSearch.py**: Wiki search for specific topics (환승, 정기승차권, etc.)
- **TMapTool.py**: Route planning using TMap and Kakao APIs
- **HumetroFare.py**: Fare calculation based on age groups
- **HumetroSchedule.py**: Schedule information
- **HumetroWebSearch.py**: Real-time website search
- **StationDistanceTool.py**: Station distance calculations
- **StationInformationTool.py**: Station information retrieval
- **GoogleRoutes.py**: Google routes integration
- **basic_tools.py**: Basic utilities (current time, etc.)

### 2. Data Processing (src/data_processor/)
- **crawler.py**: Web scraping functionality
- **generate_qa_data.py**: QA dataset generation from crawled data
- **evaluate_rag.py**: RAG pipeline evaluation using AutoRAG
- **rag_config.yaml**: RAG configuration (retrieval, reranking, embeddings)
- **dasan.py**: Data processing utilities

### 3. Legacy Components (src/legacy/)
- **app.py**: Streamlit web interface
- **haa_agent.py**: Main agent executor
- **mongo_db.py**: MongoDB integration

### 4. Common Utilities (src/common/)
- **get_root.py**: Project root path utilities
- **hwp_to_pdf_converter.py**: HWP to PDF conversion

## Main Entry Points
- **main.py**: CLI for data generation and RAG evaluation
- **src/legacy/app.py**: Streamlit web application

## Key Features
1. Multi-language support (Korean, English, Chinese, Japanese)
2. Real-time route planning with fare calculation
3. Station information and schedule queries
4. Web crawling and data processing pipeline
5. RAG-based question answering
6. Evaluation and optimization using AutoRAG

## Dependencies
- Core: OpenAI, LangChain, AutoRAG
- Vector stores: ChromaDB, Milvus, Pinecone, Weaviate
- Web: Streamlit, FastAPI
- Data processing: Pandas, NumPy
- ML/AI: Transformers, Torch, Sentence-transformers