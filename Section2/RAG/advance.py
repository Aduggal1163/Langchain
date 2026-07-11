from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel,Field
from typing import List
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

import logging
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever,BM25Retriever,ParentDocumentRetriever,EnsembleRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.storage import InMemoryStore

load_dotenv()

"""
Advance RAG Patterns
Multi-query, self-query, compression, hybrid search
"""

logging.basicConfig()
logging.getLogger("langchain.retriever.multi_query").setLevel(logging.INFO)
#enabled logging to see multi-query generation
TECH_DOCS = [
    Document(
        page_content="""
        Basic Retrieval-Augmented Generation (RAG) retrieves documents using a single user query.
        It may miss relevant information because different wording can produce different search results.
        Basic RAG also returns entire chunks, introducing unnecessary context into the prompt.
        """,
        metadata={
            "topic": "Basic RAG",
            "level": "beginner",
            "source": "rag_intro.pdf"
        }
    ),

    Document(
        page_content="""
        Multi-Query Retriever improves retrieval by generating multiple versions of the user's question.
        Each variation searches the vector store independently, and the retrieved documents are merged
        and deduplicated before being sent to the language model.
        """,
        metadata={
            "topic": "Multi-Query Retriever",
            "level": "intermediate",
            "source": "advanced_rag.pdf"
        }
    ),

    Document(
        page_content="""
        Self-Query Retriever allows an LLM to convert natural language into both a semantic search query
        and structured metadata filters. For example, 'Find sci-fi movies from 2020 with rating above 8'
        becomes a semantic search for 'sci-fi movies' with metadata filters:
        genre='sci-fi', year=2020, rating>8.
        """,
        metadata={
            "topic": "Self-Query Retriever",
            "level": "advanced",
            "source": "advanced_rag.pdf",
            "category": "retriever"
        }
    ),

    Document(
        page_content="""
        Contextual Compression Retriever reduces prompt size by extracting only the parts of retrieved
        documents relevant to the user's question. This minimizes noise and improves answer quality.
        """,
        metadata={
            "topic": "Contextual Compression",
            "level": "advanced",
            "source": "advanced_rag.pdf",
            "category": "compression"
        }
    ),

    Document(
        page_content="""
        Hybrid Search combines keyword-based retrieval (BM25) with semantic vector search.
        BM25 is effective for exact terms, identifiers, and code snippets, while semantic search
        captures concepts, synonyms, and meaning. The results are combined using weighted scoring.
        """,
        metadata={
            "topic": "Hybrid Search",
            "level": "advanced",
            "source": "retrieval_methods.pdf",
            "category": "search"
        }
    ),

    Document(
        page_content="""
        Parent Document Retriever indexes small child chunks for precise retrieval but returns the
        corresponding larger parent chunk. This preserves surrounding context while maintaining
        retrieval accuracy.
        """,
        metadata={
            "topic": "Parent Document Retriever",
            "level": "advanced",
            "source": "advanced_rag.pdf",
            "category": "retriever"
        }
    ),
]

def create_base_vectorStore():
    """Create a basic vector store for demos"""
    return Chroma.from_documents(
        documents=TECH_DOCS,
        embedding=OpenAIEmbeddings(model = 'text-embedding-3-small')
    )

