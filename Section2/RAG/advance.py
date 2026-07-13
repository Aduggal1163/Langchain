from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

def multi_query_retriever():
    """It generates multi query perspectives."""
    print("="*60,"Multi Query retriever","="*60)
    vector_store = create_base_vectorStore()
    llm = init_chat_model('gpt-4o-mini',temperature = 0)

    #multi query retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={'k':3}),
        llm=llm
    )

    query = '"What is RAG?"'
    print(f"\n Original query : {query}")
    print("\n The retriever will generate multiple query variations...")
    print("\n(Check log info above for generated queries)\n")

    #retrieve documents
    docs = retriever.invoke(query)
    print(f"Unique documents : {len(docs)}")
    for i,doc in enumerate(docs):
        print(
            f"\n{i+1} . [{doc.metadata.get('topic','N/A')}] {doc.page_content[:100]}"
        )

def contextual_compression():
    """Contextual compression extracts only relevant part"""
    print("="*60,"Contextual compression retriever","="*60)
    vector_store = create_base_vectorStore()
    llm = init_chat_model('gpt-4o-mini',temperature = 0)
    query = 'what do you mean by retrievers in LLM'
    
    #Create compression
    compressor = LLMChainExtractor.from_llm(llm)
    #Wrap retriever with compression
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_store.as_retriever(search_kwargs = {'k':3})
    )
    compression_docs = compression_retriever.invoke(query)
    print(f"Original Query: {query}")

    basic_retriever = vector_store.as_retriever(search_kwargs = {'k':3})
    basic_docs = basic_retriever.invoke(query)

    print("="*60,"Basic compression retriever results","="*60)
    for i,doc in enumerate(basic_docs,start=1):
        print(i)
        print(f"Length : {len(doc.page_content)} chars")
        print(f"Page content: {doc.page_content}")

    print("="*60,"Contextual compression retriever results","="*60)
    for i,doc in enumerate(compression_docs,start=1):
        print(i)
        print(f"Length : {len(doc.page_content)} chars")
        print(f"Page content: {doc.page_content}")

def hybrid_search():
    """Hybrid Search containing (BM25) for keywords + sementic search"""
    print("="*60,"Hybrid retriever","="*60)
    vector_store = create_base_vectorStore()
    llm = init_chat_model('gpt-4o-mini',temperature = 0)
    #BM25 keyword retriever
    bm25_retriever = BM25Retriever.from_documents(TECH_DOCS)
    bm25_retriever.k = 3

    #Sementic Search
    sementic_retriever = vector_store.as_retriever(search_kwargs = {'k':3})

    #Combine both of them togethers
    essemble_both_retrievers = EnsembleRetriever(
        retrievers=[bm25_retriever,sementic_retriever],
        weights=[0.4,0.6] #40% keywords and 60 sementic% 
    )

    queries = [
        "PostgresSQL", # keyword heavy
        "What database stores embeddings"
    ]

    for query in queries:
        print(f"Query: {query}")
        print("="*100)
        BM25_result = bm25_retriever.invoke(query)
        sementic_result = sementic_retriever.invoke(query)
        essemble_result = essemble_both_retrievers.invoke(query)
        print(f"BM25 result is : {BM25_result[0].page_content[:60]}")
        print(f"sementic result is : {sementic_result[0].page_content[:60]}")
        print(f"essemble result is : {essemble_result[0].page_content[:60]}")

def parent_document_retriever():
    """Small chunks for search and large for context"""
    print("="*60,"Parent Document Retriever","="*60)

    long_document =Document(
        page_content="""
        Retrieval-Augmented Generation (RAG) improves the responses of Large Language Models by retrieving
        relevant information from an external knowledge base before generating an answer. A typical RAG pipeline
        includes document loading, text splitting, embedding generation, vector storage, retrieval, prompt
        construction, and answer generation.

        Chunking plays a crucial role in retrieval quality. Small chunks are more precise because each chunk
        focuses on a single idea, making it easier for vector search to find relevant information. However,
        small chunks often lose the surrounding context needed to fully understand the information.

        Parent Document Retriever solves this problem by splitting large documents into small child chunks for
        indexing while keeping the original larger parent document. During retrieval, similarity search is
        performed on the child chunks, but instead of returning only the matching child chunk, the retriever
        returns the entire parent document. This gives the language model enough surrounding context while
        maintaining accurate retrieval.

        Parent Document Retriever is especially useful for large PDFs, technical documentation, books, and
        research papers where individual paragraphs may not contain enough information on their own. It combines
        the precision of small chunks with the completeness of larger documents, leading to more informative and
        accurate answers.
        """,
        metadata={"topic": "Parent Document Retriever"}
    )

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)

    vector_store = Chroma(
        collection_name="parent_child_demo",
        embedding_function=OpenAIEmbeddings(model='text-embedding-3-small')
    )

    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    retriever.add_documents([long_document])

    query = "Why does Parent Document Retriever use child chunks?"

    child_docs = vector_store.similarity_search(query,k=1)
    print("="*30,"child chunk","="*30)
    print(f"Length is : {len(child_docs[0].page_content)}chars")
    print(f"content is : {child_docs[0].page_content[:300]}")

    parent_docs = retriever.invoke(query)
    print("="*30,"parent chunk","="*30)
    print(f"Length is : {len(parent_docs[0].page_content)}chars")
    print(f"content is : {parent_docs[0].page_content[:300]}")
    
def advance_rag_chain():
    """
    Complete RAG Chain with advanced retrieval.
    multi-query + compression + rag
    """
    
    vector_store =  create_base_vectorStore()
    llm = init_chat_model('gpt-4o-mini',temperature = 0.7)
    #multi query
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs = {'k':2}),
        llm = llm
    )
    #compressor
    compressor = LLMChainExtractor.from_llm(llm)
    advance_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_retriever
    )

    #RAG Prompt 
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based on the context. Be specific
        context: {context}
        question: {question}
        Answer:
        """
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    #Build chain
    rag_chain = {'context':advance_retriever | format_docs , 'question':RunnablePassthrough()} | prompt | llm | StrOutputParser()

    questions=[
        "What is RAG",
        "How does Basic RAG work"
    ]   
    for q in questions:
        print(f"\n Q: {q}")
        ans = rag_chain.invoke(q)
        print(f"A: {ans}")

if __name__ == "__main__":
    # multi_query_retriever()
    # contextual_compression() 
    # hybrid_search()
    # parent_document_retriever()
    advance_rag_chain()