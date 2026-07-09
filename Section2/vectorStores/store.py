from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

#sample documents
SAMPLE_DOCS = [
    Document(
        page_content="Python is a high-level programming language used for web development, AI, and automation.",
        metadata={
            "source": "python_guide.pdf",
            "topic": "Programming",
            "page": 1
        }
    ),
    Document(
        page_content="Machine Learning enables computers to learn patterns from data without being explicitly programmed.",
        metadata={
            "source": "ml_notes.pdf",
            "topic": "Artificial Intelligence",
            "page": 5
        }
    ),
    Document(
        page_content="LangChain is a framework for building applications powered by large language models.",
        metadata={
            "source": "langchain_docs",
            "topic": "LLMs",
            "page": 2
        }
    ),
    Document(
        page_content="Chroma is an open-source vector database used to store and search embeddings efficiently.",
        metadata={
            "source": "chroma_docs",
            "topic": "Vector Databases",
            "page": 3
        }
    ),
    Document(
        page_content="Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        metadata={
            "source": "deep_learning_book",
            "topic": "AI",
            "page": 12
        }
    )
]

def chroma_basics():
    # create vector store for documents
    vector_store = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings,
        persist_directory='./chroma_db'
    )

    print(f"===+Vector Store created {vector_store._collection.count()} and perssisted+======")

    #perform similarity search
    query = "What is Langchain"
    results = vector_store.similarity_search(query,k=2)
    print(f"Top 2 search for query {query} is:")
    for i,res in enumerate(results,start=1):
        print(f"result for {i} is {res.page_content}")

def similarity_search_with_score():
    vector_store = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings,
        persist_directory='./chroma_db'
    )
    #perform similarity search with score
    query = "What is Langchain"
    result_with_score = vector_store.similarity_search_with_score(query,k=2)
    print("======================================================================================================="*1)
    print(f"Top 2 search for query {query} & score is:")
    for i,(res,score) in enumerate(result_with_score,start=1):
        print(f"result for {i} is {res.page_content} that has score => {score:.4f}")

def filtering_search():
    vector_store = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings,
        persist_directory='./chroma_db'
    )
    text = "What databases are available"
    result = vector_store.similarity_search(text);
    print("="*100)
    print(f"===+Vector Store created {vector_store._collection.count()} and perssisted+======")
    for i,res in enumerate(result,start=1):
        print(f"{i} result is {res.page_content}")
    print("="*20,"With filtering","="*25)

    filtering_criteria = {"topic":"vector databases"}
    filter_result = vector_store.similarity_search(
        text,
        filter=filtering_criteria,
        k=4
    )
    for i,res in enumerate(filter_result,start=1):
        print(f"{i} result is {res.page_content}")
        
def persist_Chroma():
    # This function demonstrates Chroma's persistence feature. 
    # It shows that after storing embeddings on disk, you can "restart" your application
    # and still search the same vector database without re-embedding the documents.
    #We use persistent vector stores because generating embeddings is expensive. You don't want to recompute them every time your application starts.
    persist_dir = "./chroma_db"

    vector_store = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    original_count = vector_store._collection.count()

    print(f"Persisted vector store with original count is {original_count} document(s)")
    print(f"Vector store persisted at: {persist_dir}")

    #simulate restart
    
    del vector_store
    reloaded  = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    reloaded_count = reloaded._collection.count()
    print(f"Persisted vector store with reloaded count is {reloaded_count} document(s)")

    #verify search still works
    results = reloaded.similarity_search("Langchain",k=2)
    for i,res in enumerate(results,start=1):
        print(f"{i} result is {res.page_content}")

def as_retriver():
    vector_store = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    #basic reteriver usage
    retriver = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs={'k':3}
    )
    #use retriver to get related documents
    docs = retriver.invoke("How do i build AI applications")
    print("Retriving results:")
    for i,doc in enumerate(docs,start=1):
        print(f"Result {i} content {doc.page_content}")

    #maximum marginal relivence
    mmr_retriver = vector_store.as_retriever(
        search_type = 'mmr',
        search_kwargs = {'k':3 , 'fetch_k':5}
    )
    mmr_docs = mmr_retriver.invoke("How do i build AI applications")
    print("retriving MMR results: ")
    for i,mdoc in enumerate(mmr_docs,start=1):
        print(f"Result {i} content {mdoc.page_content}")

def exercise():
    """
    Create a complete vector setup that:
    takes a list of strings
    split them into chunks
    stores in chroma
    returns a configured retriver
    """
    SAMPLE = [
    Document(
        page_content="Artificial Intelligence enables machines to perform tasks that normally require human intelligence, such as reasoning, learning, and decision-making.",
        metadata={
            "source": "ai_handbook.pdf",
            "topic": "Artificial Intelligence",
            "page": 3
        }
    ),
    Document(
        page_content="Machine Learning is a branch of Artificial Intelligence that allows systems to learn from data without explicit programming.",
        metadata={
            "source": "ml_notes.pdf",
            "topic": "Machine Learning",
            "page": 7
        }
    ),
    Document(
        page_content="Deep Learning is a subset of Machine Learning that uses multi-layer neural networks to solve complex problems such as image recognition and natural language processing.",
        metadata={
            "source": "deep_learning.pdf",
            "topic": "Deep Learning",
            "page": 12
        }
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) combines vector search with large language models to generate accurate answers using external knowledge.",
        metadata={
            "source": "rag_guide.pdf",
            "topic": "RAG",
            "page": 5
        }
    ),
    Document(
        page_content="LangChain is an open-source framework for developing applications powered by large language models, offering chains, agents, memory, and retrieval components.",
        metadata={
            "source": "langchain_docs",
            "topic": "LangChain",
            "page": 2
        }
    ),
    Document(
        page_content="Chroma is an open-source vector database designed for storing embeddings and performing efficient similarity searches.",
        metadata={
            "source": "chroma_docs",
            "topic": "Vector Database",
            "page": 4
        }
    ),
    Document(
        page_content="FAISS is a similarity search library developed by Meta that enables efficient nearest-neighbor search over high-dimensional vectors.",
        metadata={
            "source": "faiss_docs",
            "topic": "Vector Database",
            "page": 8
        }
    ),
    Document(
        page_content="Cloud computing provides on-demand access to computing resources such as servers, storage, networking, and databases over the internet.",
        metadata={
            "source": "cloud_fundamentals.pdf",
            "topic": "Cloud Computing",
            "page": 10
        }
    ),
    Document(
        page_content="SQL databases store structured data in tables using rows and columns, making them suitable for transactional applications.",
        metadata={
            "source": "database_basics.pdf",
            "topic": "Databases",
            "page": 15
        }
    ),
    Document(
        page_content="NoSQL databases provide flexible schemas and are commonly used for handling large volumes of unstructured or semi-structured data.",
        metadata={
            "source": "nosql_guide.pdf",
            "topic": "Databases",
            "page": 18
            }
        )
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 50
    )
    text = splitter.split_documents(SAMPLE)

    vector_store = Chroma.from_documents(
        documents=text,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    mmr_retriver = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k':3,'fetch_k':5}
    )

    result = mmr_retriver.invoke("What is Langchain")
    for i,res in enumerate(result,start=1):
        print(f"for result {i} result is : {res.page_content}")


if __name__ == '__main__':
    # chroma_basics()
    # similarity_search_with_score()
    # filtering_search()
    # persist_Chroma()
    # as_retriver()
    exercise()