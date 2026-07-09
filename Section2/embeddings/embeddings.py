from langchain_openai.embeddings import OpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv
load_dotenv()
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
def simple_query():
    # simple query
    text = "This is a single text"
    embedding = embeddings.embed_query(text)
    print(f"Length of embedding is: {len(embedding)}")
    print(f"The embedding result is: {embedding}")
    #multiple texts
    embeds = embeddings.embed_documents([
        "LangChain is a framework for building LLM applications.",
        "RAG stands for Retrieval-Augmented Generation.",
        "Vector databases store embeddings."
    ])
    print(f"Length of embedding is: {len(embeds)}")
    print(f"Length of each embedding is: {len(embeds[0])}")
    print(f"The embedding result is: {embeds}")
def basic_embedding():
    #single text
    text = "What is AI and machine learning"
    single_embedding =  embeddings.embed_query(text)
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"First 5 Values: {single_embedding[:5]}")
    print(f"Vector normalization: {np.linalg.norm(single_embedding):.4f}")
def batch_embedding():
    texts= [
        "what is AI",
        "What is Machine Learning",
        "What is langchain"
    ]
    batch_emb = embeddings.embed_documents(texts)
    for i,emb in enumerate(batch_emb):
        print(f"---{i+1}-----")
        print(f"Vector Dimentions: {len(emb)}")
        print(f"First 5 values: {emb[:5]}")
        print(f"Vector normalization: {np.linalg.norm(emb):.4f}")
def similarity_search():
    docs = [
        "Python is a programming language.",
        "JavaScript is used in web development.",
        "Machine Learning enables AI applications.",
        "Deep learning uses neural networks.",
        "Cats are usually home pets."
    ]
    query = "What programming languages exist?"
    # Embed documents and query
    doc_vectors = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)
    # Cosine similarity function
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )
    # Compute similarities
    similarities = [
        cosine_similarity(query_vector, doc_vector)
        for doc_vector in doc_vectors
    ]
    # Pair documents with scores
    results = list(zip(docs, similarities))
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    # Print results
    print(f"Query: {query}\n")
    for i, (doc, score) in enumerate(results, start=1):
        print(f"Rank {i}")
        print(f"Document : {doc}")
        print(f"Similarity: {score:.4f}")
        print("-" * 40)
def free_models():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model = "BAAI/bge-small-en-v1.5")
    text = "This is a single text"
    embedding = embeddings.embed_query(text)
    print(f"Length of embedding is: {len(embedding)}")
    print(f"The embedding result is: {embedding}")
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    vector = embeddings.embed_query("What is LangChain?")
    print(len(vector))
def exercise():
    docs = [
    "Python is a programming language.",
    "Machine Learning is a subset of Artificial Intelligence.",
    "Deep Learning uses neural networks.",
    "LangChain helps build LLM applications.",
    "Cats are common household pets."
    ]
    embedding_docs = embeddings.embed_documents(docs)
    print(f"Total document embedding is : {len(embedding_docs)}")
    print(f"Length of one embedding is : {len(embedding_docs[0])}")
    for i,emb in enumerate(embedding_docs):
        print(f"for {i} first ten values are {emb[:10]}")
    query = 'Tell me about python'
    embedding_query = embeddings.embed_query(query)
    print(f"Total query embedding is : {len(embedding_query)}")
    print(f"First ten values are {embedding_query[:10]}")
    print({len(embedding_docs[0]) == len(embedding_query)})

if __name__ =='__main__':
    # simple_query()
    # free_models()
    # basic_embedding()
    # batch_embedding()
    # similarity_search()
    exercise()