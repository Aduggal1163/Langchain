from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
# embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
# #simple query
# text = "This is a single text"
# embedding = embeddings.embed_query(text)
# print(f"Length of embedding is: {len(embedding)}")
# print(f"The embedding result is: {embedding}")
# #multiple texts
# embeds = embeddings.embed_documents([
#     "LangChain is a framework for building LLM applications.",
#     "RAG stands for Retrieval-Augmented Generation.",
#     "Vector databases store embeddings."
# ])
# print(f"Length of embedding is: {len(embeds)}")
# print(f"Length of each embedding is: {len(embeds[0])}")
# print(f"The embedding result is: {embeds}")
# ---------FREE------------
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