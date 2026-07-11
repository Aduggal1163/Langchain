from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel,Field
from typing import List
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
KNOWLEDGE_BASE = """ LangChain is an open-source framework for building applications powered by LLMs. It provides components like prompt templates, chains, retrievers, memory, and agents. LangGraph is used to build stateful, multi-agent workflows."""
load_dotenv()
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
def create_kb():
    """ Create a vector store from knowledge base."""
    #Split knowledge base into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    doc = Document(
        page_content=KNOWLEDGE_BASE,
        metadata = {
            "sources":"knowledge_base.md"
        }
    )
    chunks = splitter.split_documents([doc])

    #create a vector store from chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory="./rag_db"
    )

    return vector_store

llm = init_chat_model(
        model='gpt-4o-mini',
        model_provider='openai',
        temperature = 0
    )

def basic_rag():
    vector_store = create_kb()

    retriever = vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {'k':2}
    )

    #RAG Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based on the given following context.
        {context}
        Question : {question}
        Answer :
        Make sure to ans in concise manner and if you dont know the ans just say MAINU IN PTA VEERE
        """
    )

    #format retrieved docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    #format_docs() converts the list of retrieved Document objects into a single string that can be inserted into the prompt.

    #RAG Chain
    rag_chain = (
        {'context': retriever | format_docs, 'question':RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser() 
    )

    # Test
    questions = [
        "What is langchain",
        "Who created langchain",
        "What is langGraph used for"
    ]

    print("Basic RAG working")
    for q in questions:
        ans = rag_chain.invoke(q)
        print(f"Q : {q}")
        print(f"A : {ans}")

def rag_sources():
    vector_store = create_kb()
    retriever = vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {'k':2}
    )
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based on the given following context.
        {context}
        Question : {question}
        Answer (include sources):
        Make sure to ans in concise manner and if you dont know the ans just say MAINU IN PTA VEERE
        """
    )
    def format_docs_with_sources(docs):
        formatted = []
        for i,doc in enumerate(docs):
            sources = doc.metadata.get('sources','unknown')
            formatted.append(f"[{i+1}] sources  {sources} : \n {doc.page_content}")
        return "\n\n".join(formatted)
    
    rag_chain = (
        {"context": retriever | format_docs_with_sources , 'question': RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    print("RAG with sources")
    ans = rag_chain.invoke("What are core components of Langchain")
    print(f"Q: What are core components of Langchain")
    print(f"A: {ans}")

def rag_fallback():
    vector_store = create_kb()

    retriever = vector_store.as_retriever(
        search_type = 'similarity_score_threshold',
        
        search_kwargs = { "score_threshold": 0.7,'k':2}
    )

    prompt =ChatPromptTemplate.from_template("""
    Answer the following question based on the given following context.
        {context}
        Question : {question}
        Answer :
    """
    )
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {'context': retriever | format_docs ,'question': RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )

    question = [
         "What is langchain",
        "Who created langchain",
        "What is langGraph used for"
    ]

    print('RAG with Fallbacks: ')
    for q in question:
        print(f"Q: {q}")
        docs = retriever.invoke(q)
        if not docs:
            print("Using fallback LLM...")
            ans = llm.invoke(q).content
            print(f"Ans : {ans[:100]}")
        else :
            print("USING RAG")
            ans = rag_chain.invoke(q)
            print(f"Ans : {ans}")
        
def rag_with_structure():
    """RAG with structured outputs"""
    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_kwargs = {'k':3})
    class RAGResponse(BaseModel):
        """Structured RAG Response"""
        answer : str = Field(description="The answer to the question")
        confidence : str = Field(description="high,medium or low")
        sources_used : List[str] = Field(description="sources used")
        followup_question : str = Field(description="suggest follow up questions user may asks")

    structured_llm = llm.with_structured_output(RAGResponse)

    prompt = ChatPromptTemplate.from_template(
        """
        Based on the context below, ans the following questions
        Context: {context}
        Question : {question}
        Provide a structured responses
        """
    )

    def format_docs_with_sources(docs):
        formatted = []
        for i,doc in enumerate(docs):
            sources = doc.metadata.get('sources','unknown')
            formatted.append(f"[{i+1}] sources  {sources} : \n {doc.page_content}")
        return "\n\n".join(formatted)

    rag_chain = (
        {
            'context': retriever | format_docs_with_sources, 'question': RunnablePassthrough()
        }
        | prompt
        | structured_llm
    )

    print("Structured RAG demo")

    result = rag_chain.invoke("Define LangGraph")

    print(f"\nanswer: {result.answer}")
    print(f"\nconfidence: {result.confidence}")
    print(f"\nsources_used: {result.sources_used}")
    print(f"\nfollowup_question: {result.followup_question}")

#Build a document QA system
def document_qa():
    """
    Build a complete document QA system that:
    Takes a text document as input
    Splits and embeds it
    Allows multiple questions
    Returns ans with confidence score
    """

    class QAResponse(BaseModel):
        answer : str = Field(description="Answer to your question")
        confidence : str = Field(description="High,Med or Low")
    
    vector_store = create_kb()
    retriever = vector_store.as_retriever(
        search_kwargs = {'k':3} 
    )
    structured_llm = llm.with_structured_output(QAResponse)

    def format_doc(docs):
        return '\n\n'.join([doc.page_content for doc in docs])
    
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based on the context
        context : {context}
        question : {question}

        give ans with the proper structural format
        if you dont know answer please type i dont have prior information regarding this
        """
    )

    rag_chain = {'context':retriever |format_doc, 'question':RunnablePassthrough()} | prompt | structured_llm

    while True:
        question = input("Ask Question: ")
        if question == 'exit':
           break
        response = rag_chain.invoke(question)
        print("Confidence: ",response.confidence)
        print("Answer: ",response.answer)

if __name__ == '__main__':
    # basic_rag()
    # rag_sources()
    # rag_fallback()
    # rag_with_structure()
    document_qa()