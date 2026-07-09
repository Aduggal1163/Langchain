from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader,WebBaseLoader,PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
load_dotenv()
def load_text_file():
    try:
        #Load text file using TextLoader
        loader = TextLoader('demo.txt')
        document = loader.load()
        # for doc in document:
        #     print(doc.page_content)
        print(f"Loaded {len(document)} document(s)")
        print(f"Content: {document[0].page_content[:100]}....")
        print(f"Metadata: {document[0].metadata}")
    finally:
        print("-"*50)
#---------------------
#------------
def web_loader():
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Java")
    document = loader.load()
    print(document[0].page_content[:100])
    print("-"*50)
#----------------
#------------
def doc_structure():
    doc = Document(
        page_content="Hi everyone, My name is Abhishek Duggal",
        metadata = {
            "author":"Abhishek Duggal",
            "source":"manual_creation.txt",
            "created_at":"2026-08-06"
        }
    )
    print("Document Structure")
    print(f"Page type is : {type(doc.page_content)}")
    print(f"Page content is : {doc.page_content}")
    print(f"metadata: {doc.metadata}")
    #Documents are immutable but you can always creates a new one
    updated_doc = Document(
        page_content=doc.page_content + " some Additional Content",
        metadata = {**doc.metadata,"updated":True}
    )
    print(f"Updated content : {updated_doc.page_content}")
#-------------
#------------
def pdf_loader(pdf_path : str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    print(f"Found {len(document)} document(s)")
    for i,doc in enumerate(document):
        print(f"\nDocument {i+1} content review: {doc.page_content}")
        print(f"\nDocument metadata : {doc.metadata}")
#------------
#------------
def exercise():
    loader = TextLoader('demo.txt')
    doc = loader.load()
    print(f"\nLength of document is : {len(doc)}\n")
    print(f"Content is : {doc[0].page_content}\n")
    print(f"Metadata is : {doc[0].metadata}\n")
    model = init_chat_model('gpt-4o-mini',temperature = 1)
    summary_prompt = ChatPromptTemplate.from_template("Generate me a summary of {text}")
    keyword_prompt = ChatPromptTemplate.from_template("Give me 5 keywords in bullet points from {text}")
    analyzed_chain = RunnableParallel(
    summary_chain = summary_prompt | model | StrOutputParser(),
    keyword_chain = keyword_prompt | model | StrOutputParser()
    )
    content = doc[0].page_content
    # print(type(content))
    result = analyzed_chain.invoke({'text':content})
    print("="*50,"Generating summary","="*50) 
    print(result['summary_chain'])
    print("="*50,"Generating keywords","="*50) 
    print(result['keyword_chain'])
#------------
#------------
if __name__ == '__main__':
    # load_text_file()
    # web_loader()
    # doc_structure()
    # pdf_loader("langchain.pdf")
    exercise()