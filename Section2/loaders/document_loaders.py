# import os
# import tempfile
# from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader,WebBaseLoader,PyPDFLoader
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
if __name__ == '__main__':
    load_text_file()
    web_loader()
    doc_structure()
    pdf_loader("langchain.pdf")