"""
Langsmith setup and observability
Production monitoring for LangChain/LangGraph
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langsmith.run_trees import RunTree
from dotenv import load_dotenv

load_dotenv()
os.environ['LANGSMITH_TRACING'] = 'true'

@traceable(name='basic_chaining', tags = ['production','explanation'])
def demo_basic_tracing():
    """Basic LangSmith tracing"""
    llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)
    prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
    chain = prompt | llm | StrOutputParser()
    print('Basic Tracing Demo\n')
    print("Running chian with LangSmith enabled\n")
    result = chain.invoke({'topic':'machine learning'})
    print(f"Result: {result}")
    print('\n Check LangSmith dashboard for trace details.')
if __name__ == '__main__':
    demo_basic_tracing()