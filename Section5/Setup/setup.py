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

@traceable(name='trace_with_metadata_demo', tags =['metadata','filtering'])
def demo_trace_with_metadata(user_id: str, request_type: str):
    """Add metadata for traces to filtering."""

    llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)
    result = llm.invoke(f"Hello from {user_id}")
    print(request_type," Result is: ",result.content)
    return result.content

if __name__ == '__main__':
    demo_trace_with_metadata(user_id = 'user_123', request_type = 'greetings')
    