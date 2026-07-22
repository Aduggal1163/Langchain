from langgraph.graph import StateGraph,START,END
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage
from typing_extensions import TypedDict,Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

import operator
import sqlite3

from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model('gpt-4o-mini',temperature=0.0)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],operator.add]

def memory_saver():
    """In memory checkpointing for developing"""
    def chat(state: ChatState)->dict:
        response = llm.invoke(state['messages'])
        return {
            'messages':[response]
        }
    
    graph = StateGraph(ChatState)
    graph.add_node('chat',chat)
    graph.add_edge(START,'chat')
    graph.add_edge('chat',END)

    memory = MemorySaver()
    app = graph.compile(
        checkpointer=memory
    )
    
    config = {
        'configurable': {
            'thread_id':'user123'
        }
    }

    result = app.invoke(
        {
            'messages':[HumanMessage(content='My name is Abhishek')]
        },
        config
    )

    print(f"Turn 1: AI: {result['messages'][-1].content}")

    result = app.invoke(
        {
            'messages':[HumanMessage(content='What is my name')]
        },
        config
    )

    print(f"Turn 2 - AI: {result['messages'][-1].content}")

    #CHECK FULL HISTORY
    state = app.get_state(config)
    print(f"\n Total messages in state: {len(state.values['messages'])}")

def sqlite_persistence():
    """Sqlite for durable storages"""
    def chat(state: ChatState)->dict:
        response = llm.invoke(state['messages'])
        return {
            'messages':[response]
        }
    
    graph = StateGraph(ChatState)
    graph.add_node('chat',chat)
    graph.add_edge(START,'chat')
    graph.add_edge('chat',END)

    db_path = './sqlite_db'
    print(f"\nSQLite persistence Demo")
    print(f"\nDatabase path: {db_path}")

     # Create SQLite connection
    conn = sqlite3.connect(db_path, check_same_thread=False)

    # Create SQLite checkpointer
    memory = SqliteSaver(conn)

    # Compile graph
    app = graph.compile(checkpointer=memory)

    config = {
        "configurable": {
            "thread_id": "user123"
        }
    }

    # First conversation
    result = app.invoke(
        {
            "messages": [
                HumanMessage(content="My name is Abhishek")
            ]
        },
        config=config
    )

    print(f"Turn 1 - AI: {result['messages'][-1].content}")

    # Second conversation
    result = app.invoke(
        {
            "messages": [
                HumanMessage(content="What is my name?")
            ]
        },
        config=config
    )

    print(f"Turn 2 - AI: {result['messages'][-1].content}")

    # Check stored state
    state = app.get_state(config)
    print(f"\nTotal messages in state: {len(state.values['messages'])}")

    conn.close()

if __name__ == '__main__':
    memory_saver()
    print("="*50)
    sqlite_persistence()