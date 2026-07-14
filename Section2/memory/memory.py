"""
Conversation Memory in LangChain

modern approaches to maintaining conversation context

"""

from langchain.chat_models import init_chat_model

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from langchain_core.messages import HumanMessage,SystemMessage,AIMessage,trim_messages

from langchain_core.output_parsers import StrOutputParser

from typing import Dict

from dotenv import load_dotenv

from langchain_core.chat_history import InMemoryChatMessageHistory,BaseChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

llm = init_chat_model('gpt-4o-mini',temperature = 0.3)

def basic_memory():
    """Basic conversation with RunnableWithMessageHistory"""

    print("="*50)
    print("Basic conversation memory")
    print("="*50)

    #prompt with history placeholder

    prompt = ChatPromptTemplate.from_messages([
        ('system','You are helpful assistant. Be concise'),
        MessagesPlaceholder(variable_name='history'),
        ('human','{input}')
    ])

    chain = prompt | llm | StrOutputParser()
    
    #Session store

    store: Dict[str,InMemoryChatMessageHistory] = {}

    def get_session_history(session_id : str)->BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    #Wrap with history

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key='input',
        history_messages_key='history'
    )

    #Configured for this session

    config = {
        'configurable': {'session_id':'user_123'}
    }

    #conversation

    messages = [
        "hi my name is abhishek duggal",
        "im learning Langchain",
        "what is my name and what am i learning"
    ]

    print("\n conversation")
    
    for msg in messages:
        print(f"User: {msg}")
        ans = chain_with_history.invoke({'input':msg},config=config)
        print(f"AI: {ans}")

    #Show stored history
    print(f"\n ----stored history {len(store['user_123'].messages)} messages----")
    for msg in store['user_123'].messages:
        role = "human"if isinstance (msg,HumanMessage) else 'AI'
        print(f"{role}: {msg.content[:50]}....")

def multi_session():
    print("="*60)
    print(f"Multi Conversation Session")
    print("Each user gets its own memory")
    print("="*60)
    prompt = ChatPromptTemplate.from_messages(
    [    
        ('system',"Im your helpful ai assistant"),
        MessagesPlaceholder(variable_name='history'),
        ('human','{text}')
    ]
    )
    store : Dict[str:InMemoryChatMessageHistory]={}
    def get_session_history(session_id : str)->BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    chain_with_history = RunnableWithMessageHistory(
        prompt | llm | StrOutputParser(),
        get_session_history,
        input_messages_key='text',
        history_messages_key='history'    
    )
    config_a = {
        'configurable':{
            'session_id' : 'user_a_123'
        }
    }
    config_b = {
        'configurable':{
            'session_id' : 'user_b_123'
        }
    }
    # print('\n User A')
    # print('\n UserA: My fav lang is Python')
    # res = chain_with_history.invoke({'text':"My fav lang is Python"},config=config_a)
    # print(f"AI: {res}")
    # print('\n User B')
    # print('\n UserB: My fav lang is Java')
    # res = chain_with_history.invoke({'text':"My fav lang is Java"},config=config_b)
    # print(f"AI: {res}")

    # print("="*60 , "Asking question", "="*60)
    # print('\n UserA: Which is myy fav lang?')
    # res = chain_with_history.invoke({'text':"Which is my fav lang?"},config=config_a)
    # print(f"AI (A):{res}")
    # print('\n UserB: Which is myy fav lang?')
    # res = chain_with_history.invoke({'text':"Which is my fav lang?"},config=config_b)
    # print(f"AI (B):{res}")
    conversations = [
    ("User A", config_a, "My fav lang is Python"),
    ("User B", config_b, "My fav lang is Java"),
    ("User A", config_a, "Which is my fav lang?"),
    ("User B", config_b, "Which is my fav lang?")
    ]
    for user, config, msg in conversations:
        print(f"\n{user}: {msg}")
        ans = chain_with_history.invoke(
        {"text": msg},
        config=config
        )
        print(f"AI: {ans}")

def message_trimming():
    """Trim message to fit context window"""

    print("="*60)
    print("Message Trimming")
    print("Keep message within your token limit")
    print("="*60)
    
    conversation = [

    SystemMessage(
        content="You are a helpful AI assistant. Answer concisely."
    ),

    HumanMessage(content="Hi!"),
    AIMessage(content="Hello! How can I help you today?"),

    HumanMessage(content="My name is Abhishek Duggal."),
    AIMessage(content="Nice to meet you, Abhishek."),

    HumanMessage(content="I'm learning LangChain."),
    AIMessage(content="That's great! LangChain is useful for building LLM applications."),

    HumanMessage(content="I also know Java."),
    AIMessage(content="Java is an excellent language for DSA and backend development."),

    HumanMessage(content="I'm currently learning RAG."),
    AIMessage(content="RAG combines retrieval with language models to answer using external knowledge."),

    HumanMessage(content="Can you explain MultiQueryRetriever?"),
    AIMessage(content="It generates multiple variations of your query to improve document retrieval."),

    HumanMessage(content="Now I'm studying Conversation Memory."),
    AIMessage(content="Conversation Memory allows an LLM to remember previous interactions."),

    HumanMessage(content="What's my name?"),
    ]

    print(f"Original length of conversation: {len(conversation)} message(s)")

    trimmed = trim_messages(
        messages=conversation,
        max_tokens=80,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False
    )

    print(f"Trimmed convo: {len(trimmed)} message(s)")

    for msg in trimmed:
        print(f"{type(msg).__name__}: {msg.content}")

def windowed_memory():
    """Implement sliding window manually"""
    print("="*60)
    print("Window Memory (Keep last K)")
    print("Fixed size conversation window")
    print("="*60)
    window_size = 6
    prompt = ChatPromptTemplate.from_messages([
        ('system','You are helpful AI Assistant'),
        MessagesPlaceholder(variable_name='history'),
        ('human','{text}')
    ])
    store : Dict[str,InMemoryChatMessageHistory] = {}
    def get_session_history(session_id : str)->BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    chain_with_history = RunnableWithMessageHistory(
        prompt | llm | StrOutputParser(),
        get_session_history,
        input_messages_key='text',
        history_messages_key='history'
    )
    config = {
        'configurable':{
            'session_id':'user_123'
        }
    }
    messages = [
        "Hi",
        "My name is Abhishek Duggal",
        "I am learning LangChain",
        "I know Java",
        "Explain RAG",
        "Explain MultiQueryRetriever",
        "Explain Contextual Compression",
        "Explain Conversation Memory",
        "What is my name?",
        "What am I learning?"
    ]
    for msg in messages:
        print(f"User: {msg}")
        res = chain_with_history.invoke({'text':msg},config=config)
        print(f"AI: {res}")
        history = store['user_123']
        if(len(history.messages) > window_size):
            history.messages = history.messages[-window_size:]
        print(f"Current memory length is : {len(history.messages)} message(s)")
        for hm in history.messages:
            role = 'human' if isinstance(hm,HumanMessage) else 'AI'
            print(f"role: {role} and content is: {hm.content}")

def summary_memory():

    """
    Implement conversational summary
    """
    
    print("="*60)
    print("Summarize older message to save tokens")
    print("="*60)
    
    conversation = """
    User introduced themselves as Abhi, an AI engineer from Seattle.
    
    User asked about LangChain and learned it's a framework for LLM apps.
    
    User asked about memory types: buffer, window, and summary memory.
    
    User expressed interest in building a chatbot with persistent memory.
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [('system',"""You are a helpful AI Assitant.Here's the summary of the conversation
        {summary}
        Use this context to maintain continuitys"""),
        MessagesPlaceholder(variable_name='history'),
        ('human','{text}')]
    )
    
    chain = prompt | llm | StrOutputParser()
    
    print(f"\n using summary so far")
    print(f"\n {conversation[:100]}....")
    
    res=chain.invoke({
        'summary':conversation,
        'history':[],
        'text':"What should i build next based on the convo"
    })
    
    print(res)
    print('\n---generating summary----')
    
    summary_prompt = ChatPromptTemplate.from_template(
        """
            Summarize this in 2 lines maintaing the user's key facts:
            {convo}
        """
    )
    
    summary_chain = summary_prompt | llm |StrOutputParser()
    
    updated_summary = summary_chain.invoke({
        'convo': conversation + f"\n Assistant: {res}"
    })
    
    print("\n Updated summary: ")

    print(updated_summary)

def exercise():
    """
     Build a chatbot with
     Persistent memory (SQLite)
     Automatic summary after 10 messages
     User pref tracking
    """

    print("="*60)
    print("Persistent memory chatbot")
    print("="*60)
    
    from langchain_community.chat_message_histories import SQLChatMessageHistory
    import os

    #use sqlite for persistent
    db_path = "./chat_history.db"
    
    def get_session_history(session_id : str)->BaseChatMessageHistory:

        return SQLChatMessageHistory(
            session_id=session_id,

            connection=f"sqlite:///{db_path}"

        )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system','you are a helpful asssistent also remember users prefrence'),

            MessagesPlaceholder(variable_name='history'),
            
            ('human','{input}')
        ]
    )

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        
        get_session_history,
        
        input_messages_key='input',
        
        history_messages_key='history'
    )

    config = {
        'configurable':{
            'session_id':'persistent_user'
        }
    }

    print('\nPersistent Memory Chatbot; ')

    print("Message Saved to SQLite database\n")

    #Test Conversations

    messages = [
        "Remember i love Dark themes",
        
        "What themes do i prefer?"
    ]

    for msg in messages:

        print(f"User: {msg}")
        
        res = chain_with_history.invoke({'input':msg},config=config)
        
        print(f"AI : {res}\n")

    print(f"Database Created: {db_path}")

    print("Message persist across restarts!")

    #cleanup for demo

    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == '__main__':
    # basic_memory()
    # multi_session()
    # message_trimming()
    # windowed_memory()a
    # summary_memory()
    exercise()