from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage,SystemMessage
from langgraph.graph import StateGraph, START, END,add_messages
from typing_extensions import Annotated,TypedDict
import operator

from dotenv import load_dotenv
load_dotenv()

class ConversationState(TypedDict):
    messages: Annotated[list,operator.add]
    sentiment: str
    response_count: int

def create_conversation_graph():
    llm = init_chat_model('gpt-4o-mini',temperature=0)
    
    #Define node fncs
    def analyze_sentiment(state: ConversationState)->dict:
        """
        Analyze the sentiment of function
        """
        last_message = state['messages'][-1]
        response = llm.invoke([
            SystemMessage(content='Classify this message as sentiment (positive,negative or neutral)'),
            HumanMessage(content=last_message)
        ])
        return {
            'sentiment':response.content.lower().strip()
        }
    
    def generate_response(state: ConversationState)->dict:
        """
        Generate appropriate response based on the sentiment
        """
        sentiment = state['sentiment']
        last_message = state['messages'][-1]
        system_prompts = {
            "positive": """
You are a friendly and enthusiastic assistant.

The user's message expresses positive sentiment.
Respond warmly, acknowledge their positivity, and encourage the conversation.
Keep the tone upbeat and appreciative.
""",

    "negative": """
You are a calm, empathetic, and supportive assistant.

The user's message expresses negative sentiment.
Acknowledge their feelings, show understanding, and provide helpful or constructive guidance.
Do not dismiss or exaggerate their emotions.
""",

    "neutral": """
You are a professional and informative assistant.

The user's message is neutral and factual.
Answer clearly, directly, and concisely.
Avoid adding unnecessary emotional language.
"""
        }
        prompt = system_prompts.get(sentiment,system_prompts["neutral"])
        response = llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=last_message)
        ]
        )

        return {
            'messages':[f"AI: {response.content}"],
            'response_count':1
        }

    #create graph    
    graph = StateGraph(ConversationState)

    #add nodes
    graph.add_node('analyze_sentiment',analyze_sentiment)
    graph.add_node('generate_response',generate_response)

    graph.add_edge(START,'analyze_sentiment')
    graph.add_edge('analyze_sentiment','generate_response')
    graph.add_edge('generate_response',END)

    #compile graph
    app = graph.compile()

    #run app
    return app

def conversation():
    app = create_conversation_graph()
    test_messages = [
    # Positive
    "I got promoted today!",
    "Thank you so much for your help.",
    "This is the best service I've ever used.",
    "I'm really excited about my new project.",
    "Everything worked perfectly!",

    # Negative
    "I'm very disappointed with the product.",
    "My order hasn't arrived yet.",
    "This app keeps crashing.",
    "I'm frustrated because nothing is working.",
    "I want a refund immediately.",

    # Neutral
    "What is the weather today?",
    "How do I reset my password?",
    "Where is your office located?",
    "Can you explain what Python is?",
    "What are your business hours?"
]
    for msg in test_messages:
        result = app.invoke({
            "messages":[msg],
            "sentiment":"",
            "response_count":0
        })
        print(f"Input: {msg}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Response: {result['messages'][-1]}")
        print("-"*40)


if __name__ == '__main__':
    conversation()