# User Query
# "Evaluate an AI-powered fitness app."

# Agents
# Business Agent
# Market size
# Competitors
# Revenue model

# Technical Agent
# Tech stack
# Challenges
# Scalability

# Marketing Agent
# Target audience
# Marketing strategy
# User acquisition

# Synthesis Agent
# Combine everything into a business report.

from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict


from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model(
    model='gpt-4o-mini',
    temperature=0.7
)

class Startup(TypedDict):
    query:str
    business:str
    technical:str
    marketing:str
    synthesis:str

def create_startup_analytics():
    """Startup idea rating"""

    def business_agent(state: Startup)->dict:
        """Analyze the startup from a business perspective."""
        response = llm.invoke([
            SystemMessage(content="You are a business consultant.Analyze the startup from a business perspective."),
            HumanMessage(content=f"startup idea is : {state['query']}")
        ])
        return {
            'business':response.content
        }

    def technical_agent(state:Startup)->dict:
        """Analyze the startup from a technical perspective."""
        response = llm.invoke([
                    SystemMessage(content="You are a senior software architect.Analyze the technical feasibility."),
                    HumanMessage(content=f"startup idea is : {state['query']}")
                ])
        return {
            'technical':response.content
        }

    def marketing_agent(state: Startup)->dict:
        """Analyze the startup from a marketing perspective."""
        response = llm.invoke([
            SystemMessage(content='You are a marketing strategist.Analyze how to market this startup.'),
            HumanMessage(content=f'startup idea is : {state["query"]}')
        ])
        return {
            'marketing':response.content
        }

    def synthesis_agent(state: Startup)->dict:
        """Combine all three results."""
        business = state['business']
        technical = state['technical']
        marketing = state['marketing']
        response = llm.invoke([
            SystemMessage(
            content="You are an expert synthesizer. Combine multiple perspectives into coherent insights."
            ),
            HumanMessage(content=f"Combine these three reports into one {business}, {technical}, {marketing}")
        ])
        return {
            'synthesis': response.content
        }

    graph = StateGraph(Startup)

    graph.add_node('business',business_agent)
    graph.add_node('technical',technical_agent)
    graph.add_node('marketing',marketing_agent)
    graph.add_node('synthesis',synthesis_agent)

    graph.add_edge(START,'business')
    graph.add_edge(START,'technical')
    graph.add_edge(START,'marketing')
    graph.add_edge('business','synthesis')
    graph.add_edge('technical','synthesis')
    graph.add_edge('marketing','synthesis')

    app = graph.compile()

    response = app.invoke({
        'query':'Evaluate an AI-powered fitness app.',
        'business':'',
        'marketing':'',
        'technical':'',
        'synthesis':''
    })

    print(f"\n FINAL CONCLUSION: {response['synthesis']}")

if __name__ == '__main__':
    create_startup_analytics()