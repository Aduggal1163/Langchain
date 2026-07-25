from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage,SystemMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict,Annotated
from typing import Literal
from pydantic import BaseModel,Field
from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model(
    model='gpt-4o-mini',
    temperature=0.0
)

#================================
#Shared state schema used across all levels
#================================
class TeamState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
    final_ans: str
    department: str

#================================
#Dept1: Research Dept(subgraph)
#================================
def build_research_team()->StateGraph:
    """Build the research dept subgraph"""
    def web_researcher(state: TeamState)->dict:
        """Searches the web for information."""
        query = ""
        for msg in reversed(state['messages']):
            if isinstance(msg,HumanMessage):
                query = msg.content
                break
        response = llm.invoke([
            SystemMessage(content="You are a web researcher, find key facts about this topic and provide 2 bullet points "),
            HumanMessage(content=query)
        ])

        return {
            'messages':[
                AIMessage(
                    content=response.content,
                    name = "web_researcher"
                )
            ]
        }

    def paper_reviewer(state: TeamState)->dict:
        """Reviews academic/technical sources."""
        query = ""
        for msg in reversed(state['messages']):
            if isinstance(msg,HumanMessage):
                query = msg.content
                break
        response = llm.invoke([
            SystemMessage(content="You are an academic reviewer, provide technical depth about this topic and provide 2 bullet points "),
            HumanMessage(content=query)
        ])

        return {
            'messages':[
                AIMessage(
                    content=response.content,
                    name = "paper_reviewer"
                )
            ]
        }

    def research_lead(state: TeamState)->dict:
            """Synthesizes findings from both researchers."""
            response = llm.invoke([
                SystemMessage(content="You are a research lead. Synthesize the web researcher's and paper reviewer's findings into a cohesive research brief. Keep it to one short paragraphs")
            ]+ state['messages'])
    
            return {
                'messages':[
                    AIMessage(
                        content=response.content,
                        name = "research_lead"
                    )
                ],
                'final_ans':response.content
            }

    #Build the research subgraph
    research_graph = StateGraph(TeamState)

    research_graph.add_node('web',web_researcher)
    research_graph.add_node('paper',paper_reviewer)
    research_graph.add_node('lead',research_lead)

    #Fan-out: both researchers work in parallel
    research_graph.add_edge(START,'web')
    research_graph.add_edge(START,'paper')

    #Fan-in: both feed into research lead
    research_graph.add_edge('web','lead')
    research_graph.add_edge('paper','lead')

    research_graph.add_edge('lead',END)

    return research_graph

#================================
#Dept2: Content Team(subgraph)
#================================
def build_content_team()->StateGraph:
    """Build the content department subgraph."""
    def content_writer(state: TeamState) -> dict:
        """Writes content using the research brief."""

        response = llm.invoke([
            SystemMessage(
                content=(
                    "You are a professional content writer. "
                    "Using the research brief in the conversation, "
                    "write a clear and engaging article of about 150 words."
                )
            )
        ] + state["messages"])

        return {
            "messages": [
                AIMessage(
                    content=response.content,
                    name="content_writer"
                )
            ]
        }

    def content_editor(state: TeamState) -> dict:
        """Edits and polishes the written content."""

        response = llm.invoke([
            SystemMessage(
                content=(
                    "You are a senior editor. "
                    "Read the conversation, especially the content writer's draft. "
                    "Improve grammar, clarity, flow, and readability. "
                    "Return only the final polished article."
                )
            )
        ] + state["messages"])

        return {
            "messages": [
                AIMessage(
                    content=response.content,
                    name="content_editor"
                )
            ],
            "final_ans": response.content
        }

    # Build content subgraph
    content_graph = StateGraph(TeamState)

    content_graph.add_node("writer", content_writer)
    content_graph.add_node("editor", content_editor)

    content_graph.add_edge(START, "writer")
    content_graph.add_edge("writer", "editor")
    content_graph.add_edge("editor", END)

    return content_graph

#================================
#Dept3: Analysis Team(subgraph)
#================================
def build_analysis_team() -> StateGraph:
    """Build the analysis department subgraph."""

    def data_analyst(state: TeamState) -> dict:
        """Analyzes the available information."""

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a data analyst. Analyze the information in the "
                        "conversation and identify key insights, trends, and "
                        "important observations. Keep it to 3-5 bullet points."
                    )
                )
            ] + state["messages"]
        )

        return {
            "messages": [
                AIMessage(
                    content=response.content,
                    name="data_analyst"
                )
            ]
        }

    def strategy_advisor(state: TeamState) -> dict:
        """Creates strategic recommendations from the analysis."""

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a strategy advisor. Read the conversation, "
                        "especially the data analyst's findings, and provide "
                        "clear, practical recommendations. Keep it concise."
                    )
                )
            ] + state["messages"]
        )

        return {
            "messages": [
                AIMessage(
                    content=response.content,
                    name="strategy_advisor"
                )
            ],
            "final_ans": response.content
        }

    analysis_graph = StateGraph(TeamState)

    analysis_graph.add_node("analyst", data_analyst)
    analysis_graph.add_node("advisor", strategy_advisor)

    analysis_graph.add_edge(START, "analyst")
    analysis_graph.add_edge("analyst", "advisor")
    analysis_graph.add_edge("advisor", END)

    return analysis_graph

#================================
# Top-Level Supervisor(parent graph)
#================================
def create_hierarchical_system():
    """
    Top- Level supervisor that routes to dept subgraphs.
    Each dept is a compiled subgraph added as a single node.
    """

    research_team = build_research_team().compile()
    content_team = build_content_team().compile()
    analysis_team = build_analysis_team().compile()

    class DeptRoute(BaseModel):
        dept: Literal['research','content','analysis'] = Field(description="which team to be passed on")
        reasoning: str = Field(description='Why this dept is chosen')

    router_llm = llm.with_structured_output(DeptRoute)
    def ceo_supervisor(state: TeamState)->dict:
        """Top Level supervisor routes to the right dept."""
        decision = router_llm.invoke([
            SystemMessage(content=(
                "You are a CEO supervising three departments.\n"
                    "- research: fact finding, technical explanations, information gathering\n"
                    "- content: writing blogs, articles, emails, rewriting text\n"
                    "- analysis: insights, recommendations, business strategy, data interpretation\n\n"
                    "Return only the best department."
            ))
        ]+ state["messages"])

        return {
            'messages':[
                AIMessage(
                    content=f"routing to {decision.dept} - {decision.reasoning}",
                    name = 'ceo'
                )
            ],
            'department':decision.dept,
            'final_ans':""
        }

    def route_to_dept(state: TeamState)->str:
        """Read the CEO's routing decision from last message"""
        return state["department"]
                         
    graph = StateGraph(TeamState)
    graph.add_node('ceo',ceo_supervisor)
    graph.add_node('research',research_team)
    graph.add_node('content',content_team)
    graph.add_node('analysis',analysis_team)

    graph.add_edge(START,'ceo')
    graph.add_conditional_edges(
        'ceo',
        route_to_dept,
        {
            'research':'research',
            'content':'content',
            'analysis':'analysis'
        }
    )
    graph.add_edge('research',END)
    graph.add_edge('content',END)
    graph.add_edge('analysis',END)

    return graph.compile()

def working_hierarchical_routing():
    """Full hierarchical system with routing."""
    app  = create_hierarchical_system()
    print("HIERARCHICAL ROUTING DEMO")
    queries = [
        'What are the latest rends in llm',
        'Write a short blog introduction about ai agents',
        'should my startup invest in building ai features this year'
    ]
    for query in queries:
        print(f"Query: {query}")
        print("-"*40)
        result = app.invoke(
            {
                'messages':[HumanMessage(content=query)],
                "final_ans":"",
                'department':""
            }
        )
        for msg in result['messages']:
            if isinstance(msg,AIMessage) and msg.name == 'ceo':
                print(f"{msg.content}")
        print(f"Final ans is : {result['final_ans']}")
        print("-"*40)

def trace():
    """Show full trace through the hierarchy."""
    app = create_hierarchical_system()
    print("FULL HIERARCHICAL TRACE\n")
    inputs = {
            'messages':[
                HumanMessage(content='Research the impact of ai agents on software dev production')
            ],
            "final_ans":"",
            "department":""
        }

    for r in app.stream(inputs,stream_mode='updates'):
        print("="*63)
        print(r)

if __name__ == '__main__':
    working_hierarchical_routing()
    print('\n\n')
    trace()