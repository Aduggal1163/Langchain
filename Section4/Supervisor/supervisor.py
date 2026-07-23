"""
Supervisor Architecture in LangGraph
One agent coordinates multiple specialist agents
"""
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage,SystemMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict,Annotated
from typing import Literal
from pydantic import BaseModel,Field
import operator
import json
from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model(
    model='gpt-4o-mini',
    temperature=0.0
)

class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
    next_agent: str
    task_complete: bool
    final_result: str

def create_supervisor_system():
    """Create a supervisor with specialist agents."""

    #Define the routing schema
    class RouteDecision(BaseModel):
        next: Literal['researcher','writer','critic','FINISH'] = Field(
            description='The next agent to call or FINSIH if task is completed'
        )
        reasoning: str = Field(description="Why we decided to choose this")

    supervisor_llm = llm.with_structured_output(RouteDecision)

    #Supervisor node
    def supervisor(state: SupervisorState)->dict:
        system_prompt = """
    You are a supervisor managing a team of AI specialists.

Available agents:

1. researcher
   - Finds facts, gathers information, performs analysis.
   - Use whenever information needs to be collected or verified.

2. writer
   - Writes, rewrites, summarizes, or formats content.
   - Produces polished final text.

3. critic
   - Reviews the writer's output.
   - Checks for factual errors, grammar, clarity, completeness, and suggests improvements.

Your responsibilities:
- Examine the conversation and previous agent outputs.
- Decide which single agent should work next.
- Call only one agent at a time.
- If the task is complete and no more work is needed, choose FINISH.

Routing guidelines:
- Need information → researcher
- Need content creation → writer
- Need review or improvement → critic
- Everything complete → FINISH

Always explain your reasoning briefly.
    """
        messages = [SystemMessage(content=system_prompt)] + state['messages']
        decision = supervisor_llm.invoke(messages)

        if decision.next == 'FINISH':
            return {
                'next_agent':"FINISH",
                'task_complete':True,
                "final_result": state["messages"][-1].content,
            }
        return {
            'next_agent': decision.next,
            'task_complete':False,
            'messages': [AIMessage(content=f"[Supervisor] routing to {decision.next}: {decision.reasoning}")]
        }

    #Define specialist agents 
    
    # -------------------------
    # Researcher Agent
    # -------------------------
    def researcher(state: SupervisorState) -> dict:
        system_prompt = """
You are a research specialist.

Your job is to:
- Gather relevant information.
- Explain concepts accurately.
- Do NOT write polished articles.
- Return only research notes and facts.
"""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)

        return {
            "messages": [
                AIMessage(
                    content=f"[Research]\n{response.content}"
                )
            ]
        }


    # -------------------------
    # Writer Agent
    # -------------------------
    def writer(state: SupervisorState) -> dict:
        system_prompt = """
You are a professional writer.

Use the available research to produce a polished answer.

Do not critique your own work.
"""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)

        return {
            "messages": [
                AIMessage(
                    content=f"[Draft]\n{response.content}"
                )
            ]
        }


    # -------------------------
    # Critic Agent
    # -------------------------
    def critic(state: SupervisorState) -> dict:
        system_prompt = """
You are an expert reviewer.

Review the writer's response.

Check:
- factual correctness
- grammar
- clarity
- completeness
- organization

Suggest improvements if needed.
If everything looks good, clearly state that the draft is approved.
"""

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)

        return {
            "messages": [
                AIMessage(
                    content=f"[Review]\n{response.content}"
                )
            ]
        }

    def route_to_agent(state: SupervisorState)->str:
        if state['task_complete']:
            return 'FINISH'
        return state['next_agent']

    graph = StateGraph(SupervisorState)
    graph.add_node('supervisor',supervisor)
    graph.add_node('researcher',researcher)
    graph.add_node('critic',critic)
    graph.add_node('writer',writer)

    graph.add_edge(START,'supervisor')
    graph.add_edge('researcher','supervisor')
    graph.add_edge('critic','supervisor')
    graph.add_edge('writer','supervisor')

    graph.add_conditional_edges(
        'supervisor',
        route_to_agent,
        {
            'researcher':'researcher',
            'critic':'critic',
            'writer':'writer',
            'FINISH':END
        }
    )

    return graph.compile()

def demo_supervisor():
    "Run the supervisor system"
    app = create_supervisor_system()
    print("---------")
    result = app.invoke({
         'messages': [HumanMessage(content='write a short article on Langgraph')],
         'next_agent':"",
         "task_complete": False,
         'final_result':""
    })
    print("Agent Conversation")
    for msg in result['messages']:
        print(f"{msg.type}: {msg.content[:100]}")
    print(f"\n\n Final Response: \n\n {result['final_result']}")

if __name__ == '__main__':
    demo_supervisor()
