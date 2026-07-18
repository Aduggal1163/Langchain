"""
Langgraph core concepts 
StateGraph, nodes, edges and basic patterns
"""

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage
from dotenv import load_dotenv
import operator
from langgraph.graph import StateGraph, START, END,add_messages
from typing_extensions import Annotated,TypedDict

load_dotenv()

#Basic State
class SimpleState(TypedDict):
    input : str
    output : str
    step : int

def simple_graph():
    #define node functions
    def process(state: SimpleState)->dict:
        return {
            'output':state['input'].upper(),
            'step':state['step']+1
        }
    
    #create graph
    graph = StateGraph(SimpleState)

    #add nodes
    graph.add_node('process',process)

    #add edges
    graph.add_edge(START,'process')
    graph.add_edge('process',END)

    #execute graph/ compile
    app = graph.compile()

    #run app
    result = app.invoke({
        'input':'hello',
        'output':{},
        'step':0
    })

    print(f"Simple Graph Result IS:")
    print(f"Input: {result['input']}\n Output: {result['output']}\n Step: {result['step']}")

#State With Reducers
class AnnotatingState(TypedDict):
    messages : Annotated[list[str],operator.add]
    count : Annotated[int,operator.add]

def accumulating_state():
    #create function nodes
    def step_one(state : AnnotatingState)->dict:
        return {
            'messages': ['step 1 executed'],
            'count':1
        }
    def step_two(state : AnnotatingState)->dict:
        return {
            'messages': ['step 2 executed'],
            'count':1
        }
    
    #create graph
    graph = StateGraph(AnnotatingState)

    #creating nodes
    graph.add_node('step1',step_one)
    graph.add_node('step2',step_two)

    #creating edges
    graph.add_edge(START,'step1')
    graph.add_edge('step1','step2')
    graph.add_edge('step2',END)

    #compile graph
    app = graph.compile()

    #run app
    result = app.invoke({
        'messages':['initial message'],
        'count':0
    })

    print('ACCUMULATING STATE RESULTS')
    print(f"Messages: {result['messages']}")
    print(f"count: {result['count']}")

#===Message State===
class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def message_state():
    llm = init_chat_model('gpt-4o-mini',temperature=0)
    def chat_node(state: MessageState)->dict:
        response = llm.invoke(state['messages'])
        return {
            'messages':[response]
        }    
    graph = StateGraph(MessageState)
    graph.add_node('chat_node',chat_node)
    graph.add_edge(START,'chat_node')
    graph.add_edge('chat_node',END)
    app=graph.compile()
    result = app.invoke({
        'messages':[HumanMessage(content='Say I Love You in 10 different languages')]
    })
    print("\nChat Message State Responses")
    for msg in result['messages']:
        role = 'Human' if isinstance(msg,HumanMessage) else 'AI'
        print(role,": ",msg.content)




if __name__ == '__main__':
    # simple_graph()
    # accumulating_state()
    message_state()