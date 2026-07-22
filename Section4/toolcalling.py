"""
Tool calling agent with langGraph
Building agents that can call tools
"""
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict,Annotated
from typing import Literal
import operator
import json
from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model(
    model='gpt-4o-mini',
    temperature=0.0
)

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression"""
    try:
        result = eval(expression)
        return f"The result for {expression} is: {result}"
    except Exception as e:
        return f"Error while calculating: {e}"

@tool
def search(query: str) -> str:
    """Search the web for general information."""
    # Dummy implementation
    return f"Search results for '{query}': Python is a popular programming language developed by Guido van Rossum."

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def create_tool_agent():
    """Create a basic tool calling agent"""
    tools = [calculate,search]
    
    llm_with_tools = llm.bind_tools(tools)

    def agent_tools(state: AgentState)->dict:
        response = llm_with_tools.invoke(state['messages'])
        return {
            'messages':[response]
        }
    
    def should_continue(state:AgentState)->Literal['tools','end']:
        """Check if we should continue to tools or end"""
        last_message = state['messages'][-1]
        #if no tools then we are done
        if isinstance(last_message,AIMessage) and last_message.tool_calls: #last_message.tool_calls is a property of an AIMessage. It contains the list of tool calls that the LLM wants to execute.
            return 'tools'
        else:
            return 'end'    
    #create tool node
    tool_node = ToolNode(tools)
    #create graph
    graph = StateGraph(AgentState)
    graph.add_node('agent',agent_tools)
    graph.add_node('tools',tool_node)
    graph.add_edge(START,'agent')
    graph.add_conditional_edges(
        'agent',
        should_continue,
        {
            'tools':'tools',
            'end':END
        }
    )
    graph.add_edge('tools','agent')
    
    app = graph.compile()
    messages = [
        "what is 25+2",
        "What is python"
    ]
    for msg in messages:
        result = app.invoke({
            'messages':[
                HumanMessage(content=msg)
            ]
        })
        print(result['messages'][-1].content)
        print(len(result['messages']))

@tool
def find_city(country: str)->str:
    """Returns the capity of this country""" 
    result = llm.invoke(
        [HumanMessage(content=f"What is the capital of {country} in one word")]
    )
    return result.content
    
@tool
def find_weather(city: str)->str:
    """Returns the weather in F of this city"""
    # result = llm.invoke(
    #         [HumanMessage(content=f"today's weather of this city {city} in F is 63")]
    #     )
    # return result.content
    return "63.27"

@tool
def f_to_c(f: float)->float:
    """Convert temp f to c"""
    c = (f-32)*5/9
    return c

class WeatherState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

def create_weather_tool_agent():
    tools = [find_weather,find_city,f_to_c]

    llm_with_tools = llm.bind_tools(tools)

    def weather_agent(state: WeatherState)->dict:
        result = llm_with_tools.invoke(
            state['messages']
        )
        return {
            'messages':[result]
        }

    def should_continue(state: WeatherState)->Literal['tools','end']:
        last_msg = state['messages'][-1]
        if isinstance(last_msg,AIMessage) and last_msg.tool_calls:
            return 'tools'
        else:
            return 'end'

    toolnode = ToolNode(tools)

    graph = StateGraph(WeatherState)
    graph.add_node('weatheragent',weather_agent)
    graph.add_node('tools',toolnode)
    graph.add_edge(START,'weatheragent')
    graph.add_conditional_edges(
        'weatheragent',
        should_continue,
        {
            'tools':'tools',
            'end':END
        }
    )
    graph.add_edge('tools','weatheragent')
    app = graph.compile()
    response = app.invoke(
        {
            'messages':[
                HumanMessage(content='Tell me todays temp (in c) and capital of Argentina')
            ]
        }
    )
    for res in response['messages']:
        print(res.content)
    print(len(response['messages']))


#----------------
#----------------

@tool
def divide(a:float, b:float) ->str:
    """Divide two numbers"""
    if b == 0:
        return "Error divisible by 0"
    result = a/b
    return f"Result after dividng {a} with {b} is {result}"

def tool_with_error():
    """Tool calling with errors"""
    tools = [divide]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state['messages'])
        return {
            'messages':[response]
        }
    def should_continue(state: AgentState) -> Literal['tools','end']:
        last_msg = state['messages'][-1]
        if isinstance(last_msg,AIMessage) and last_msg.tool_calls:
            return 'tools'
        else: return 'end'
    tool_node = ToolNode(tools)
    graph = StateGraph(AgentState)
    graph.add_node('agent',agent_node)
    graph.add_node('tools',tool_node)
    graph.add_edge(START,'agent')
    graph.add_conditional_edges(
        'agent',
        should_continue,
        {
            'tools':'tools',
            'end':END
        }
    )
    graph.add_edge('tools','agent')
    app = graph.compile()
    print("Tool with error handling")
    messages = [
        'Divide 100 by 5',
        "Divide 100 by 0"
    ]
    for msg in messages:
        result = app.invoke({
            'messages':[
                HumanMessage(content=msg)
            ]
        })
        print(f"Query: {msg}")
        print(f"Response: {result['messages'][-1].content}")
        print("-"*40)

if __name__ == '__main__':
    # create_tool_agent()
    # create_weather_tool_agent()
    tool_with_error()