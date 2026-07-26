"""I'm planning a trip to Japan. Tell me the capital, currency, language, current weather, and top tourist attractions."""

from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict,Annotated
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model(
    model='gpt-4o-mini',
    temperature=0.3
)

class TravelState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
#------------------------------
@tool
def get_capital(country: str)->str:
    """Returns the capity of this country"""
    response = llm.invoke(f"What is the capital of this country {country}")
    return response.content

@tool
def get_currency(country:str)->str:
    """Returns the currency of this country"""
    response = llm.invoke(f"What is the currency of this country {country}")
    return response.content

@tool
def get_languages(country: str)->str:
    """Returns what languages this country speaks"""
    response = llm.invoke(f"What languages do the people of {country} speaks")
    return response.content

@tool
def get_weather(city: str)->str:
    """Returns the weather in Celcius of this city"""
    return "63.27"

@tool
def get_tourist_attractions(country:str)->str:
    """Returns top tourist attractions of this country"""
    response = llm.invoke(f'What are the top tourist attractions of this {country}')
    return response.content
#------------------------------
def create_tourist_planner_agent():
    tools = [get_capital,get_currency,get_languages,get_weather,get_tourist_attractions]
    llm_with_tools = llm.bind_tools(tools)

    def get_response(state: TravelState)->dict:
        """Get the response from the tools"""
        response = llm_with_tools.invoke(state['messages'])
        return {
            'messages':[response]
        }

    def should_continue(state:TravelState)->Literal['tools','end']:
        last_msg = state['messages'][-1]
        if isinstance(last_msg,AIMessage) and last_msg.tool_calls:
            return 'tools'
        return 'end'

    toolnode = ToolNode(tools)
    graph = StateGraph(TravelState)
    graph.add_node('response',get_response)
    graph.add_node('tools',toolnode)
    graph.add_edge(START,'response')
    graph.add_conditional_edges(
        'response',
        should_continue,
        {
            'tools':'tools',
            'end':END
        }
    )
    graph.add_edge('tools','response')
    app = graph.compile()
    result = app.invoke({
        'messages':[
            HumanMessage(content= "I'm planning a trip to Japan. Tell me the capital, currency, language, current weather, and top tourist attractions.")
        ]
    }
    )

    print(result['messages'][-1].content)
if __name__ == '__main__':
    create_tourist_planner_agent()