from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from langchain_core.output_parsers import (
    StrOutputParser
)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

prompt = ChatPromptTemplate.from_template("Write me a short poem about {topic}")

model = init_chat_model(
    model='gpt-4o-mini',
    model_provider='openai',
    temperature=1
)

#Simple output parser
parser = StrOutputParser()
chain = prompt | model | parser
response = chain.invoke({'topic' : 'vacations'})
print(response)

#JSON output parser
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser()
prompt = ChatPromptTemplate.from_template("Return me a JSON object with name and age for {description}")
chain = prompt | model | parser
response = chain.invoke({'description':'My name is Abhishek Duggal im currently 22 years.'})
print(response)

#Pydantic output parser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
class Person(BaseModel):
    name : str = Field(description="Enter name")
    age : int = Field(description="Enter age")
    occupation : str = Field(description="Enter name")
parser = PydanticOutputParser(pydantic_object=Person)
# prompt = ChatPromptTemplate.from_template("Return me a JSON object with name, age, occupation for {description}").partial(
    # format_instructions = parser.get_format_instructions()
# )
#OR
prompt = ChatPromptTemplate.from_template("""
Return the person's information.
{format_instructions}
Description:
{description}
""").partial(format_instructions=parser.get_format_instructions())
chain = prompt | model | parser
result = chain.invoke({'description':"My name is Abhishek Duggal PG student im currently 22 years."})
print(result)

#Structured Output Parser
class MovieReview(BaseModel):
    name : str = Field(description="Enter Name")
    description : str = Field(description="Enter description")
    rating : str = Field(description="Enter rating")
#bind schema to model
structured_model = model.with_structured_output(MovieReview)
result = structured_model.invoke("Review:Carry on Jatta 4 punjabi comedy movie IDMB rating 2.7/10")
print(result)
