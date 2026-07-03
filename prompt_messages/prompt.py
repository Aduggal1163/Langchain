from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage,HumanMessage
load_dotenv()
#Chat prompt message
prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}")
#format and inspect
message = prompt.format_messages(adjective='funny', topic='cricket')
print()
print(message)
print()
#multi-message template
prompt2 = ChatPromptTemplate.from_messages(
    [
        ('system',"You are a helpful guy help me to translate from {input_language} to {output_language}"),
        ('human',"Translate the following text {text}")
    ]
)
message2 = prompt2.format_messages(input_language="English",output_language="French",text="hi")
print(message2)
print()
model = init_chat_model(
    model='gpt-4o-mini',
    model_provider='openai',
    temperature=1
)
response = model.invoke(message2)
print(response.content)
