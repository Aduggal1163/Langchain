"""
Working with LLMs in Langchain V.1
Multiple providers, configuration, streaming and cost optimization 
"""
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def demo_chat_init_model():
    chat_model = init_chat_model(
        model='gpt-4o-mini',
        model_provider='openai',
        temperature=0.7,
        max_retries=3,
        max_tokens=1000,
        streaming=True
    )
    # return chat_model
    response = chat_model.invoke("What is the best thing about Nabha,Punjab INDIA in one line")
    print(response.content)

    #easy to switch between model providers
    if os.getenv("ANTHROPIC_API_KEY"):
        anthropic_chat_model = init_chat_model(
            model='claude-sonnet-4-5-20250929',
        model_provider='anthropic',
        temperature=0.7,
        max_retries=3,
        max_tokens=1000,
        streaming=True
        )
        anthropic_response = anthropic_chat_model.invoke("Which is Capital of India in one word")
        print(f"Response from anthropic : {anthropic_response.content}")

def demo_model_compare():
    prompt = "Explain recurssion in two sentences"
    models = {
        "gpt-4o-mini" : init_chat_model(
            model='gpt-4o-mini',
            model_provider='openai',
            temperature=0.7,
            streaming=True
        ),
        "claude-sonnet-4-5-20250929" : init_chat_model(
            model = 'claude-sonnet-4-5-20250929',
            model_provider = 'anthropic',
            temperature = 0.7,
            streaming = True
        )
    }
    print(f"Prompt: {prompt}\n\n")
    for model_name,model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name} is {response.content}\n\n")

def demo_messages():
    model = ChatOpenAI(model='gpt-4o-mini',temperature=0)

    messages = [
        SystemMessage(content="You are a pirate so always answer me like a pirate."),
        HumanMessage(content="Whats your goal today")
    ]
    response = model.invoke(messages)

    print(f"response from pirate : {response.content} \n \n")
    
    #multi-turn conversation using message objects
    messages.append(response)
    messages.append(HumanMessage(content="what about tommorow's"))
    response = model.invoke(messages)

    print(f"continuing message from pirate: {response.content}")

if __name__ == '__main__':
    # demo_chat_init_model()
    # demo_model_compare()
    demo_messages()