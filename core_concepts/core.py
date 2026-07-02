from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
def demo_basic_chain():
    """Demonstrates a basic chain using LCEL and Runnables"""
    #step1 define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template("You are a helpful AI Assistent. answer my question in one sentence {question}")
    model = ChatOpenAI(model="gpt-4o-mini",temperature=0)
    parser = StrOutputParser()

    #step2 componse with pipe operator
    chain = prompt | model | parser
    # it means prompt goes into the model and model does what it needs to done and give a output and we then pass to parser for our final result in string

    #step3 execute the chain with input

    response = chain.invoke({"question":"What is history of 2 Jul?"})
    print(f"response is : {response}")

    return chain

def demo_batch_exectution():
    """Demonstrates a Batch execution with multiple inputs"""
    prompt = ChatPromptTemplate.from_template("You are a helpful AI Assistent. Help me to translate this text to punjabi {text}")
    model = ChatOpenAI(model='gpt-4o-mini',temperature=0)
    parser = StrOutputParser()

    chain = prompt | model | parser

    #Batch - run with multiple inputs
    inputs = [
        {"text":"Hi, How are you"},{"text":"May god bless you"},{"text":"Stay Happy my friend"},{"text":"Which is the best place to visit here"},
    ]
    # response = chain.invoke(inputs)
    responses = chain.batch(inputs)
    # for response in responses:
    #     print(response)
    for i, response in enumerate(responses, start=1): #ENUMERATE FOR NUMBERING
        print(f"Response {i}: {response}")

def demo_streaming():
    """Demonstrates streaming for real-time updates"""
    prompt = ChatPromptTemplate.from_template("You are a helpful AI Assistent. Write a haiku about {topic}")
    model = ChatOpenAI(model="gpt-4o-mini",temperature=0)
    parser = StrOutputParser()

    chain = prompt | model | parser

    #Streaming - run with streaming embedded

    print("Streaming output: ")
    for chunk in chain.stream({"topic":"WWE"}):
        print(chunk , end = "", flush= True)
        print()

def demo_schema_inspection():
    """Demonstrates input/output schema inspection """
    prompt = ChatPromptTemplate.from_template("You are a helpful AI Assistent. Summarize this text{topic}")
    model = ChatOpenAI(model="gpt-4o-mini",temperature=0)
    parser = StrOutputParser()

    chain = prompt | model | parser

    #inspect input and output schema
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input schema: {input_schema}")
    print(f"Output schema: {output_schema}")

def new_way():
    # NEW universal way to init a model
    model = init_chat_model(modeL="gpt-4o-mini",temperature=0.7)

#------Exercise---------#
def demo_product_tagline():
    """
    create a chain that:
    take product name and target audience
    generate marketing tagline
    return just tagline as string
    """
    prompt = ChatPromptTemplate.from_template("You are a helpful ai assistent. Help me to generate a tagline for product : {product} and mainly target audience is {audience} return only the tagline")
    model = ChatOpenAI(model='gpt-4o-mini',temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    response = chain.invoke({"product":"Macbook AIR M5","audience":"Developers"})
    print(response)

if __name__ == "__main__":
    # demo_basic_chain()
    # demo_batch_exectution()
    # demo_streaming()
    # demo_schema_inspection()
    demo_product_tagline()