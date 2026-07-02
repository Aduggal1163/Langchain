from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

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


if __name__ == "__main__":
    demo_basic_chain()
    demo_batch_exectution()