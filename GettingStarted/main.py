from dotenv import load_dotenv
load_dotenv()

from importlib.metadata import version
core_version = version("langchain-core")
lg_version = version("langgraph")

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
print(f"langchain core version is: {core_version}")
print(f"langgraph version is: {lg_version}")

def main():
    
    print("Hello from langchain-course!")
    #testing openapi 
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
    response = llm.invoke("Say Good to go in one word")
    print(f"Response from open ai : {response}")
    print("---------")
    llm_anthropic = ChatAnthropic(model = "claude-sonnet-4-5-20250929",temperature=0)
    response_anthropic = llm_anthropic.invoke("say hi in hindi")
    print(f"Response from anthropic is : {response_anthropic}")
    print("Setup Complete!")

if __name__ == "__main__":
    main()
