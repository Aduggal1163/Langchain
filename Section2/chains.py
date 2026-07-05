"""
Understanding chains in Langchain V.1
LCEL Patterns, compositions and debugging
"""

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

def simple_chain():
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")

    model = init_chat_model(
        model='gpt-4o-mini',
        model_provider='openai',
        temperature=0.7
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    response = chain.invoke({'text':"My name is Er. Abhishek Duggal"})

    print(f"Giving summary is this: {response}")


def parallel_chain():
    """
    Runs multiple chains in parallel
    """
    #define multiple chains
    summarize_prompt = ChatPromptTemplate.from_template("Give me the summary in two lines of this after generating print two empty lines{text} ")
    keyword_prompt = ChatPromptTemplate.from_template("Give me 5 keywords for this after generating print two empty lines {text} ")
    sentimental_prompt = ChatPromptTemplate.from_template("Tell me the sentiment of this {text}")

    model = init_chat_model('gpt-4o-mini',temperature = 0)

    parser = StrOutputParser()

    analyzed_chain = RunnableParallel(
        summary = summarize_prompt | model | parser,
        keywords = keyword_prompt | model | parser,
        sentiment = sentimental_prompt | model | parser
    )

    text = """
        Sequential Pattern

A Sequential Chain executes tasks one after another. The output produced by one step becomes the input for the next step. Since each stage depends on the previous one, the order of execution is important. This pattern is useful for workflows where each operation builds upon the results of the earlier operation, such as document processing, data transformation, or multi-step reasoning.

Flow: Input → Step 1 → Step 2 → Step 3 → Output

Parallel Pattern

A Parallel Chain executes multiple independent tasks simultaneously using the same input. Each chain performs a different operation without waiting for the others. Once all chains finish, their outputs are combined into a single final result. This pattern improves efficiency and is ideal when tasks do not depend on one another.

Example: An input document is sent to two chains at the same time—one generates a summary while the other extracts keywords. The final output contains both the summary and the keywords.

Flow: Input → (Summarize Chain + Keyword Chain) → Output {summary, keywords}
        """
    
    result = analyzed_chain.invoke({'text': text})
    print("Analyzed Results ........")
    print(f"Summary is : {result['summary']}")
    print(f"Keywords are : {result['keywords']}")
    print(f"Sentiment is : {result['sentiment']}")

if __name__ == '__main__':
    # simple_chain()
    parallel_chain()
