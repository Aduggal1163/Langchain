"""
Understanding chains in Langchain V.1
LCEL Patterns, compositions and debugging
"""

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from dotenv import load_dotenv
load_dotenv()

def simple_runnable():
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

def parallel_runnable():
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

def passthrough_runnable():
    """
    A chain that demonstrate passthrough funtionality
    """
    model = init_chat_model('gpt-4o-mini',temperature = 0)

    prompt = ChatPromptTemplate.from_template(
        "Original Question: {question}\n"
        "Context: {context}\n"
        "Answer the following question based on this context"
    )
    
    #Simulate a reterival operation
    def fake_reterival(input_dict):
        return "Langchain was discoved by Harisson Chase in 2022"

    chain = (
        RunnableParallel(
            context = RunnableLambda(fake_reterival),
            question = RunnablePassthrough(),
        )
        | prompt | model  
    ) | StrOutputParser()

    result = chain.invoke({'question':"Who created LangChain?"})
    print(f"Answer: {result}")

def branch_runnable():
    model = init_chat_model('gpt-4o-mini',temperature = 0)
    """
    A chain that demonstrate branching functionality
    """
    # Different prompts for different intents
    code_prompt = ChatPromptTemplate.from_template(
        "You are coding expert. Help me with {text}"
    )
    general_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistent. Answer me this {text}"
    )
    
    # Classifier
    classifier_prompt = ChatPromptTemplate.from_template(
        "Classify this as code or general: {text} \n"
    )

    classifier = classifier_prompt | model | StrOutputParser()

    # Branching chain based on classifier
    def is_code_classifer(input_dict):
        classification = classifier.invoke(input_dict)
        return "code" in classification.lower()
    
    branch = RunnableBranch(
        (is_code_classifer, code_prompt |model | StrOutputParser()),
        general_prompt | model | StrOutputParser() # Default Branch
    )

    #Test
    questions = [
        "How do i write the for loop in python",
        "Whats your plans tonight"
    ]
    for ques in questions:
        result = branch.invoke({'text':ques})
        print(f"Q:{ques}")
        print(f"A: {result[:100]}...\n")

def demo_debugging():
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")
    model = init_chat_model('gpt-4o-mini',temperature = 0)
    chain = prompt | model | StrOutputParser()

    #Method 1 : Get Configuration
    print(f"Chain input Schema:",chain.input_schema.model_json_schema())
    print(f"Chain output Schema:",chain.output_schema.model_json_schema())

    #Method 2 : Use with_config for tracing
    result = chain.with_config(
        run_name = "greeting_chain"
    ).invoke({"name":"Alice"})
    print(f"Greeting {result}")

    #Method 3 : Inspect intermediate steps
    #Use runnable lambda for logging
    def log_input(x):
        print("\n===== INPUT =====")
        print(x)
        return x

    def log_prompt(x):
        print("\n===== PROMPT =====")
        print(x)
        return x

    def log_model_output(x):
        print("\n===== MODEL OUTPUT =====")
        print(x)
        return x

    debug_chain = (
        RunnableLambda(log_input)
        | prompt
        | RunnableLambda(log_prompt)
        | model
        | RunnableLambda(log_model_output)
        | StrOutputParser()
    )

    print("\nRunning debug chain...\n")
    final_result = debug_chain.invoke({"name": "Alice"})
    print("\n===== FINAL RESULT =====")
    print(final_result)

def demo_email():
    model = init_chat_model(
        model='gpt-4o-mini',
        model_provider='openai',
        temperature=0.7
    )

    generation_prompt = ChatPromptTemplate.from_template("Generate a professional email using the following {text}")
    summary_prompt = ChatPromptTemplate.from_template("summarize the entire generated email in one sentence {text}")
    
    text = """
    "name": "John",
    "topic": "Annual Leave",
    "reason": "Family vacation",
    "days": 5
    """

    chain = generation_prompt | model | StrOutputParser()
    chain_summary = summary_prompt | model | StrOutputParser()

    response = chain.invoke({'text':text})

    summary_response = chain_summary.invoke({'text':response})

    print("="*50,"Generating email","="*50) 
    print(f"{response}")
    print("="*50,"Generating summary","="*50) 
    print(f"{summary_response}")

if __name__ == '__main__':
    # simple_runnable()
    # parallel_runnable()
    # passthrough_runnable()
    # branch_runnable()
    # demo_debugging()
    demo_email()