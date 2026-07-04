"""
Section 1 Project : Smart Q&A Bot
A production-ready question-answering bot with structured output
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel,Field
from typing import List
from langsmith import traceable,Client
import os
from dotenv import load_dotenv

load_dotenv()

#----Langsmith Configuration------
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT_NAME","SMART Q/A BOT") # setdefault() means "If this variable doesn't exist, create it. Otherwise leave it unchanged."
    print(f"LangSmith is configured -- Project : {os.getenv('LANGSMITH_PROJECT_NAME')}")
else:
    print("Issue in Langsmith configurations")

#----Schema Defination---------
class QAResponse(BaseModel):
    answer : str = Field(description="The answer to the user's question.")
    confidence : str = Field(description='Confidence Level : high,med,low.')
    reasoning : str = Field(description="The reason behind this answer.")
    follow_up_questions : List[str] = Field(description="Any follow up question to this given answer?",default_factory=list) # creates a new empty list every time a QAResponse object is created.
    sources_needed : bool = Field(description="Any source needed from where we get this result",default=False)

#__init__ is a special method that Python calls automatically when you create an object.
#self refers to the current object.
class SmartQABot:
    def __init__(
            self,
            model_name : str = 'gpt-4o-mini',
            temperature : float = 0.3
    ):
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        ).with_structured_output(QAResponse)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    """
                        You are a knowledgeable Q&A assistant.
                        Your guidelines:
                        - Answer questions accurately and concisely
                        - Be honest about uncertainty - set confidence to 'low' if unsure (choose confidence from low,med,high only)
                        - Provide clear reasoning for your answers
                        - Suggest relevant follow-up questions
                        - Indicate if external sources would help
                        Always respond with accurate, helpful information.
                    """
                ),
                (
                    'human','{question}'
                )
            ]
        )
        self.chain = self.prompt | self.model #self.parser not required because we have used structured output already

    @traceable(name='ask_question',run_type='chain')
    def ask(self,question : str) -> QAResponse:
        try:
            response = self.chain.invoke({'question':question})
            return response
        except Exception as e:
                #return a graceful error response
            return QAResponse(
                answer="I'm sorry, I could not process the question",
                confidence='low',
                reasoning=str(e),
                follow_up_questions=['Could you please try again later!'],
                sources_needed=False
            )
            
    @traceable(name='ask_batch',run_type='chain')
    def ask_batch(self,questions: List[str]) ->List[QAResponse]:
        """ASK multiple questions in parallel"""
        inputs = [{'question':q} for q in questions]
        return self.chain.batch(inputs)
    
        
def demo_qa_bot():
    bot = SmartQABot()
    questions = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "How does photosynthesis work?",
    ]
    print("=" * 60)
    print("SMART Q&A BOT DEMO")
    print("=" * 60)
    for question in questions:

        print(f"\n Question: {question}")
        print("-" * 40)

        response = bot.ask(question)

        print(f"Question: {question}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Follow-up Questions: {response.follow_up_questions}")
        print(f"Sources Needed: {response.sources_needed}")
        print("-" * 60)

@traceable(name="error_handling_demo", run_type="chain")
def demo_error_handling():
    """Demonstrate error handling"""
    bot = SmartQABot()
    print("\n" + "=" * 60)
    print("ERROR HANDLING DEMO")
    print("=" * 60)

    # Test with a very long question (edge case)
    long_question = "what is " + "very" *100 +"important?"
    response = bot.ask(long_question)
    print(f"Handled gracefully: {response.confidence}")

@traceable(name="batch_demo", run_type="chain")
def demo_batch_processing():
    """Demonstrate batch processing."""

    bot = SmartQABot()

    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    print("\n" + "=" * 60)
    print("BATCH PROCESSING DEMO")
    print("=" * 60)

    responses = bot.ask_batch(questions)

    for q, r in zip(questions, responses):
        print(f"\n{q}")
        print(f"  -> {r.answer[:100]}...")
        print(f"  Confidence: {r.confidence}")

if __name__ == "__main__":
    try:
        demo_qa_bot()
        demo_error_handling()
        demo_batch_processing()

    finally:
        pass
        # Client().flush() # ensure traces are sent to langSmith