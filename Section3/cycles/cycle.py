from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage,SystemMessage
from langgraph.graph import StateGraph, START, END,add_messages
from typing_extensions import Annotated,TypedDict
from typing import Literal
import operator
from dotenv import load_dotenv
load_dotenv()

"""
Cycles and loops in langgraph
self-correcting agents and iterative refinement
"""
llm = init_chat_model("gpt-4o-mini", temperature=0.0)
class CodeGenState(TypedDict):
    task:str
    code:str
    errors:Annotated[list[str],operator.add]
    iterations:int
    max_iterations:int
    success:bool

def self_correcting_code():
    """Self correcting code generator"""

    def generate_code(state:CodeGenState)->dict:
        if state["iterations"] == 0:
            #First attempt
            prompt = f"Write the python code for : {state['task']}. Return only the code"
        else:
            #Correction attemp
            prompt = (
                f"Fix the python code:\n {state['code']}\n\n"
                f"Errors: \n {state['errors'][-1]}\n\n"
                "return only rhe corrected code"
            )
        response = llm.invoke(prompt)
        code = response.content.strip()

        # Clean up markdown code blocks if present
        if code.startswith("```"):
            code = code.split("```")[1]
            if code.startswith("python"):
                code = code[6:]
        
        return {
            'code':code,
            'iterations':state["iterations"]+1
        }
    
    def validate_code(state: CodeGenState) -> dict:
        code = state["code"]

        # Step 1: Does it compile?
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            return {"errors": [f"SyntaxError: {e}"], "success": False}

        # Step 2: Does it RUN and produce correct results?
        test_cases = [
            ([3, 1, 4, 1, 5, 9], 5),  # normal case
            ([1, 1, 1], None),  # all same → no second largest
            ([7], None),  # single element
            ([3, -1, 3, 5, 5], 3),  # duplicates at top
        ]

        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return {"errors": [f"Runtime error: {e}"], "success": False}

        if "solve" not in namespace:
            return {"errors": ["Function 'solve' not found in code"], "success": False}

        for inputs, expected in test_cases:
            try:
                result = namespace["solve"](inputs)
                if result != expected:
                    return {
                        "errors": [
                            f"solve({inputs}) returned {result}, expected {expected}"
                        ],
                        "success": False,
                    }
            except Exception as e:
                return {"errors": [f"solve({inputs}) raised {e}"], "success": False}

        return {"success": True}

    def should_continue(state: CodeGenState) -> Literal['generate','end']:
        if state["success"]:
            return 'end'
        elif state["iterations"] >= state["max_iterations"]:
            return 'end'
        else: return "generate" 
        
    graph = StateGraph(CodeGenState)
    graph.add_node('generate',generate_code)
    graph.add_node('validate',validate_code)
    graph.add_edge(START,'generate')
    graph.add_edge('generate','validate')
    graph.add_conditional_edges(
        'validate',
        should_continue,
        {
            'generate':'generate',
            'end':END
        }
    )
    app=graph.compile()
    print("Self-Correcting Code Generator:\n")

    result = app.invoke(
        {
            # "task": "a function that calculates factorial recursively",
            "task": """Write a function solve(arr) that returns the second largest unique
number in the list.
Return None if it doesn't exist.""",
            "code": "",
            "errors": [],
            "iterations": 0,
            "max_iterations": 3,
            "success": False,
        }
    )
    print(f"Task: {result['task']}")
    print(f"Iterations{result['iterations']}")
    print(f"Success: {result['success']}")
    print("Final Code:\n",result['code'])


if __name__ == '__main__':
    self_correcting_code()