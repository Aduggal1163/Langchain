"""
Human-In-The-Loop patterns in LangGraph
Interrupt, review, modify and resume
"""

from langgraph.graph import StateGraph,START,END
from langchain.chat_models import init_chat_model
from typing import Literal
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model('gpt-4o-mini',temperature=0.0)

class ApprovedState(TypedDict):
    request:str
    draft:str
    decision:bool
    feedback:str
    final:str

def interrupt_for_approval():
    """Interrupt execution for human approval."""
    
    def create_draft(state: ApprovedState)->dict:
        response = llm.invoke(
            f"Create a professional and precise response for: \n {state['request']}"
        )
        return {
            'draft':response.content
        }
    
    def wait_for_approval(state: ApprovedState)->dict:
        # This node is where we'll interrupt
        return state
    
    def finalize(state: ApprovedState)->dict:
        if state['decision']:
            return {
                'final':state['draft']
            }
        else:
            #Inappropriate feedback
            response = llm.invoke(
                f"Revise this draft based on the feedback\n"
                f"Draft: {state['draft']}\n\n\n"
                f"Feedback: {state['feedback']}"
            )
            return {
                'final':response.content
            }
    
    graph = StateGraph(ApprovedState)

    graph.add_node('draft',create_draft)
    graph.add_node('decision',wait_for_approval)
    graph.add_node('final',finalize)

    graph.add_edge(START,'draft')
    graph.add_edge('draft','decision')
    graph.add_edge('decision','final')
    graph.add_edge('final',END)
    
    memory = MemorySaver() # To save intermediate status for review
    app = graph.compile(
        checkpointer=memory,
        interrupt_before=['decision'] # pause before this node
    )

    print("="*50)
    print('HUMAN IN A LOOP: APPROVAL WORKFLOW')
    print("="*50)

    config = {
        'configurable':{
            'thread_id':'demo-1'
        }
    }

    #Run untill interrupt
    result = app.invoke(
        {
            'request': "Wrtie a thank you email for a job interview",
            'draft':"",
            'decision':False,
            'feedback':"",
            'final':""
        },
        config
    )
    print(f"Draft created: {result['draft'][:270]}")
    print("\n\nExecution paused for human review")

    #Update state with human input
    app.update_state(
        config,
        {
            'decision':False,
            'feedback':"Make it more concise and add specific mention of the company GOOGLE"
        }
    )

    #Continue Execution
    final_result = app.invoke(None,config)

    #Result
    print(f"\n\n\n\nFinal Result email is: {final_result['final']}")

#Human inn loop + Cycle combiine
class ReviewState(TypedDict):
    document:str
    review_comments:list[str]
    revision_count:int
    status:str

def iterative_review():
    """Multiple rounds of human review."""

    def submit_for_review(state:ReviewState)->dict:
        if(state['status'] == 'approved'): return {}
        return {
            'status':'pending_review'
        }
    
    def apply_feedback(state:ReviewState)->dict:
        if not state['review_comments']:
            print(f"No cmnt to apply. Passing through")
            return state
        
        feedback = state['review_comments'][-1]
        response = llm.invoke(
            f"Revise this document based on feedback\n"
            f"Document: {state['document']}\n\n\n"
            f"Feedback: {feedback}"
        )
        
        return {
            'document':response.content,
            'revision_count':state['revision_count']+1,
            'status':'revised'
        }
    
    def route_after_review(state:ReviewState)->Literal['apply','done']:
        if state['status'] =='approved':
            return 'done'
        else: return 'apply'
    
    def finalize(state: ReviewState)->dict:
        return {
            'status':'finalized'
        }
    
    graph = StateGraph(ReviewState)
    graph.add_node('submit',submit_for_review)
    graph.add_node('apply',apply_feedback)
    graph.add_node('done',finalize)

    graph.add_edge(START,'submit')
    graph.add_conditional_edges(
        'submit',
        route_after_review,
        {
            'apply':'apply',
            'done':'done'
        }
    )
    graph.add_edge('apply','submit')
    graph.add_edge('done',END)

    memory = MemorySaver()
    app = graph.compile(
        checkpointer=memory,
        interrupt_after=['submit']
    )

    config = {
        'configurable':{
            'thread_id':'demo123'
        }
    }

    result = app.invoke(
        {
            'document':"AI is vast technology ",
            'review_comments':[],
            'revision_count':0,
            'status':""
        },
        config
    )

    print(f"Initial Document: {result['document']}")

    app.update_state(
        config,
        {
            'review_comments':["Add more technical depths"],
            'status':'need_revisions'
        }
    )

    result = app.invoke(None,config)

    print(f"After revision 1 : {result['document']}")

    app.update_state(config,{
        'status':'approved'
    })

    result = app.invoke(None,config)
    print(f"After revision 1 : {result['document']}")
    print(f"Final revision 1 : {result['revision_count']}")


if __name__ == '__main__':
    # interrupt_for_approval()
    iterative_review()