from langgraph.graph import END, StateGraph
from .graph_state import GraphState
from .programmer_controllerEDGE import judgeProgram
from .programmer_modelNODE import makeProgram
from .question_classifierNODE import classify_question
from .nocodeNODE import answer
from .memoryNODE import process_graph_state
from .resultNODE import getFormattedResult
from .getLinksPromptNODE import links_prompt
from .extractLinksDataNODE import load_links
from .preResearchEdge import classify_prompt_researchNeeded
failedThreshold = 5
workflow = StateGraph(GraphState)
workflow.add_node("classify_question", classify_question)
workflow.add_node("makeProgram", makeProgram)
workflow.add_node("retarded_radek", answer)
workflow.add_node("result", getFormattedResult)
workflow.add_node("context", process_graph_state)
workflow.add_node("linksPrompt", links_prompt)
workflow.add_node("load_links", load_links)


workflow.set_entry_point("linksPrompt")
workflow.add_conditional_edges(
    "linksPrompt",
    classify_prompt_researchNeeded,
    {
        "RESEARCH": "context",
        "SUMMARIZE": "load_links"
    }
)

workflow.add_edge("load_links", "classify_question")
    
workflow.add_edge("context", "classify_question")

workflow.add_conditional_edges(
    "classify_question",
    lambda state: state["type"],
    {
        "other": "retarded_radek",
        "code_related": "retarded_radek",
        "python": "retarded_radek"
    },
)


def try_again(state: GraphState) -> str:
    if judgeProgram(state):
        return "done"
    elif state.get("failedTimes", 0) > failedThreshold:
        return "failed"
    else:
        return "again"

workflow.add_conditional_edges(
    "makeProgram",
    try_again,
    {
        "again": "makeProgram",
        "done": "result",
        "failed": "retarded_radek"
    }
)

workflow.add_edge("retarded_radek", END)
workflow.add_edge("result", END)
chatBot = workflow.compile()
