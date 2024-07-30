from langgraph.graph import END, StateGraph
from .graph_state import GraphState
from .programmer_controllerEDGE import judgeProgram
from .programmer_modelNODE import makeProgram
from .question_classifierNODE import classify_question
from .retarded_radekNODE import answer
from .resultNODE import getFormattedResult
failedThreshold = 5
workflow = StateGraph(GraphState)
workflow.add_node("classify_question", classify_question)
workflow.add_node("makeProgram", makeProgram)
workflow.add_node("retarded_radek", answer)
workflow.add_node("result", getFormattedResult)

workflow.set_entry_point("classify_question")
workflow.add_conditional_edges(
    "classify_question",
    lambda state: state["type"],
    {
        "other": "retarded_radek",
        "needs_code": "makeProgram"
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
