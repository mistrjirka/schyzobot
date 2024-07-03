from graph_state import GraphState

def getFormattedResult(result: GraphState):
    code_execution_result = result["code_output"]
    code = result["code"]
    explanation = result["explanation"]
    example = result["examples"]

    result["answer"] = f"Code: {code}\nExample:{example} Result: {code_execution_result}\nExplanation: {explanation}"

    return result