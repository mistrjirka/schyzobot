from graph_state import GraphState

def getFormattedResult(result: GraphState):
    code_execution_result = result["code_output"]
    code = result["code"]
    explanation = result["explanation"]
    example = result["examples"]

    result["answer"] = f"""\n
        ## Code:
	\n```python
	{code}
	```
	## Example: 
	```python
	{example}
	```\n
	## Result: 
	{code_execution_result}
        ## Explanation:
	{explanation}
    """

    return result
