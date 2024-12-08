from smart.helpers.graph_state import GraphState

def getFormattedResult(result: GraphState):
    if "update_process" in result:
        result["update_process"]("Formatting the final result")
    code_execution_result = result["code_output"]
    code = result["code"]
    explanation = result["explanation"]
    example = result["examples"]
    # extract metadata links
    resources = result["additionalResources"]
    links = ""
    for resource in resources:
        links += f"{resource.metadata["source"]}\n"
    
    result["answer"] = f"""

## Code:

```python

{code}

```

## Example: 

```python

{example}

```

## Result: 
```bash
{code_execution_result}
```
## Explanation:

{explanation}

## Additional Resources:

{links}
    """

    return result
