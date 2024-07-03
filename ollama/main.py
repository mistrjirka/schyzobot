from graph import chatBot
from graph_state import GraphState
QUESTION = "I want to solve fitting a linear regression model to a dataset. I have it here [(1.0, 2.5), (2, 3), (3, 4), (4, 5), (5, 6)]. Can you find parameters for me?"
QUESTION2 = "I want to fuck your mom bitch."
QUESTION3 = """
## Virtual DAC

Published by AniXDownLoe in Python

### functional_programming

In electronics, a digital-to-analog converter (DAC, D/A, or D-to-A) is a system that converts a binary representation of that signal into an analog output. An 8-bit converter can represent a maximum of 2^8 different values, with each successive value differing by 1/256 of the full scale value, this becomes the system resolution.

Create a function that takes a decimal number representation of a signal and returns the analog voltage level that would be created by a DAC if it were given the same number in binary.

While value range is 0-1023, reference range is 0-5.00 volts. Value and reference is directly proportional.

This DAC has 10 bits of resolution and the DAC reference is set at 5.00 volts.

### Examples

```
V_DAC(0) ➞ 0

V_DAC(1023) ➞ 5

V_DAC(400) ➞ 1.96
```

### Notes

You should return your value rounded to two decimal places.

"""

QUESTION4 = """

## Prison Break

Published by Helen Yu in Python

### arrays games logic loops

A prison can be represented as a list of cells. Each cell contains exactly one prisoner. A `1` represents an unlocked cell and a `0` represents a locked cell.

```
[1, 1, 0, 0, 0, 1, 0]
```

Starting inside the leftmost cell, you are tasked with seeing how many prisoners you can set free, with a catch. You are the prisoner in the first cell. If the first cell is locked, you cannot free anyone. Each time you free a prisoner, the locked cells become unlocked, and the unlocked cells become locked again.

So, if we use the example above:

```
[1, 1, 0, 0, 0, 1, 0]
# You free the prisoner in the 1st cell.

[0, 0, 1, 1, 1, 0, 1]
# You free the prisoner in the 3rd cell (2nd one locked).

[1, 1, 0, 0, 0, 1, 0]
# You free the prisoner in the 6th cell (3rd, 4th and 5th locked).

[0, 0, 1, 1, 1, 0, 1]
# You free the prisoner in the 7th cell - and you are done!
```

Here, we have set free `4` prisoners in total.

Create a function that, given this unique prison arrangement, returns the number of freed prisoners.

### Examples

```
freed_prisoners([1, 1, 0, 0, 0, 1, 0]) ➞ 4

freed_prisoners([1, 1, 1]) ➞ 1

freed_prisoners([0, 0, 0]) ➞ 0

freed_prisoners([0, 1, 1, 1]) ➞ 0
```

### Notes

- **You are the prisoner in the first cell. You must be freed to free anyone else.**
- You must free a prisoner in order for the locks to switch. So in the second example where the input is `[1, 1, 1]` after you release the first prisoner, the locks change to `[0, 0, 0]`. Since all cells are locked, you can release no more prisoners.
- You always start within the leftmost element in the list (the first prison cell). If all the prison cells to your right are zeroes, you cannot free any more prisoners.

---
"""
state = GraphState()
state["prompt"] = QUESTION4
state["previous_result"] = "None"
state["previous_code"] = "None"
state["failedTimes"] = 0

result = chatBot.invoke(state)

print(result)
print(result["answer"])