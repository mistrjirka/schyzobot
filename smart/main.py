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

"""

QUESTION5 = """
The graph shows the development of the average gross wage (AGW) in the Czech Republic from 2000 to 2008. The time data is in the format t = year + (quarter-1)/4, where year ∈ {2000,…,2008} and quarter ∈ {1,2,3,4}.
```
2000.00 11941
2000.25 13227
2000.50 12963
2000.75 14717
2001.00 13052
2001.25 14391
2001.50 14117
2001.75 15908
2002.00 14083
2002.25 15599
2002.50 15268
2002.75 17133
2003.00 14986
2003.25 16529
2003.50 16088
2003.75 18096
2004.00 16231
2004.25 17223
2004.50 17190
2004.75 19183
2005.00 17067
2005.25 18112
2005.50 18203
2005.75 19963
2006.00 18270
2006.25 19300
2006.50 19305
2006.75 21269
2007.00 19687
2007.25 20740
2007.50 20721
2007.75 22641
2008.00 21647
2008.25 22370
2008.50 22282
2008.75 24484
```

Implement the function `x = fit_wages(t, M)` where `t` and `M` are vectors of length `m` with times and wages, and `x` is a vector of length 2 with parameters (x1, x2).

The goal is to approximately predict the value of AGW at times for which the AGW value is not known. We will first find a function that best fits the given AGW data, and then use this function to estimate AGW at the desired time. From the graph, it is evident that the dependence of AGW on time is almost linear. Therefore, we will look for a linear function:
```
M(t) = x1 + x2 * t
```
where `M(t)` is the estimated AGW at time `t` and `x1, x2 ∈ R` are parameters. Our measured sample denoted as {(t1, M1),…, (tm, Mm)} contains `m` pairs (time, AGW). The optimal parameters will be found from this sample in the sense of least squares, i.e., so that the sum of the squares of the deviations of the actual and estimated wage is minimal at the measured points. This means we minimize the function:
```
sum_{i=1}^{m} (M(ti) - Mi)^2
```

Implement the function `M = quarter2_2009(x)` that estimates AGW in the second quarter of 2009 for parameters `x` estimated by the function `fit_wages`.

### Expected Output:
```
2000.00 -> 11941
2000.25 -> 13227
2000.50 -> 12963
2000.75 -> 14717
2008.50 -> 22282
2008.75 -> 24484
```

Implement everything in Python!
"""

QUESTION6 = """
   Graf ukazuje vývoj průměrné hrubé mzdy (PHM) v České republice v období od roku 2000 do roku 2008. časový údaj je ve formátu t = rok+(kvartál-1)/4, kde rok ∈{2000,…,2008} a kvartál ∈{1,2,3,4}.
        2000.00 11941
        2000.25 13227
        2000.50 12963
        2000.75 14717
        2001.00 13052
        2001.25 14391
        2001.50 14117
        2001.75 15908
        2002.00 14083
        2002.25 15599
        2002.50 15268
        2002.75 17133
        2003.00 14986
        2003.25 16529
        2003.50 16088
        2003.75 18096
        2004.00 16231
        2004.25 17223
        2004.50 17190
        2004.75 19183
        2005.00 17067
        2005.25 18112
        2005.50 18203
        2005.75 19963
        2006.00 18270
        2006.25 19300
        2006.50 19305
        2006.75 21269
        2007.00 19687
        2007.25 20740
        2007.50 20721
        2007.75 22641
        2008.00 21647
        2008.25 22370
        2008.50 22282
        2008.75 24484

    Implementujte funkci x = fit_wages(t,M) kde t a M jsou vektory délky m s časy a mzdami, a x je vektor délky 2 s parametry (x1,x2)  
    Cílem je přibližně předpovědět hodnotu PHM v časech, pro které není hodnota PHM známá. To uděláme tak, že nejprve nalezneme funkci, která co nejlépe odpovídá zadaným údajům o PHM, a tuto funkci pak použijeme pro odhad PHM v požadovaném čase. Z grafu je vidět, že závislost PHM na čase je téměř lineární. Tudíž budeme hledat lineární funkci 
    M(t) = x1 + x2*t
    kde M(t) je odhad PHM v čase t a x1,x2∈R jsou parametry. Náš naměřený vzorek označíme {(t1,M1),…,(tm,Mm)} obsahuje m dvojic (čas,PHM). Optimální parametry nalezneme z tohoto vzorku ve smyslu nejmenších čtverců, tj. tak, aby součet kvadrátů odchylek skutečné a odhadnuté mzdy byl v naměřených bodech minimální. To znamená, minimalizujeme funkci 
    sum_{i=1}^{m} (M(ti)-Mi)^2

    Implementujte funkci M = quarter2_2009(x), která pro parametry x odhadnuté funkcí fit_wagesspočítá odhad PHM ve druhém kvartálu roku 2009. 

    Očekávaný výstup:
    2000.00 -> 11941
    2000.25 -> 13227
    2000.50 -> 12963
    2000.75 -> 14717
    2008.50 -> 22282
    2008.75 -> 24484
    Implementujte vše v pythonu.
"""
state = GraphState()
state["prompt"] = QUESTION4
state["previous_result"] = "None"
state["previous_code"] = "None"
state["failedTimes"] = 0

result = chatBot.invoke(state)

print(result)
print(result["answer"])