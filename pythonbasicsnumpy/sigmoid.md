# Sigmoid Function Implementation

![sigmoid](images/sigmoid.png)


This Python function computes the sigmoid of a scalar input `x`.

### Function: `basic_sigmoid(x)`

```python
import math

def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + math.exp(-x))
    return s
    

