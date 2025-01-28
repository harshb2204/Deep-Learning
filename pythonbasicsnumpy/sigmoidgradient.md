# Sigmoid Derivative Implementation



This Python function computes the **gradient of the sigmoid function** with respect to its input.

---

### Function: `sigmoid_derivative(x)`

```python
import numpy as np

def sigmoid_derivative(x):
    """
    Compute the gradient of the sigmoid function with respect to x.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Gradient of sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds
