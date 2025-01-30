# Implementation of L1 Loss Function

## Overview

The L1 loss function, also known as the Mean Absolute Error, measures the average magnitude of errors in predictions without considering their direction. It is defined as:

\[
L1(yhat, y) = \sum_{i=0}^{m-1} |y^{(i)} - \hat{y}^{(i)}|
\]

where:
- \( yhat \) is the vector of predicted values.
- \( y \) is the vector of true values.
- \( m \) is the number of samples.

The L1 loss is used to evaluate the performance of a model by calculating the total absolute difference between predicted values and actual values.

## Python Implementation

```python
import numpy as np

def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    # Calculate the L1 loss
    loss = np.sum(np.abs(yhat - y))
    
    return loss

# Test the function
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat, y)))
