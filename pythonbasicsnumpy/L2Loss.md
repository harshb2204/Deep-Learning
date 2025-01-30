# Implementation of L2 Loss Function

## Overview

The L2 loss function, also known as Mean Squared Error (MSE) or quadratic loss, measures the average squared difference between the predicted values (\( yhat \)) and true values (\( y \)). It is commonly used in regression tasks to evaluate model performance.

The formula for the L2 loss function is:

\[
L2(yhat, y) = \sum_{i=0}^{m-1} \left(y^{(i)} - \hat{y}^{(i)}\right)^2
\]

where:
- \( yhat \) is the vector of predicted values.
- \( y \) is the vector of true values.
- \( m \) is the number of samples.

The L2 loss penalizes larger errors more heavily due to the square of the differences.

## Python Implementation

Below is the Python implementation of the L2 loss function using NumPy:

```python
import numpy as np

# GRADED FUNCTION: L2

def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    # Calculate the L2 loss
    loss = np.sum((yhat - y)**2)
    
    return loss

# Test the function
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat, y)))
