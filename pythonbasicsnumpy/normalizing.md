# normalize_rows Implementation

This Python function normalizes each row of a matrix to have unit length.

---

### Function: `normalize_rows(x)`

```python
import numpy as np

def normalize_rows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)  # Compute the L2 norm of each row
    x = x / x_norm  # Normalize each row by dividing by its norm
    return x
