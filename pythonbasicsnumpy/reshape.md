# image2vector Implementation

This Python function reshapes an input image array of shape `(length, height, depth)` into a column vector of shape `(length * height * depth, 1)`.

---

### Function: `image2vector(image)`

```python
import numpy as np

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    v = image.reshape(-1, 1)  # Reshapes the input to a vector of shape (length*height*depth, 1)
    return v
