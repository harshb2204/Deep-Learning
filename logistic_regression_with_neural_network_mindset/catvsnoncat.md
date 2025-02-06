# Cat vs Non-Cat Image Classification Problem

## Problem Overview

You are tasked with building a simple image-recognition algorithm to classify images as either a cat (`y=1`) or non-cat (`y=0`). The dataset provided, `data.h5`, consists of:

- A training set (`m_train`) of labeled images.
- A test set (`m_test`) of labeled images.
- Images are square-shaped with dimensions `(num_px, num_px, 3)` where 3 represents the RGB channels.

The objective is to preprocess the dataset and use it to train a model capable of identifying cats in images.

---

## Dataset Details

### Loading the Dataset
The dataset can be loaded using the following code:
```python
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]


print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))
```


## Exercise 2: Reshaping the Data

To prepare the data for the model, the training and test datasets are reshaped so that each image (of shape `(num_px, num_px, 3)`) is flattened into a single vector of shape `(num_px * num_px * 3, 1)`. 

This ensures the data is ready for the neural network input.

---

### Code
```python
# Reshape the training and test examples

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T


# Check that the first 10 pixels of the second image are in the correct place
assert np.alltrue(train_set_x_flatten[0:10, 1] == [196, 192, 190, 193, 186, 182, 188, 179, 174, 213]), "Wrong solution. Use (X.shape[0], -1).T."
assert np.alltrue(test_set_x_flatten[0:10, 1] == [115, 110, 111, 137, 129, 129, 155, 146, 145, 159]), "Wrong solution. Use (X.shape[0], -1).T."

# Print the shapes of the datasets
print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))

```


## Preprocessing the Dataset

In machine learning, preprocessing the dataset is a critical step to ensure better model performance and convergence. For image datasets, standardizing the pixel values simplifies the input and improves learning.

---

### RGB Representation of Images
- Each pixel in an image is represented by three values: red, green, and blue (RGB channels).
- These values range from 0 to 255.

---

### Preprocessing Step: Standardization
Instead of centering and normalizing by subtracting the mean and dividing by the standard deviation, a simpler approach is to divide all pixel values by 255. This rescales the data to a range between 0 and 1, which is sufficient for most image-recognition tasks.

---

### Code for Standardization
```python
# Standardize the dataset
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.
```

# 4.2 - Initializing Parameters



Implement parameter initialization in the cell below. You have to initialize `w` as a vector of zeros. If you don't know what numpy function to use, look up `np.zeros()` in the Numpy library's documentation.

```python
import numpy as np

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias) of type float
    """

    # Corrected code
    w = np.zeros((dim, 1))  # Added the missing parenthesis
    b = 0.0  # Initialize bias as a float scalar

    return w, b

# Example usage
dim = 2
w, b = initialize_with_zeros(dim)

assert type(b) == float  # Check if b is a float
print("w = " + str(w))
print("b = " + str(b))

# Test the function
initialize_with_zeros_test_1(initialize_with_zeros)
initialize_with_zeros_test_2(initialize_with_zeros)

```
# Propagate Function for Forward and Backward Propagation

## Introduction
The `propagate()` function computes the cost function and its gradient during forward and backward propagation. This is an essential step in training a logistic regression model using gradient descent.

### The following steps are covered:
1. **Forward Propagation**: Compute the activation values and the cost function.
2. **Backward Propagation**: Compute the gradients of the weights and bias.

## Forward Propagation

The forward propagation involves the following steps:
1. **Compute Activation**: The activation function used is the sigmoid function:
   \[
   A = \sigma(w^T X + b)
   \]
   Where:
   - \( w \) is the weight vector.
   - \( X \) is the input data.
   - \( b \) is the bias.

2. **Compute Cost Function**: The cost function for logistic regression is the negative log-likelihood function:
   \[
   J = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(a^{(i)}) + (1 - y^{(i)}) \log(1 - a^{(i)}) \right]
   \]
   Where:
   - \( m \) is the number of examples.
   - \( a^{(i)} \) is the activation for the \( i \)-th example.
   - \( y^{(i)} \) is the true label for the \( i \)-th example.

## Backward Propagation

For backward propagation, the gradients are calculated as follows:
1. **Gradient of Weights**:
   \[
   \frac{\partial J}{\partial w} = \frac{1}{m} X (A - Y)^T
   \]
   Where:
   - \( X \) is the input data.
   - \( A \) is the activation.
   - \( Y \) is the true labels.

2. **Gradient of Bias**:
   \[
   \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)} - y^{(i)})
   \]

## Code Implementation

```python
import numpy as np

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    grads -- dictionary containing the gradients of the weights and bias
            (dw -- gradient of the loss with respect to w, thus same shape as w)
            (db -- gradient of the loss with respect to b, thus same shape as b)
    cost -- negative log-likelihood cost for logistic regression
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = 1 / (1 + np.exp(-(np.dot(w.T, X) + b)))  # Compute activation
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # Compute cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    
    cost = np.squeeze(cost)  # Ensures cost is a scalar
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
```
# Optimization with Gradient Descent

This section demonstrates the implementation of the optimization function using gradient descent to minimize the cost function \( J \). The function optimizes weights (\( w \)) and bias (\( b \)) over several iterations.

## GRADED FUNCTION: `optimize`

### Implementation

The `optimize` function follows these steps:
1. **Calculate the cost and gradient** using the `propagate` function.
2. **Update the parameters** \( w \) and \( b \) using the gradient descent update rule:  
   \[
   w = w - \alpha \cdot dw
   \]  
   \[
   b = b - \alpha \cdot db
   \]  
   where \( \alpha \) is the learning rate.

### Function Code

```python
def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # Update rule
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
            
            # Print the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
```
