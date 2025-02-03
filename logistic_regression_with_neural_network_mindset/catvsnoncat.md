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
