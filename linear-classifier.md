A linear classifier is a type of machine learning model that makes predictions by creating a linear decision boundary between different classes. The decision boundary is represented by a hyperplane in the feature space, and the classification is determined based on which side of the hyperplane a data point lies. Linear classifiers are particularly effective when the classes are linearly separable, meaning they can be separated by a straight line (or hyperplane).

### Mathematical Representation:

Given a feature vector \( \mathbf{x} = [x_1, x_2, \ldots, x_n] \) and corresponding weights \( \mathbf{w} = [w_1, w_2, \ldots, w_n] \), the decision function of a linear classifier can be expressed as:

\[ f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b \]

Here, \( \mathbf{w} \cdot \mathbf{x} \) represents the dot product of the weight vector and the input vector, and \( b \) is the bias term. If \( f(\mathbf{x}) \) is greater than or equal to zero, the input is classified as one class; otherwise, it is classified as the other class.

### Example:

Let's consider a simple example with two features (\( x_1, x_2 \)) and a binary classification problem.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random linearly separable data
np.random.seed(42)
X = np.random.rand(100, 2) * 2 - 1  # Random points in the range [-1, 1]
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Linear decision boundary

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.title("Linearly Separable Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

In this example, the decision boundary is determined by the line \(x_1 + x_2 = 0\). Points above the line belong to one class (positive), and points below the line belong to the other class (negative).

### Code for Linear Classifier:

Now, let's implement a simple linear classifier using the Perceptron algorithm:

```python
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations


Continuing from the previous code snippet, here is the implementation of the Perceptron algorithm for a simple linear classifier:

```python
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def train(self, X, y):
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iterations):
            for i in range(X.shape[0]):
                prediction = np.dot(self.weights, X[i]) + self.bias
                if prediction >= 0:
                    y_pred = 1
                else:
                    y_pred = 0

                # Update weights and bias using the Perceptron learning rule
                self.weights += self.learning_rate * (y[i] - y_pred) * X[i]
                self.bias += self.learning_rate * (y[i] - y_pred)

    def predict(self, X):
        # Return predictions for input data
        return np.dot(X, self.weights) + self.bias
```

Now, you can use this `Perceptron` class to train and predict on the previously generated data:

```python
# Train the perceptron
perceptron = Perceptron()
perceptron.train(X, y)

# Plot the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.title("Linear Classifier Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Plot decision boundary line
x_line = np.linspace(-1, 1, 100)
y_line = (-perceptron.weights[0] * x_line - perceptron.bias) / perceptron.weights[1]
plt.plot(x_line, y_line, 'r--')

plt.show()
```

This will visualize the linear decision boundary created by the trained Perceptron on the linearly separable data. The red dashed line represents the decision boundary, and points on one side of the line belong to one class, while points on the other side belong to the other class.
