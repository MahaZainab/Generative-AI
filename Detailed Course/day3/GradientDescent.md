# Gradient Descent in Machine Learning

Gradient Descent is an optimization algorithm used to minimize the cost function in machine learning models. It is a fundamental tool for adjusting model parameters to improve accuracy and performance.

## Key Concepts

### What is Gradient Descent?

Gradient Descent iteratively adjusts model parameters to minimize a cost function, which measures the difference between predicted and actual values. The algorithm uses the gradient of the cost function to guide updates in the parameters.

### How Does It Work?

1. **Initialization**: Begin with initial parameter guesses.
2. **Compute the Gradient**: Calculate partial derivatives of the cost function.
3. **Update the Parameters**: Adjust parameters opposite to the gradient direction.
4. **Convergence Check**: Repeat until convergence.

### Types of Gradient Descent

- **Batch Gradient Descent**: Uses the entire dataset for each step.
- **Stochastic Gradient Descent (SGD)**: Uses one random sample per step.
- **Mini-Batch Gradient Descent**: Uses a small batch of samples per step.

## Advantages and Challenges

**Advantages**:
- Simple to implement and understand.
- Applicable to various machine learning models.

**Challenges**:
- Selecting an appropriate learning rate is crucial.
- Potential to get stuck in local minima.

## Example in Python

```python
# Pseudo-code for Gradient Descent
def gradient_descent(X, y, learning_rate, epochs):
    m = len(y)
    theta = np.zeros(X.shape[1])
    for epoch in range(epochs):
        gradient = (1/m) * X.T.dot(X.dot(theta) - y)
        theta -= learning_rate * gradient
    return theta
