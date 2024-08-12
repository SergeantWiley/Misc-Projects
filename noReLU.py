import numpy as np
import matplotlib.pyplot as plt

# Generate some data
np.random.seed(42)  # For reproducibility
predictions = np.random.uniform(-5, 5, 100)
print(predictions[10])  # 100 random prediction values between -5 and 5
target_value = 2  # Fixed target value for the standard loss function
target_value_modified = 2  # Fixed target value for the modified loss function

# Define the loss functions
def loss_function(theta):
    return (theta + target_value) ** 2

def modified_loss_function(theta):
    return (theta - target_value_modified) ** 2

# Calculate the loss values
loss_values = loss_function(predictions)
modified_loss_values = modified_loss_function(predictions)

# Plotting the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(predictions, loss_values, color='blue', label='Loss without ReLU')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(target_value, color='red', lw=0.5, ls='--', label='Target Value (Theta=2)')
plt.title('Quadratic Loss Function')
plt.xlabel('Predictions')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(predictions, modified_loss_values, color='orange', label='Loss with ReLU')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(target_value_modified, color='red', lw=0.5, ls='--', label='Target Value (Theta=2)')
plt.title('Modified Quadratic Loss Function')
plt.xlabel('Predictions')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
