"""
This script builds a simple linear regression model to predict house prices
based on the size of the house and the number of bedrooms. It then visualizes
the model's predictions with a 3D plot, showing the relationship between the
features and the predicted price.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- 1. Data Loading and Preparation ---
# Load the dataset from a remote CSV file using pandas.
# The file contains three columns: size, number of bedrooms, and price.
url = 'https://raw.githubusercontent.com/enuguru/aiandml/refs/heads/master/machine_learning_algorithms_using_frameworks/python_files/regression/house_price_prediction/home.csv'
data = pd.read_csv(url, header=None, names=['size', 'bedrooms', 'price'])

# Extract the features (size, bedrooms) and the target variable (price).
size = data['size'].values
bedrooms = data['bedrooms'].values
# Scale the price to be in thousands
price = data['price'].values / 1000

# Create the feature matrix X and the target vector y.
X = np.array([size, bedrooms]).T
y = price

# --- 2. Data Splitting ---
# Split the data into training and testing sets (70% training, 30% testing).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 3. Model Training ---
# Initialize and train a linear regression model on the training data.
model = LinearRegression()
model.fit(X_train, y_train)

# --- 4. Model Evaluation ---
# Evaluate the model's performance on the test set.
score = model.score(X_test, y_test)

# --- 5. Data Preparation for 3D Visualization ---
# Create a meshgrid of feature values to generate a prediction surface.
size_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
bedrooms_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
size_grid, bedrooms_grid = np.meshgrid(size_range, bedrooms_range)

# Predict the house prices for each point on the meshgrid.
price_pred = model.predict(np.c_[size_grid.ravel(), bedrooms_grid.ravel()])
price_pred_grid = price_pred.reshape(size_grid.shape)

# --- 6. 3D Plot Generation ---
# Create a new figure and a 3D subplot.
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot of the training data.
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='blue', marker='o', s=50, label='Training Data')

# Create a scatter plot of the test data.
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c='red', marker='^', s=50, label='Test Data')

# Plot the regression plane (the model's prediction surface).
ax.plot_surface(size_grid, bedrooms_grid, price_pred_grid, alpha=0.3, cmap='viridis', label='Predicted Price Plane')

# --- 7. Plot Customization ---
# Set the labels for the x, y, and z axes.
ax.set_xlabel('Size (sq ft)', fontsize=12)
ax.set_ylabel('Number of Bedrooms', fontsize=12)
ax.set_zlabel('Price ($1000s)', fontsize=12)

# Set a descriptive title for the plot.
fig.suptitle('3D Visualisation of House Price Prediction Model', fontsize=16)

# Add the R^2 score to the plot.
ax.text2D(0.05, 0.95, f'Model R^2 score on test data: {score:.2f}', transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

# Add a legend to identify the data points and the prediction surface.
ax.legend()

# Display the plot.
plt.show()