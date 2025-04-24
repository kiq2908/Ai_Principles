# Team Name: Quoc Huynh, Yale Miller and Alex Pham
# Assignment 5 - Linear Regression using Normal Equation
# Course: 20CS4033 - Spring 2025
# Instructor: Prof. A. Ralescu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load and preprocess the data
df = pd.read_csv("insurance.csv")
#drop categorical columns: sex, smoker, reggion
df_numeric = df.drop(columns=["sex", "smoker", "region"])

df_numeric.head()
X = df_numeric.drop(columns="charges").values
y = df_numeric["charges"].values

#train sizes for experiment
train_sizes = [0.2, 0.4, 0.5, 0.6, 0.8]
train_errors = []
test_errors = []

#compute weight using the normal equation
def linear_regression(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

#predict given weights
def predict(X, weights):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b @ weights

#loop over different training set sizes
for size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=size, random_state=42
    )
    w = linear_regression(X_train, y_train)
    #print the learned weights for reporting
    print("Weights (intercept, age, bmi, children):", w)
    y_train_pred = predict(X_train, w)
    y_test_pred = predict(X_test, w)
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

print("Training and Test MSE for different training set sizes:")
for size, tr_err, te_err in zip(train_sizes, train_errors, test_errors):
    print(f"Train size {int(size*100)}%: Training MSE = {tr_err:.2f}, Test MSE = {te_err:.2f}")


plt.figure()
plt.plot(
    [int(s * 100) for s in train_sizes], train_errors, label="Training MSE", marker="o"
)
plt.plot([int(s * 100) for s in train_sizes], test_errors, label="Test MSE", marker="o")
plt.xlabel("Training Set Size (%)")
plt.ylabel("Mean Squared Error")
plt.title("Modeling vs Generalization Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("error_plot.png")

#50% Training regression line visualization
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.5, random_state=42
)
w = linear_regression(X_train, y_train)

feature_names = ["age", "bmi", "children"]
for i, name in enumerate(feature_names):
    plt.figure()
    plt.scatter(X_test[:, i], y_test, color="blue", label="Actual")
    
    feature_range = np.linspace(X_test[:, i].min(), X_test[:, i].max(), 100)
    X_line = np.tile(X_test.mean(axis=0), (100, 1))  
    X_line[:, i] = feature_range                   
    y_line = predict(X_line, w)                    

    # Plot regression line (sorted)
    plt.plot(feature_range, y_line, label="Regression line")
    plt.xlabel(name)
    plt.ylabel("Charges")
    plt.title(f"Regression Line vs {name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"regression_vs_{name}.png")

