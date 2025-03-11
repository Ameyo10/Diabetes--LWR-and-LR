from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random

# Load dataset
diabetes = load_diabetes()

# Convert to DataFrame
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

# df['sex'] = df['sex'].apply(lambda x: 'male' if np.isclose(x, 0.050680) else 'female')

# Add target variable
df['target'] = diabetes.target

# Displaying the Dataset Info
def df_info():
    # Display first 5 rows
    print(df.head())

    # Display the column types and non- null values
    print (df.info())

    #Display the statistical summary
    print(df.describe())

    #Displaying the shape of the dataframe
    print(df.shape)

# Visualizing the relationship between bmi and target
def bmi_vs_target_vis():
    df['sex'] = df['sex'].apply(lambda x: 'male' if np.isclose(x, 0.050680) else 'female')
    sns.scatterplot(data = df, x= 'bmi', y= 'target', size  = 'target', style = 'sex')
    plt.title('Progression of Diabetes Annually', fontweight = 'bold', fontsize = 20)
    plt.xlabel('BMI of the patient', fontsize = 12)
    plt.ylabel('Disease Progression', fontsize = 12)
    plt.legend()
    plt.grid(True, alpha = 0.5)
    plt.show()

    plt.figure(figsize=(10, 6))
    # Changing the sex column, previously string into numeric
    df['sex_numeric'] = df['sex'].map({'male': 1, 'female': 0})

    # Computing the correlation matrix
    corr_matrix = df.drop(columns = ['sex']).corr()

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation with Target")
    plt.show()


def locally_weighted_regression(X_train, y_train, X_query, tau=0.1):
    """Performs Locally Weighted Regression (LWR) for a given bandwidth (tau)."""
    m = len(X_train)
    X_train_bias = np.c_[np.ones((m, 1)), X_train]  # Add bias term
    y_pred = []

    for x_q in X_query:
        # Compute weights using Gaussian kernel
        diff = X_train - x_q  # Now works for multiple features
        weights = np.exp(-np.sum(diff ** 2, axis=1) / (2 * tau ** 2))
        W = np.diag(weights)  # Convert to diagonal matrix

        # Compute LWR solution: θ = (X^T W X + λI)^(-1) X^T W y
        reg = 1e-5 * np.eye(X_train_bias.shape[1])  # Small regularization term
        theta = np.linalg.pinv(X_train_bias.T @ W @ X_train_bias + reg) @ X_train_bias.T @ W @ y_train
        y_pred.append(np.array([1, *x_q]) @ theta)  # Handle multiple features

    return np.array(y_pred), theta

def three_D_plot(X, y, y_pred_lwr, y_pred_lr, tau, feature1_name, feature2_name):
    """Creates a 3D scatter plot with LWR and LR predictions."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of actual data points
    ax.scatter(X[:, 0], X[:, 1], y, color='red', label='Actual Data Points')

    # Plotting the LWR regression surface
    ax.plot_trisurf(X[:, 0], X[:, 1], y_pred_lwr, alpha=0.5, color='cyan', label='LWR Predictions')

    # Plotting the LR regression surface
    ax.plot_trisurf(X[:, 0], X[:, 1], y_pred_lr, alpha=0.5, color='green', label='LR Predictions')

    # Labels and title
    ax.set_xlabel(f'Feature 1: {feature1_name}')
    ax.set_ylabel(f'Feature 2: {feature2_name}')
    ax.set_zlabel('Target')
    ax.set_title(f'LWR vs LR Predictions (Tau={tau})')

    ax.legend()
    plt.show()

def lwr():
    """Main function to execute LWR and visualize results."""
    # Standardizing features
    scaler = StandardScaler()
    df[['bmi', 'age', 'bp', 'sex']] = scaler.fit_transform(df[['bmi', 'age', 'bp', 'sex']])  

    # Splitting features and target
    X1 = df[['bmi']].values  # Feature1
    X2 = df[['age']].values  # Feature2
    X3 = df[['bp']].values   # Feature3
    X4 = df[['sex']].values  # Feature4
    y = df['target'].values  # Target
    
    # Different bandwidths for LWR
    tau_values = [0.5, 1, 5, 10]

    # Randomly select 2 feature pairs
    feature_combinations = [
        (X1, X2, 'bmi', 'age'),
        (X1, X3, 'bmi', 'bp'),
        (X1, X4, 'bmi', 'sex'),
        (X2, X3, 'age', 'bp'),
        (X2, X4, 'age', 'sex'),
        (X3, X4, 'bp', 'sex')
    ]
    selected_features = random.sample(feature_combinations, 2)  

    for features in selected_features:
        X = np.c_[features[0], features[1]]  # Combine selected features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for tau in tau_values:
            # Compute LWR predictions for the entire dataset
            y_pred_lwr, theta = locally_weighted_regression(X_train, y_train, X, tau)
            
            # Train a Linear Regression model for comparison
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            y_pred_lr = lr_model.predict(X)
            
            # Plot the results
            three_D_plot(X, y, y_pred_lwr, y_pred_lr, tau, features[2], features[3])

# df_info()
# bmi_vs_target_vis()
lwr()

