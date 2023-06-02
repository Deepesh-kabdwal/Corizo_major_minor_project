import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv(r'C:\Users\deepe\OneDrive\Desktop\corizo project major and minor\major project\wine_data.csv')

# Split the dataset into input (X) and output (y) variables
X = data.drop('quality', axis=1)
y = data['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Outlier detection using Isolation Forest algorithm
outlier_detector = IsolationForest(contamination=0.05)  # Adjust contamination parameter as needed
outliers = outlier_detector.fit_predict(X_train)
X_train = X_train[outliers == 1]
y_train = y_train[outliers == 1]

# Feature selection using SelectKBest with f_regression scoring
feature_selector = SelectKBest(f_regression, k=5)  # Adjust k value as needed
X_train_selected = feature_selector.fit_transform(X_train, y_train)
X_test_selected = feature_selector.transform(X_test)

# Train a linear regression model on the selected features
regressor = LinearRegression()
regressor.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test_selected)

# Evaluate the model's performance using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
