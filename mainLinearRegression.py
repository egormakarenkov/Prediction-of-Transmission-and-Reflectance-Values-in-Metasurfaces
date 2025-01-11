# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from Excel
data = pd.read_excel('New_data_02.xlsx', skiprows=3)  # Skip the first three rows

# Rename columns to letters
data.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Drop columns with no data
data = data.drop(columns=['d', 'h', 'i', 'k', 'm', 'n', 'p', 'r', 's', 'u', 'v', 'x', 'y'])

# Extract features and target variables
X = data[['a', 'b', 'c', 'e', 'f', 'g']]  # Features
T = data['q']  # Transmittance (T)
R = data['l']  # Reflectance (R)

# Split data into training and testing sets
X_train, X_test, T_train, T_test, R_train, R_test = train_test_split(X, T, R, test_size=0.2, random_state=42)

# Create linear regression models for T and R
model_T = LinearRegression()
model_R = LinearRegression()

# Train the models
model_T.fit(X_train, T_train)
model_R.fit(X_train, R_train)

# Make predictions
T_pred = model_T.predict(X_test)
R_pred = model_R.predict(X_test)

# Evaluate the models
mse_T = mean_squared_error(T_test, T_pred)
r2_T = r2_score(T_test, T_pred)

mse_R = mean_squared_error(R_test, R_pred)
r2_R = r2_score(R_test, R_pred)

print(f"Transmission Model - MSE: {mse_T}, R-squared: {r2_T}")
print(f"Reflection Model - MSE: {mse_R}, R-squared: {r2_R}")

# New data for testing (replace with actual values)
new_data = pd.DataFrame({
    'a': [5.75, 5, 6],  # Example values
    'b': [0.000000000000202, 0.000000000000202, 0.000000000000255],
    'c': [0.000000000005, 0.000000000005, 0.000000000005],
    'e': [0.00000000078, 0.000000000727, 0.00000000078],
    'f': [1.6, 10, 10],
    'g': [20, 10, 0]
})

# Predict for new data
T_new_pred = model_T.predict(new_data)
R_new_pred = model_R.predict(new_data)

# Print the predictions for the new data
for i, (t_pred, r_pred) in enumerate(zip(T_new_pred, R_new_pred)):
    print(f"Set {i+1} - Predicted Transmittance: {t_pred}, Predicted Reflectance: {r_pred}")
