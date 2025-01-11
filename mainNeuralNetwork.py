# Import necessary libraries
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Update with the path to your data file
data_file = 'New_data_02.xlsx'

# Read the Excel file, skipping the first three rows to account for the header information
# Remove columns J, O, W, and Z
data = pd.read_excel(data_file, skiprows=3, usecols='A:C,E:G,L,Q,T')

# Define and assign column names based on your knowledge of the columns
data.columns = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'L', 'Q', 'col9']

# Print the initial shape of the data
print("Initial data shape:", data.shape)

# Print column names to verify
print("Column names:", data.columns.tolist())

# Print non-numeric columns
non_numeric_cols = data.select_dtypes(exclude=[float, int]).columns
print("Non-numeric columns:", non_numeric_cols)

# Handle non-numeric columns: converting to numeric and encoding categorical columns
for col in non_numeric_cols:
    if col == 'Label':  # Assuming 'Label' is categorical
        data[col] = data[col].astype('category').cat.codes
    else:  # Convert other columns to numeric
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Print shape after converting non-numeric columns
print("Shape after converting non-numeric columns:", data.shape)

# Drop rows with NaN values
data = data.dropna()

# Print shape after dropping NaN values
print("Shape after dropping NaN values:", data.shape)

# Check if data is empty after preprocessing
if data.empty:
    raise ValueError("The dataset is empty after preprocessing. Check the data and preprocessing steps.")

# Extract features and target variables from the DataFrame
X = data.drop(columns=['L', 'Q']).values  # Use all columns except L and Q as features
y = data[['L', 'Q']].values  # Use columns L and Q as targets

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a neural network model with two output units
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2)  # Two output units for predicting L and Q
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# New data for testing (replace with actual values)
new_data = pd.DataFrame({
    'col1': [5.75, 5, 6],  # Example values
    'col2': [0.000000000000202, 0.000000000000202, 0.000000000000255],
    'col3': [0.000000000005, 0.000000000005, 0.000000000005],
    'col4': [0.00000000078, 0.000000000727, 0.00000000078],
    'col5': [1.6, 10, 10],
    'col6': [20, 10, 0],
    'col9': [0, 0, 0]  # Add col9 as well since it's part of the features
})

# Standardize the new data
new_data_scaled = scaler.transform(new_data)

# Predict for new data
new_predictions = model.predict(new_data_scaled)

# Print the predictions for the new data
for i, (L_pred, Q_pred) in enumerate(new_predictions):
    print(f"Set {i+1} - Predicted L: {L_pred}, Predicted Q: {Q_pred}")

