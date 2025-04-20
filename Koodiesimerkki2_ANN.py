import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('global_traffic_accidents.csv', delimiter=',', quotechar='"')

#printtimuuttuja
whereWeAt = "All Locations"


#Helpottaa printtiä kun tallentaa muuttujaksi
epokit = 100

#kommentoi pois niin saa kaikki sijainnit
whereWeAt = 'Toronto, Canada'
df = df[df['Location'] == 'Toronto, Canada']
#Tähän asti


# Drop unnecessary columns
df = df.drop(['Accident ID', 'Latitude', 'Longitude', 'Date', 'Time', 'Location'], axis=1)

# Dummyfying Weather, Road and Cause
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(sparse_output=False, drop="first"), ["Weather Condition", "Road Condition", "Cause"])],
    remainder="passthrough"
)

data_transformed = ct.fit_transform(df)
letsTry = pd.DataFrame(data_transformed)

# #Jos haluaa ennustaa Casualties Vehicles Involved sijaan
# # Define features (X) and target (y) for predicting Casualties
# X = letsTry.drop(letsTry.columns[16], axis=1)  # Drop 'Casualties' column
# y = letsTry.iloc[:, 16]  # Casualties



# Define features (X) and target (y) for predicting V-I
X = letsTry.drop(letsTry.columns[15], axis=1)  #0-14 dummmies, 16 Casualties
y = letsTry.iloc[:, 15]  # Vehicles Involved



# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Build the ANN model
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=epokit, batch_size=8, validation_split=0.2, verbose=1)


# Predict
y_pred = model.predict(X_test).flatten()
y_pred_rounded = np.round(y_pred)

# Print predictions vs actual
for i in range(len(y_pred)):
    print(f"Predicted: {round(y_pred[i])}, Actual: {y_test.iloc[i]}")
    
#What We doing here?
print(f"\nClarification: Vehicles Involved, epochs={epokit}, {whereWeAt} \n")


# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (ANN): {mse}")

# Initialize counters for the categories
correct_predictions = 0
incorrect_by_one = 0
incorrect_by_more_than_one = 0

# Compare predictions with actual values
for i in range(len(y_pred)):
    difference = abs(round(y_pred[i]) - y_test.iloc[i])
    
    if difference == 0:
        correct_predictions += 1
    elif difference <= 1:
        incorrect_by_one += 1
    else:
        incorrect_by_more_than_one += 1

# Print the results
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect by 1: {incorrect_by_one}")
print(f"Incorrect by more than 1: {incorrect_by_more_than_one}")
del correct_predictions, incorrect_by_one, incorrect_by_more_than_one, difference

#tuntiesimerkki opetuksen arvioinnista
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.ylim(bottom=0, top=5 * min(history.history['val_loss']))
plt.grid(True)
plt.title(f'ANN-malli, Vehicles Involved, Epochs = {epokit}, {whereWeAt}')
plt.show()

