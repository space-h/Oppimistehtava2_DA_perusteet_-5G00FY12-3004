# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 21:15:11 2025

@author: jonne
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor



# Load the CSV file
df = pd.read_csv('global_traffic_accidents.csv', delimiter=',', quotechar='"')

# Display check
print(df.head())

#drop useless data
df = df.drop(['Accident ID', 'Latitude', 'Longitude'], axis=1)
df = df.drop(['Date'], axis=1) 
df = df.drop(['Time'], axis=1)
df = df.drop(['Location'], axis=1)



#Data processing. Dummyfying Weather, Road and Cause
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(sparse_output=False, drop="first"), ["Weather Condition", "Road Condition", "Cause"])], remainder="passthrough")
data_transformed = ct.fit_transform(df)

letsTry = pd.DataFrame(data_transformed)

#Ennustetaan Vehicles Involved arvoa
X = letsTry.drop(letsTry.columns[15], axis=1) #0-14 dummmies, 16 Casualties
y = letsTry.iloc[:, 15] #Vehicles involved



# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=2)

# Train the model
model.fit(X_train, y_train)

#-----------------Feature Importance ------------------------

# importances = model.feature_importances_

# # If you used ColumnTransformer + OneHotEncoder, get the correct feature names
# encoded_features = ct.named_transformers_["encoder"].get_feature_names_out(["Weather Condition", "Road Condition", "Cause"])
# # Assuming the rest of your features are the leftover numeric columns
# remainder_features = df.drop(['Vehicles Involved'], axis=1).columns.difference(["Weather Condition", "Road Condition", "Cause"])

# # Combine feature names into one list (OneHotEncoded + remainder)
# feature_names = list(encoded_features) + list(remainder_features)

# # Create a DataFrame to display them neatly
# importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# # Print the ranked list
# print("\nFeature Importances:")
# print(importance_df.to_string(index=False))
#-----------------Feature Importance ends ------------------------


y_pred = model.predict(X_test)


# # Print predictions vs actual
# for i in range(len(y_pred)):
#     print(f"Predicted: {round(y_pred[i])}, Actual: {y_test.iloc[i]}")

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Random Forest): {mse}")

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
print(f"Incorrect 1: {incorrect_by_one}")
print(f"Incorrect by more than 1: {incorrect_by_more_than_one}")
del correct_predictions, incorrect_by_one, incorrect_by_more_than_one, difference

# Predict new data
new_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  # Adjust length if needed
predicted_value = model.predict(new_data)
print(f"Predicted value for {new_data}: {predicted_value[0]}")

