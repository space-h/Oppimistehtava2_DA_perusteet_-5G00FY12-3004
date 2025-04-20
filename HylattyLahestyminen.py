import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('global_traffic_accidents.csv', delimiter=',', quotechar='"')

# Drop unnecessary columns
df = df.drop(['Accident ID', 'Latitude', 'Longitude', 'Date', 'Time'], axis=1)

# Focus on Toronto only
df_toronto = df[df['Location'] == 'Toronto, Canada'].drop(['Location'], axis=1)

# Separate features and target
X = df_toronto.drop(['Casualties'], axis=1)
y = df_toronto['Casualties']

# Dummyfying Weather, Road and Cause
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(sparse_output=False, drop="first"), ["Weather Condition", "Road Condition", "Cause"])],
    remainder="passthrough"
)

X_transformed = ct.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f"Linear Regression R² score: {score:.2f}")

# Predict on test data
y_pred = model.predict(X_test)

# Plot predictions vs real values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='green', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)  # Ideal line
plt.xlabel('Actual Casualties')
plt.ylabel('Predicted Casualties')
plt.title('Linear Regression Predictions vs Actual Casualties')
plt.grid(True)
plt.show()
