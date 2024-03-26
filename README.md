# SmartCrop-
SmartCrop は、エッジコンピューティングと機械学習を活用して、農作物の病害を早期に予測し、農家に警告を発信することで、収穫量の最大化を図ります。
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Simulated dataset: Features might include temperature, humidity, soil pH, etc.
# Target is whether the crop is diseased (1) or not (0)
data = {
    'temperature': [23, 25, 20, 18, 27, 24, 22, 30, 28, 21],
    'humidity': [80, 85, 70, 90, 75, 82, 88, 65, 70, 95],
    'soil_ph': [6.5, 7.0, 5.5, 7.5, 6.0, 7.2, 6.8, 5.8, 6.4, 7.1],
    'diseased': [0, 1, 0, 1, 0, 0, 1, 0, 1, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['temperature', 'humidity', 'soil_ph']]
y = df['diseased']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Example prediction
# Let's predict the disease status for a new data point
new_data = [[24, 80, 6.7]]  # temperature, humidity, soil pH
prediction = model.predict(new_data)
if prediction == 1:
    print("Warning: The crop is likely diseased.")
else:
    print("The crop is healthy.")
