# Step 1: Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load dataset
df = pd.read_csv("/content/RS-A4_SEER Breast Cancer Dataset .csv")

# Step 3: Clean and preprocess data
# Drop unnamed or irrelevant columns
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')

# Encode target column (Alive = 1, Dead = 0)
df['Status'] = df['Status'].map({'Alive': 1, 'Dead': 0})

# Drop missing values
df = df.dropna()

# Separate features and target
X = df.drop(columns=['Status'])
y = df['Status']

# Convert categorical variables to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("🎯 Model Accuracy:", round(accuracy, 2))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Visualize the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Dead (0)', 'Alive (1)'], yticklabels=['Dead (0)', 'Alive (1)'])
plt.title("🧩 Confusion Matrix - Breast Cancer Prognosis")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Step 8: Recommendation function
def prognosis_recommendation(features):
    """
    Function to provide prognosis recommendation based on model predictions.
    :param features: Array of patient features (must match X columns)
    :return: Recommendation string
    """
    prediction = model.predict([features])
    if prediction[0] == 0:
        return "⚠️ High risk of malignant cancer. Immediate consultation and further tests recommended."
    else:
        return "✅ Benign/Alive prognosis. Routine monitoring suggested, but follow up with a healthcare provider."

# Step 9: Example prediction (using a test sample)
example_patient = X_test.iloc[0].values
recommendation = prognosis_recommendation(example_patient)

print("\n------------------------------------------")
print("🩺 Prognosis Recommendation for Example Patient:")
print(recommendation)
print("------------------------------------------")
