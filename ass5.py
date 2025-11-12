from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Data
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)

# Model
m = RandomForestClassifier(100).fit(Xtr, ytr)
yp = m.predict(Xte)

# Results
print("Accuracy:", accuracy_score(yte, yp))
print("\nReport:\n", classification_report(yte, yp))
print("Confusion Matrix:\n", confusion_matrix(yte, yp))

# Recommendation
def advise(f):
    return (" Malignant! Consult doctor." if m.predict([f])[0]==0
            else " Benign. Routine checkup.")

print("\nRecommendation:", advise(Xte[0]))
