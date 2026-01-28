import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==========================================
# STEP 1: Generate Synthetic Training Data
# ==========================================
# We create 500 fake patient records to train the model
np.random.seed(42)
n_samples = 500

# Generate random features
glucose = np.random.randint(70, 300, n_samples)  # Glucose levels
bmi = np.random.uniform(15, 40, n_samples)       # BMI
age = np.random.randint(20, 80, n_samples)       # Age

# Create a DataFrame
df = pd.DataFrame({
    'Glucose': glucose,
    'BMI': bmi,
    'Age': age
})

# Define simple logic to assign "Correct" labels (Ground Truth)
# This mimics what a doctor would say, so the model has something to learn.
conditions = []
diets = []

for i in range(n_samples):
    g = glucose[i]
    b = bmi[i]
    
    if g > 126:
        conditions.append("Diabetes")
        diets.append("Diabetic Friendly (Low Sugar)")
    elif b > 30:
        conditions.append("Obesity")
        diets.append("Low Carb / Keto")
    elif g > 100 and b > 25:
        conditions.append("Pre-Diabetic")
        diets.append("Mediterranean Diet")
    else:
        conditions.append("Healthy")
        diets.append("Balanced Diet")

df['Condition'] = conditions
df['Diet_Plan'] = diets

# Save this training data so you can show you have a "Dataset"
df.to_csv("training_dataset.csv", index=False)
print("✅ Synthetic training data generated and saved to 'training_dataset.csv'")

# ==========================================
# STEP 2: Train the Model (Milestone 2)
# ==========================================

# Features (X) and Target (y)
X = df[['Glucose', 'BMI', 'Age']]
y = df['Diet_Plan']

# Split data: 80% training, 20% testing [cite: 74]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier 
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
print("🔄 Training Model...")
model.fit(X_train, y_train)

# ==========================================
# STEP 3: Evaluation (Milestone 2 Metrics)
# ==========================================
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n📊 Model Evaluation Metrics:")
print(f"Accuracy Score: {accuracy * 100:.2f}%") # [cite: 57]
print("Classification Report:")
print(classification_report(y_test, predictions)) # [cite: 59, 61]

# ==========================================
# STEP 4: Save the Model
# ==========================================
joblib.dump(model, "diet_model.pkl")
print("💾 Model saved as 'diet_model.pkl'")