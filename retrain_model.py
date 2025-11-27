import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Retraining the body language detection model...")
print("="*60)

# Load the training data
print("Loading training data from coords.csv...")
df = pd.read_csv('coords.csv')
print(f"Loaded {len(df)} samples")

# Prepare features and labels
X = df.drop('class', axis=1)  # features
y = df['class']  # target value

print(f"Classes: {', '.join(sorted(y.unique()))}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create and train the model (Random Forest - best performer from original)
print("\nTraining Random Forest model...")
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
model = pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2%}")

# Save the model
output_file = 'body_language_v2.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(model, f)

print(f"\n✓ Model saved as '{output_file}'")
print("="*60)
print("Model is now compatible with your scikit-learn version!")
