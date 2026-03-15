# scripts/04_train_classifier.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
from datetime import datetime

# Create results directory
Path("results").mkdir(exist_ok=True)

# Load data
print("Loading data...")
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

print(f"Train: {len(train)}")
print(f"Test: {len(test)}")

# Vectorize text
print("\nVectorizing text...")
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,3))
X_train = vectorizer.fit_transform(train['text'])
X_test = vectorizer.transform(test['text'])
y_train = train['toxic']
y_test = test['toxic']

print(f"X_train shape: {X_train.shape}")

# Train model
print("\nTraining Logistic Regression...")
model = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Create results dataframe
results_df = test.copy()
results_df['predicted'] = y_pred
results_df['probability'] = y_proba
results_df['correct'] = (results_df['toxic'] == results_df['predicted'])

# Save predictions
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_df.to_csv(f"results/predictions_{timestamp}.csv", index=False)
print(f"\n✅ Saved predictions to results/predictions_{timestamp}.csv")

# Save misclassified examples
misclassified = results_df[results_df['correct'] == False]
misclassified.to_csv(f"results/misclassified_{timestamp}.csv", index=False)
print(f"✅ Saved misclassified to results/misclassified_{timestamp}.csv")

# Save probability distribution
proba_df = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': y_pred,
    'probability_toxic': y_proba
})
proba_df.to_csv(f"results/probabilities_{timestamp}.csv", index=False)
print(f"✅ Saved probabilities to results/probabilities_{timestamp}.csv")

# Evaluate
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
report = classification_report(y_test, y_pred, target_names=['Non-Toxic', 'Toxic'], output_dict=True)
print(classification_report(y_test, y_pred, target_names=['Non-Toxic', 'Toxic']))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save metrics
metrics_df = pd.DataFrame({
    'metric': ['accuracy', 'precision_non_toxic', 'recall_non_toxic', 'f1_non_toxic', 
               'precision_toxic', 'recall_toxic', 'f1_toxic'],
    'value': [
        accuracy_score(y_test, y_pred),
        report['Non-Toxic']['precision'],
        report['Non-Toxic']['recall'],
        report['Non-Toxic']['f1-score'],
        report['Toxic']['precision'],
        report['Toxic']['recall'],
        report['Toxic']['f1-score']
    ]
})
metrics_df.to_csv(f"results/metrics_{timestamp}.csv", index=False)
print(f"✅ Saved metrics to results/metrics_{timestamp}.csv")

# Save model info
model_info = pd.DataFrame({
    'model': ['LogisticRegression'],
    'features': [X_train.shape[1]],
    'train_samples': [len(train)],
    'test_samples': [len(test)],
    'accuracy': [accuracy_score(y_test, y_pred)],
    'timestamp': [timestamp]
})
model_info.to_csv(f"results/model_info_{timestamp}.csv", index=False)

# Save model
print("\nSaving model...")
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/toxic_classifier.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("✅ Saved model to models/")

print(f"\n📁 All results saved to results/ folder")