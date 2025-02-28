import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report

# Data loading function
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['person_gender'] = data['person_gender'].map({'male': 1, 'female': 0})
    y = data['loan_status'].values
    sensitive = data['person_gender'].values
    X = data.drop(columns=['loan_status'])
    return X, y, sensitive

# Load and split data
X_train_val, y_train_val, sensitive_train_val = load_data('train.csv')
X_test, y_test, sensitive_test = load_data('test.csv')

X_train, X_val, y_train, y_val, sensitive_train, sensitive_val = train_test_split(
    X_train_val, y_train_val, sensitive_train_val, test_size=0.2, random_state=42
)

# Preprocessing pipeline
categorical_cols = ['person_education', 'loan_intent', 'person_home_ownership', 'previous_loan_defaults_on_file']
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(sparse_output=False), categorical_cols)
    ])

# Preprocess the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)
X_test_preprocessed = preprocessor.transform(X_test)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_preprocessed.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model - set verbose=0 to suppress epoch output
history = model.fit(
    X_train_preprocessed, y_train,
    validation_data=(X_val_preprocessed, y_val),
    epochs=60,
    batch_size=250,
    verbose=0
)

# Get training accuracy (from the last epoch)
train_accuracy = history.history['accuracy'][-1]
print(f"Training Accuracy: {train_accuracy:.4f}")

# Get testing accuracy
test_loss, test_accuracy = model.evaluate(X_test_preprocessed, y_test, verbose=0)
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Get predictions on test set
y_test_pred_probs = model.predict(X_test_preprocessed, verbose=0).flatten()
y_test_pred = (y_test_pred_probs >= 0.5).astype(int)

# Calculate TPR for men and women separately
male_indices = np.where(sensitive_test == 1)[0]
female_indices = np.where(sensitive_test == 0)[0]

# Function to calculate TPR (Recall for positive class)
def calculate_tpr(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]  # True Positives
    fn = cm[1, 0]  # False Negatives
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    return tpr

# Calculate TPR for men
male_tpr = calculate_tpr(y_test[male_indices], y_test_pred[male_indices])
print(f"Male True Positive Rate: {male_tpr:.4f}")

# Calculate TPR for women
female_tpr = calculate_tpr(y_test[female_indices], y_test_pred[female_indices])
print(f"Female True Positive Rate: {female_tpr:.4f}")

# Calculate TPR difference (bias measure)
tpr_diff = abs(male_tpr - female_tpr)
print(f"TPR Difference (gender bias measure): {tpr_diff:.4f}")

# Print full classification reports by gender
print("\nClassification Report for Men:")
print(classification_report(y_test[male_indices], y_test_pred[male_indices]))

print("\nClassification Report for Women:")
print(classification_report(y_test[female_indices], y_test_pred[female_indices]))

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.ylim(0.5, 1.0)  # Set y-axis to start from 0.5 for better visualization
plt.show()

# Plot TPR comparison between men and women
plt.figure(figsize=(8, 6))
bars = plt.bar(['Men', 'Women'], [male_tpr, female_tpr], color=['blue', 'red'])
plt.title('True Positive Rate Comparison by Gender')
plt.ylabel('True Positive Rate')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

plt.show()