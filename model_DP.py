import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
import numpy as np
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import confusion_matrix, classification_report
import random



random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)  # TF 2.7+ comprehensive seeding
tf.config.experimental.enable_op_determinism()

# Data loading function
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['person_gender'] = data['person_gender'].map({'male': 1, 'female': 0})
    y = data['loan_status'].values
    sensitive = data['person_gender'].values
    X = data.drop(columns=['loan_status', 'person_gender'])
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

# Define DP para#meters
noise_multiplier = 2.0
l2_norm_clip = 1.0
batch_size = 250
epochs = 60
microbatches = 25
learning_rate = 0.1

# Build the DP model
print("\nTraining DP Model...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_preprocessed.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Configure DP optimizer
optimizer = DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=microbatches,
    learning_rate=learning_rate
)

# Compile model
loss = tf.keras.losses.BinaryCrossentropy(
    from_logits=False,
    reduction=tf.keras.losses.Reduction.NONE
)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_preprocessed, y_train,
    validation_data=(X_val_preprocessed, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# Calculate privacy budget
num_train_samples = len(X_train)
delta = 1e-5

epsilon = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
    n=num_train_samples,
    batch_size=batch_size,
    noise_multiplier=noise_multiplier,
    epochs=epochs,
    delta=delta
)[0]

print(f"\nPrivacy Budget:")
print(f"ε = {epsilon:.2f} (with δ = {delta})")

# Get predictions for training, validation, and test sets
y_train_pred_probs = model.predict(X_train_preprocessed).flatten()
y_train_pred = (y_train_pred_probs >= 0.5).astype(int)
y_val_pred_probs = model.predict(X_val_preprocessed).flatten()
y_val_pred = (y_val_pred_probs >= 0.5).astype(int)
y_test_pred_probs = model.predict(X_test_preprocessed).flatten()
y_test_pred = (y_test_pred_probs >= 0.5).astype(int)

# Calculate train and test accuracy
train_accuracy = np.mean(y_train == y_train_pred)
val_accuracy = np.mean(y_val == y_val_pred)
test_accuracy = np.mean(y_test == y_test_pred)

print(f"\nModel Accuracy:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Function to calculate TPR (Recall for positive class)
def calculate_tpr(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]  # True Positives
    fn = cm[1, 0]  # False Negatives
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    return tpr

# Evaluate model
def evaluate_model(y_true, y_pred, sensitive_features):
    dp_diff = demographic_parity_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    eo_diff = equalized_odds_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    # Calculate group-specific metrics
    male_mask = sensitive_features == 1
    female_mask = sensitive_features == 0
    
    male_accuracy = np.mean(y_true[male_mask] == y_pred[male_mask])
    female_accuracy = np.mean(y_true[female_mask] == y_pred[female_mask])
    
    # Calculate TPR for men and women
    male_tpr = calculate_tpr(y_true[male_mask], y_pred[male_mask])
    female_tpr = calculate_tpr(y_true[female_mask], y_pred[female_mask])
    tpr_diff = abs(male_tpr - female_tpr)
    
    print("\nDP Model Metrics:")
    print(f"Demographic Parity Difference: {dp_diff:.4f}")
    print(f"Equalized Odds Difference: {eo_diff:.4f}")
    print(f"Male Accuracy: {male_accuracy:.4f}")
    print(f"Female Accuracy: {female_accuracy:.4f}")
    print(f"Accuracy Gap: {abs(male_accuracy - female_accuracy):.4f}")
    
    print("\nTrue Positive Rate (TPR) Analysis:")
    print(f"Male TPR: {male_tpr:.4f}")
    print(f"Female TPR: {female_tpr:.4f}")
    print(f"TPR Difference (gender bias measure): {tpr_diff:.4f}")

    # Add prediction distribution analysis
    print("\nPrediction Analysis:")
    print(f"Mean prediction probability: {np.mean(y_test_pred_probs):.4f}")
    print(f"Prediction distribution: {np.bincount(y_pred)}")
    
    # Print classification reports
    print("\nClassification Report for Men:")
    print(classification_report(y_true[male_mask], y_pred[male_mask]))
    
    print("\nClassification Report for Women:")
    print(classification_report(y_true[female_mask], y_pred[female_mask]))
    
    # Plot TPR comparison
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Men', 'Women'], [male_tpr, female_tpr], color=['blue', 'red'])
    plt.title('DP Model: True Positive Rate Comparison by Gender')
    plt.ylabel('True Positive Rate')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.show()
    
    return male_tpr, female_tpr

# Evaluate the model and get TPR values
male_tpr, female_tpr = evaluate_model(y_test, y_test_pred, sensitive_test)

# Plot training and validation metrics (from history)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('DP Model Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('DP Model Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.ylim(0.5, 1.0)
plt.show()

# Plot final accuracy comparison between train, validation, and test sets
plt.figure(figsize=(10, 6))
accuracy_bars = plt.bar(['Training', 'Validation', 'Testing'], 
                        [train_accuracy, val_accuracy, test_accuracy],
                        color=['blue', 'green', 'orange'])
plt.title('DP Model: Final Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add labels on top of bars
for bar in accuracy_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

plt.show()