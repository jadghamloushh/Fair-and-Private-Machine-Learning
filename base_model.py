import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

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

# Build the base model
print("\nTraining Base Model...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_preprocessed.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile with same loss type as DP version
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
model.compile(
    optimizer='adam',
    loss=loss,
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train_preprocessed, y_train,
    validation_data=(X_val_preprocessed, y_val),
    epochs=60,
    batch_size=250,
    verbose=1
)

# Get predictions (using same prediction process as DP version)
y_test_pred_probs = model.predict(X_test_preprocessed).flatten()
y_test_pred = (y_test_pred_probs >= 0.5).astype(int)

# Evaluate using same metrics as DP version
def evaluate_model(y_true, y_pred, sensitive_features):
    accuracy = np.mean(y_true == y_pred)
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
    
    print("\nBase Model Metrics:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Demographic Parity Difference: {dp_diff:.4f}")
    print(f"Equalized Odds Difference: {eo_diff:.4f}")
    print(f"Male Accuracy: {male_accuracy:.4f}")
    print(f"Female Accuracy: {female_accuracy:.4f}")
    print(f"Accuracy Gap: {abs(male_accuracy - female_accuracy):.4f}")

    # Add prediction distribution analysis
    print("\nPrediction Analysis:")
    print(f"Mean prediction probability: {np.mean(y_test_pred_probs):.4f}")
    print(f"Prediction distribution: {np.bincount(y_pred)}")

# Evaluate the model
evaluate_model(y_test, y_test_pred, sensitive_test)