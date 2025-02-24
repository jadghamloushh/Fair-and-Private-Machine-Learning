import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
import numpy as np
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# Keep existing data loading and preprocessing code
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['person_gender'] = data['person_gender'].map({'male': 1, 'female': 0})
    y = data['loan_status'].values
    sensitive = data['person_gender'].values
    X = data.drop(columns=['loan_status', 'person_gender'])
    return X, y, sensitive

# Load and split data (keep existing code)
X_train_val, y_train_val, sensitive_train_val = load_data('train.csv')
X_test, y_test, sensitive_test = load_data('test.csv')

X_train, X_val, y_train, y_val, sensitive_train, sensitive_val = train_test_split(
    X_train_val, y_train_val, sensitive_train_val, test_size=0.2, random_state=42
)

# Keep existing preprocessing pipeline
categorical_cols = ['person_education', 'loan_intent', 'person_home_ownership', 'previous_loan_defaults_on_file']
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(sparse_output=False), categorical_cols)
    ])

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)
X_test_preprocessed = preprocessor.transform(X_test)

# Define DP parameters (keep existing)
noise_multiplier = 2.0
l2_norm_clip = 1.0
batch_size = 250
epochs = 60
microbatches = 25
learning_rate = 0.15

# Create a class for the DP-enabled base estimator
class DPNeuralNetwork:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def fit(self, X, y, sample_weight=None):
        # Build the model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.input_shape,)),
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
        
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE
        )
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
        # Apply sample weights if provided
        if sample_weight is not None:
            self.history = self.model.fit(
                X, y,
                sample_weight=sample_weight,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
        else:
            self.history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
        
        return self
    
    def predict(self, X):
        return (self.model.predict(X).flatten() >= 0.5).astype(int)

# Initialize base estimator
estimator = DPNeuralNetwork(input_shape=X_train_preprocessed.shape[1])

# Define fairness constraints for EGR
constraint = EqualizedOdds()  # Can also use DemographicParity()

# Initialize EGR
print("Initializing Exponentiated Gradient Reduction...")
egr = ExponentiatedGradient(
    estimator=estimator,
    constraints=constraint,
    eps=0.001,  # Maximum allowed fairness violation
    max_iter=50,  # Maximum number of iterations
    nu=1e-6    # Convergence threshold
)

# Fit the EGR model
print("Training the DP-EGR model...")
egr.fit(
    X_train_preprocessed,
    y_train,
    sensitive_features=sensitive_train
)

# Get predictions
y_pred = egr.predict(X_test_preprocessed)

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

# Evaluate model performance
def evaluate_fairness(y_true, y_pred, sensitive_features):
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
    
    return {
        'accuracy': accuracy,
        'demographic_parity_diff': dp_diff,
        'equalized_odds_diff': eo_diff,
        'male_accuracy': male_accuracy,
        'female_accuracy': female_accuracy,
        'accuracy_gap': abs(male_accuracy - female_accuracy)
    }

# Print results
print("\nDP-EGR Model Metrics:")
metrics = evaluate_fairness(y_test, y_pred, sensitive_test)
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")