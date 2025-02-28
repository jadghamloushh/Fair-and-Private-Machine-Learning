import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
import numpy as np
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
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

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)
X_test_preprocessed = preprocessor.transform(X_test)

# Define DP parameters - adjust for better privacy-fairness tradeoff
noise_multiplier = 2.0  # Increase for better privacy
l2_norm_clip = 1.0
batch_size = 250
epochs = 20  # Reduced from 60 to lower privacy cost
microbatches = 25
learning_rate = 0.15

# Define max iterations for ExponentiatedGradient - reduced to lower privacy cost
max_iter = 5  # Reduced from 50 to lower overall privacy cost

# Create a class for the DP-enabled base estimator with proper sample weight handling
class DPNeuralNetwork:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def fit(self, X, y, sample_weight=None):
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # IMPORTANT: Normalize sample weights to ≤ 1 to prevent breaking DP guarantees
        if sample_weight is not None:
            # Ensure weights don't exceed 1 by normalizing
            max_weight = np.max(sample_weight)
            if max_weight > 1:
                sample_weight = sample_weight / max_weight
        
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
        X = np.asarray(X)
        return (self.model.predict(X, verbose=0).flatten() >= 0.5).astype(int)
    
    # Required for sklearn compatibility
    def get_params(self, deep=True):
        return {"input_shape": self.input_shape}
    
    # Required for sklearn compatibility
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# Initialize base estimator
estimator = DPNeuralNetwork(input_shape=X_train_preprocessed.shape[1])

# Define fairness constraints
constraint = EqualizedOdds()  # Can also use DemographicParity()

# Initialize EGR
print("Initializing Exponentiated Gradient Reduction...")
egr = ExponentiatedGradient(
    estimator=estimator,
    constraints=constraint,
    eps=0.01,  # Maximum allowed fairness violation
    max_iter=max_iter,  # Reduced number of iterations
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

# CORRECTED Privacy Budget Calculation
# Calculate total number of accesses to the dataset
num_train_samples = len(X_train)
delta = 1e-5

# Correct privacy budget using total epochs across all iterations
total_epochs = max_iter * epochs  # Total number of epochs across all iterations
print(f"\nCalculating privacy budget for {max_iter} iterations, each with {epochs} epochs = {total_epochs} total epochs")

# Calculate privacy budget - this is an approximation since DPSGD in each iteration
# is a separate training that should be composed
epsilon = compute_dp_sgd_privacy(
    n=num_train_samples,
    batch_size=batch_size,
    noise_multiplier=noise_multiplier,
    epochs=total_epochs,  # Accounting for all iterations
    delta=delta
)[0]

print(f"\nPrivacy Budget:")
print(f"ε = {epsilon:.2f} (with δ = {delta})")
print("Note: This is an approximation based on total training epochs. The actual DP guarantee may be different")
print("      due to the composition of multiple separate training runs in the EGR algorithm.")

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

# Print fairness-privacy tradeoff information
print("\nFairness-Privacy Tradeoff Analysis:")
print(f"- Privacy parameters: noise={noise_multiplier}, l2_clip={l2_norm_clip}, ε={epsilon:.2f}")
print(f"- Fairness constraint: {constraint.__class__.__name__}, allowed violation={egr.eps}")
print(f"- EGR iterations: {max_iter}, epochs per iteration: {epochs}")
print("- Privacy cost increases with more iterations and epochs")
print("- Fairness improves with more iterations but at higher privacy cost")

# Evaluate fairness on validation set
val_pred = egr.predict(X_val_preprocessed)
val_metrics = evaluate_fairness(y_val, val_pred, sensitive_val)
print("\nValidation Set Metrics:")
for metric, value in val_metrics.items():
    print(f"{metric}: {value:.4f}")