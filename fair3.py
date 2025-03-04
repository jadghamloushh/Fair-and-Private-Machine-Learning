import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
import numpy as np
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference


def load_data(file_path):
    data = pd.read_csv(file_path)
    data['person_gender'] = data['person_gender'].map({'male': 1, 'female': 0})
    y = data['loan_status'].values
    sensitive = data['person_gender'].values
    X = data.drop(columns=['loan_status'])
    return X, y, sensitive

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

# DP parameters
noise_multiplier = 2.0
l2_norm_clip = 1.0
batch_size = 250
epochs_per_iteration = 60  
microbatches = 25
learning_rate = 0.15


class DPNeuralNetwork:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = sample_weight / np.max(sample_weight)
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.input_shape,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

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
                epochs=epochs_per_iteration,
                batch_size=batch_size,
                verbose=0
            )
        else:
            self.history = self.model.fit(
                X, y,
                epochs=epochs_per_iteration,
                batch_size=batch_size,
                verbose=0
            )
        
        return self
    
    def predict(self, X):
        return (self.model.predict(X).flatten() >= 0.5).astype(int)

estimator = DPNeuralNetwork(input_shape=X_train_preprocessed.shape[1])

constraint = EqualizedOdds()  

print("Initializing Exponentiated Gradient Reduction...")
egr = ExponentiatedGradient(
    estimator=estimator,
    constraints=constraint,
    eps=0.01,  
    max_iter=50, 
    nu=1e-6   
)

print("Training the DP-EGR model...")
egr.fit(
    X_train_preprocessed,
    y_train,
    sensitive_features=sensitive_train
)

y_pred = egr.predict(X_test_preprocessed)
num_train_samples = len(X_train)
delta = 1e-5
total_epochs = egr.max_iter * epochs_per_iteration  

epsilon = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
    n=num_train_samples,
    batch_size=batch_size,
    noise_multiplier=noise_multiplier,
    epochs=total_epochs,
    delta=delta
)[0]

print(f"\nPrivacy Budget:")
print(f"ε = {epsilon:.2f} (with δ = {delta})")


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

   
    male_mask = sensitive_features == 1
    female_mask = sensitive_features == 0

    male_accuracy = np.mean(y_true[male_mask] == y_pred[male_mask])
    female_accuracy = np.mean(y_true[female_mask] == y_pred[female_mask])

    print("\nBase Model Metrics:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Male Accuracy: {male_accuracy:.4f}")
    print(f"Female Accuracy: {female_accuracy:.4f}")
    print(f"Accuracy Gap: {abs(male_accuracy - female_accuracy):.4f}")
    

    male_true_positives = np.sum((y_true[male_mask] == 1) & (y_pred[male_mask] == 1))
    male_total_positives = np.sum(y_true[male_mask] == 1)
    male_tpr = male_true_positives / male_total_positives if male_total_positives > 0 else 0

    female_true_positives = np.sum((y_true[female_mask] == 1) & (y_pred[female_mask] == 1))
    female_total_positives = np.sum(y_true[female_mask] == 1)
    female_tpr = female_true_positives / female_total_positives if female_total_positives > 0 else 0

    print("\nTrue Positive Rates (TPR):")
    print(f"Male TPR: {male_tpr:.4f}")
    print(f"Female TPR: {female_tpr:.4f}")
    print(f"TPR Gap: {abs(male_tpr - female_tpr):.4f}")
