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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix


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
noise_multiplier = 3.0
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
   
    def predict_proba(self, X):
        return self.model.predict(X).flatten()

estimator = DPNeuralNetwork(input_shape=X_train_preprocessed.shape[1])

constraint = EqualizedOdds()  

print("Initializing Exponentiated Gradient Reduction...")
egr = ExponentiatedGradient(
    estimator=estimator,
    constraints=constraint,
    eps=0.01,  
    max_iter=8,
    nu=1e-6  
)

print("Training the DP-EGR model...")
egr.fit(
    X_train_preprocessed,
    y_train,
    sensitive_features=sensitive_train
)

# Calculate training accuracy
y_train_pred = egr.predict(X_train_preprocessed)
train_accuracy = np.mean(y_train == y_train_pred)
print(f"\nTraining Accuracy: {train_accuracy:.4f}")

# Calculate testing accuracy
y_pred = egr.predict(X_test_preprocessed)
test_accuracy = np.mean(y_test == y_pred)
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Get probability predictions for ROC curves and TPR analysis
def get_egr_probas(egr_model, X):
    # For EGR, we need to combine the predictions from all estimators
    if hasattr(egr_model, '_predictors'):
        # Extract the base estimators and their weights
        predictors = egr_model._predictors
        weights = egr_model._weights
       
        # Get probability predictions from each estimator
        probas = []
        for predictor in predictors:
            if hasattr(predictor, 'predict_proba'):
                probas.append(predictor.predict_proba(X))
            else:
                # Use the underlying model's prediction
                proba = predictor.model.predict(X).flatten()
                probas.append(proba)
               
        # Combine the predictions using the weights
        combined_probas = np.zeros(len(X))
        for i, weight in enumerate(weights):
            combined_probas += weight * probas[i]
       
        return combined_probas
    else:
        # If it's not an EGR model or prediction details aren't available
        try:
            return egr_model.predict_proba(X)
        except:
            # Fallback to binary predictions
            return egr_model.predict(X).astype(float)

# Try to get probability predictions
try:
    y_proba = get_egr_probas(egr, X_test_preprocessed)
except Exception as e:
    print(f"Could not get probability predictions: {e}")
    # Fallback to binary predictions
    y_proba = y_pred.astype(float)

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


def evaluate_model(y_true, y_pred, sensitive_features, y_proba=None):
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

    # Group-specific metrics
    male_mask = sensitive_features == 1
    female_mask = sensitive_features == 0

    male_accuracy = np.mean(y_true[male_mask] == y_pred[male_mask])
    female_accuracy = np.mean(y_true[female_mask] == y_pred[female_mask])

    # Calculate confusion matrix metrics
    male_cm = confusion_matrix(y_true[male_mask], y_pred[male_mask])
    female_cm = confusion_matrix(y_true[female_mask], y_pred[female_mask])
   
    # True Positive Rate (Recall)
    male_tpr = male_cm[1, 1] / (male_cm[1, 0] + male_cm[1, 1]) if (male_cm[1, 0] + male_cm[1, 1]) > 0 else 0
    female_tpr = female_cm[1, 1] / (female_cm[1, 0] + female_cm[1, 1]) if (female_cm[1, 0] + female_cm[1, 1]) > 0 else 0
   
    # True Negative Rate (Specificity)
    male_tnr = male_cm[0, 0] / (male_cm[0, 0] + male_cm[0, 1]) if (male_cm[0, 0] + male_cm[0, 1]) > 0 else 0
    female_tnr = female_cm[0, 0] / (female_cm[0, 0] + female_cm[0, 1]) if (female_cm[0, 0] + female_cm[0, 1]) > 0 else 0
   
    # False Positive Rate
    male_fpr = 1 - male_tnr
    female_fpr = 1 - female_tnr
   
    # Print basic metrics
    print("\nModel Metrics:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Demographic Parity Difference: {dp_diff:.4f}")
    print(f"Equalized Odds Difference: {eo_diff:.4f}")
    print(f"Male Accuracy: {male_accuracy:.4f}")
    print(f"Female Accuracy: {female_accuracy:.4f}")
    print(f"Accuracy Gap: {abs(male_accuracy - female_accuracy):.4f}")
   
    print("\nTrue Positive Rates (TPR):")
    print(f"Male TPR: {male_tpr:.4f}")
    print(f"Female TPR: {female_tpr:.4f}")
    print(f"TPR Gap: {abs(male_tpr - female_tpr):.4f}")
   
    print("\nTrue Negative Rates (TNR):")
    print(f"Male TNR: {male_tnr:.4f}")
    print(f"Female TNR: {female_tnr:.4f}")
    print(f"TNR Gap: {abs(male_tnr - female_tnr):.4f}")
   
    # Return metrics dictionary for use in plots
    metrics_dict = {
        'accuracy': accuracy,
        'demographic_parity_diff': dp_diff,
        'equalized_odds_diff': eo_diff,
        'male_accuracy': male_accuracy,
        'female_accuracy': female_accuracy,
        'male_tpr': male_tpr,
        'female_tpr': female_tpr,
        'male_tnr': male_tnr,
        'female_tnr': female_tnr,
        'male_fpr': male_fpr,
        'female_fpr': female_fpr
    }
   
    return metrics_dict

# Plot ROC curves for each demographic group
def plot_group_roc_curves(y_true, y_score, sensitive_features, title="ROC Curves by Group"):
    male_mask = sensitive_features == 1
    female_mask = sensitive_features == 0
   
    # Calculate ROC curve for males
    fpr_male, tpr_male, _ = roc_curve(y_true[male_mask], y_score[male_mask])
    roc_auc_male = auc(fpr_male, tpr_male)
   
    # Calculate ROC curve for females
    fpr_female, tpr_female, _ = roc_curve(y_true[female_mask], y_score[female_mask])
    roc_auc_female = auc(fpr_female, tpr_female)
   
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_male, tpr_male, color='blue', lw=2, label=f'Male (AUC = {roc_auc_male:.2f})')
    plt.plot(fpr_female, tpr_female, color='red', lw=2, label=f'Female (AUC = {roc_auc_female:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('group_roc_curves.png')
    plt.close()
   
    return {'male_auc': roc_auc_male, 'female_auc': roc_auc_female}

# Plot TPR at different thresholds
def plot_tpr_at_thresholds(y_true, y_score, sensitive_features, thresholds=None, title="TPR by Group at Different Thresholds"):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)
       
    male_mask = sensitive_features == 1
    female_mask = sensitive_features == 0
   
    male_tpr_values = []
    female_tpr_values = []
   
    for threshold in thresholds:
        # Get binary predictions at this threshold
        y_pred_threshold = (y_score >= threshold).astype(int)
       
        # Calculate TPR for males at this threshold
        male_tp = np.sum((y_pred_threshold[male_mask] == 1) & (y_true[male_mask] == 1))
        male_p = np.sum(y_true[male_mask] == 1)
        male_tpr_values.append(male_tp / male_p if male_p > 0 else 0)
       
        # Calculate TPR for females at this threshold
        female_tp = np.sum((y_pred_threshold[female_mask] == 1) & (y_true[female_mask] == 1))
        female_p = np.sum(y_true[female_mask] == 1)
        female_tpr_values.append(female_tp / female_p if female_p > 0 else 0)
   
    # Plot TPR values
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, male_tpr_values, 'b-', lw=2, label='Male TPR')
    plt.plot(thresholds, female_tpr_values, 'r-', lw=2, label='Female TPR')
   
    # Indicate default threshold (0.5)
    default_idx = np.abs(thresholds - 0.5).argmin()
    plt.axvline(x=0.5, color='gray', linestyle='--', lw=1)
    plt.plot(0.5, male_tpr_values[default_idx], 'bo', markersize=8)
    plt.plot(0.5, female_tpr_values[default_idx], 'ro', markersize=8)
   
    # Calculate and display TPR gap at default threshold
    tpr_gap = abs(male_tpr_values[default_idx] - female_tpr_values[default_idx])
    plt.annotate(f'TPR Gap at 0.5 = {tpr_gap:.4f}',
                 xy=(0.5, (male_tpr_values[default_idx] + female_tpr_values[default_idx])/2),
                 xytext=(0.6, (male_tpr_values[default_idx] + female_tpr_values[default_idx])/2),
                 arrowprops=dict(facecolor='black', shrink=0.05))
   
    plt.grid(True)
    plt.xlabel('Classification Threshold')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig('tpr_vs_threshold.png')
    plt.close()
   
    return male_tpr_values, female_tpr_values, thresholds

# Plot TPR parity gap at different thresholds
def plot_tpr_gap(male_tpr, female_tpr, thresholds, title="TPR Parity Gap at Different Thresholds"):
    tpr_gap = np.abs(np.array(male_tpr) - np.array(female_tpr))
   
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, tpr_gap, 'g-', lw=2)
    plt.axvline(x=0.5, color='gray', linestyle='--', lw=1)
   
    # Find minimum gap
    min_gap_idx = np.argmin(tpr_gap)
    min_gap_threshold = thresholds[min_gap_idx]
    min_gap = tpr_gap[min_gap_idx]
   
    plt.plot(min_gap_threshold, min_gap, 'ro', markersize=8)
    plt.annotate(f'Min Gap = {min_gap:.4f} at threshold = {min_gap_threshold:.2f}',
                 xy=(min_gap_threshold, min_gap),
                 xytext=(min_gap_threshold + 0.1, min_gap + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))
   
    plt.grid(True)
    plt.xlabel('Classification Threshold')
    plt.ylabel('|TPR_male - TPR_female|')
    plt.title(title)
    plt.savefig('tpr_gap_vs_threshold.png')
    plt.close()
   
    return min_gap, min_gap_threshold

# Plot fairness metrics comparison
def plot_fairness_metrics_comparison(metrics):
    metrics_to_plot = [
        ('TPR', metrics['male_tpr'], metrics['female_tpr']),
        ('TNR', metrics['male_tnr'], metrics['female_tnr']),
        ('Accuracy', metrics['male_accuracy'], metrics['female_accuracy'])
    ]
   
    labels = [m[0] for m in metrics_to_plot]
    male_values = [m[1] for m in metrics_to_plot]
    female_values = [m[2] for m in metrics_to_plot]
   
    x = np.arange(len(labels))
    width = 0.35
   
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, male_values, width, label='Male')
    bars2 = ax.bar(x + width/2, female_values, width, label='Female')
   
    # Add gap annotations
    for i, (m, f) in enumerate(zip(male_values, female_values)):
        gap = abs(m - f)
        ax.annotate(f'Gap: {gap:.4f}',
                    xy=(i, max(m, f) + 0.02),
                    ha='center')
   
    ax.set_ylabel('Value')
    ax.set_title('Fairness Metrics Comparison by Gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
   
    fig.tight_layout()
    plt.savefig('fairness_metrics_comparison.png')
    plt.close()

# Evaluate model and get metrics
metrics = evaluate_model(y_test, y_pred, sensitive_test, y_proba)

# Plot ROC curves if we have probability predictions
if len(np.unique(y_proba)) > 2:  # Only plot if we have more than binary values
    auc_metrics = plot_group_roc_curves(y_test, y_proba, sensitive_test,
                                        title="ROC Curves by Gender Group (DP-EGR Model)")
   
    print("\nAUC by Group:")
    print(f"Male AUC: {auc_metrics['male_auc']:.4f}")
    print(f"Female AUC: {auc_metrics['female_auc']:.4f}")
    print(f"AUC Gap: {abs(auc_metrics['male_auc'] - auc_metrics['female_auc']):.4f}")
   
    # Plot TPR at different thresholds
    thresholds = np.linspace(0.01, 0.99, 50)
    male_tpr, female_tpr, thresholds = plot_tpr_at_thresholds(
        y_test, y_proba, sensitive_test, thresholds,
        title="True Positive Rate by Gender at Different Thresholds"
    )
   
    # Plot TPR gap at different thresholds
    min_gap, min_gap_threshold = plot_tpr_gap(
        male_tpr, female_tpr, thresholds,
        title="TPR Parity Gap at Different Thresholds"
    )
   
    print(f"\nMinimum TPR gap of {min_gap:.4f} achieved at threshold = {min_gap_threshold:.2f}")
else:
    print("\nWarning: Only binary predictions are available. Cannot generate probability-based plots.")

# Plot fairness metrics comparison
plot_fairness_metrics_comparison(metrics)

print("\nPlots saved:")
print("1. group_roc_curves.png - ROC curves comparing model performance between gender groups")
print("2. tpr_vs_threshold.png - How TPR varies by gender across different classification thresholds")
print("3. tpr_gap_vs_threshold.png - The TPR gap between gender groups at different thresholds")
print("4. fairness_metrics_comparison.png - Bar chart comparing key fairness metrics by gender")