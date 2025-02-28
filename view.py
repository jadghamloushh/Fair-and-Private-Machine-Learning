import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

# TensorFlow and Keras imports for Adam optimizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load data
print("Loading data...")
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Display basic information about the datasets
print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Check for missing values
print("\nMissing values in training data:")
print(train_data.isnull().sum())

# Data exploration
print("\nSummary statistics of training data:")
print(train_data.describe())

# Check target variable distribution
print("\nTarget variable distribution in training data:")
print(train_data['loan_status'].value_counts(normalize=True))

# Separate features and target
X_train = train_data.drop('loan_status', axis=1)
y_train = train_data['loan_status']
X_test = test_data.drop('loan_status', axis=1) if 'loan_status' in test_data.columns else test_data
y_test = test_data['loan_status'] if 'loan_status' in test_data.columns else None

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Convert 'Yes'/'No' to 1/0 for previous_loan_defaults_on_file if it's not already numeric
if 'previous_loan_defaults_on_file' in X_train.columns and X_train['previous_loan_defaults_on_file'].dtype == 'object':
    X_train['previous_loan_defaults_on_file'] = X_train['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
    if X_test is not None:
        X_test['previous_loan_defaults_on_file'] = X_test['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
    if 'previous_loan_defaults_on_file' in numerical_cols:
        pass
    else:
        numerical_cols.append('previous_loan_defaults_on_file')
    if 'previous_loan_defaults_on_file' in categorical_cols:
        categorical_cols.remove('previous_loan_defaults_on_file')

print(f"\nNumerical columns: {numerical_cols}")
print(f"Categorical columns: {categorical_cols}")

# Create preprocessor
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Preprocess the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Get the number of features after preprocessing
n_features = X_train_preprocessed.shape[1]
print(f"\nNumber of features after preprocessing: {n_features}")

# Calculate class weights for imbalanced data
total = len(y_train)
n_pos = sum(y_train)
n_neg = total - n_pos
weight_for_0 = (1 / n_neg) * (total / 2.0)
weight_for_1 = (1 / n_pos) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}
print(f"\nClass weights: {class_weight}")

# Build TensorFlow model with Adam optimizer
def build_model(input_dim, learning_rate=0.001):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

# Define callbacks for training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_loan_model.h5',
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    verbose=1
)

# Train final model with best parameters
print("\nTraining final model...")
final_model = build_model(input_dim=n_features, learning_rate=0.001)

final_history = final_model.fit(
    X_train_preprocessed, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint],
    class_weight=class_weight,
    verbose=1
)

# Load the best model (saved by ModelCheckpoint)
final_model = tf.keras.models.load_model('best_loan_model.h5')

# Evaluate on test data
print("\nEvaluating on test data...")
test_results = final_model.evaluate(X_test_preprocessed, y_test, verbose=0)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
print(f"Test AUC: {test_results[2]:.4f}")

# Make predictions
y_pred_prob = final_model.predict(X_test_preprocessed)
y_pred = (y_pred_prob > 0.5).astype(int)

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('tf_confusion_matrix.png')
plt.close()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(final_history.history['loss'], label='Training Loss')
plt.plot(final_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(final_history.history['accuracy'], label='Training Accuracy')
plt.plot(final_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('tf_training_history.png')
plt.close()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('tf_roc_curve.png')
plt.close()

# Save predictions
predictions_df = test_data.copy()
predictions_df['predicted_probability'] = y_pred_prob
predictions_df['predicted_loan_status'] = y_pred

predictions_df.to_csv('tensorflow_predictions.csv', index=False)
print("\nPredictions saved to 'tensorflow_predictions.csv'")

# Manual implementation of feature importance for TensorFlow models
def calculate_feature_importance(model, X, y, feature_names, n_repeats=10):
    """
    Calculate permutation feature importance for a TensorFlow model
    
    Parameters:
    model: Trained TensorFlow model
    X: Feature matrix
    y: Target values
    feature_names: List of feature names
    n_repeats: Number of times to repeat the permutation
    
    Returns:
    importance_results: Dictionary with feature importances
    """
    # Base score without permutation
    y_pred = model.predict(X).flatten()
    baseline_score = roc_auc_score(y, y_pred)
    print(f"Baseline AUC score: {baseline_score:.4f}")
    
    # Calculate importance for each feature
    importance_scores = []
    importance_stds = []
    
    for i in range(X.shape[1]):
        feature_scores = []
        
        for j in range(n_repeats):
            # Create a copy of the data
            X_permuted = X.copy()
            
            # Permute the feature
            perm_idx = np.random.permutation(X.shape[0])
            X_permuted[:, i] = X_permuted[perm_idx, i]
            
            # Score with permuted feature
            perm_pred = model.predict(X_permuted).flatten()
            perm_score = roc_auc_score(y, perm_pred)
            
            # Importance is the drop in performance
            feature_scores.append(baseline_score - perm_score)
        
        # Calculate mean and std of importance
        importance_scores.append(np.mean(feature_scores))
        importance_stds.append(np.std(feature_scores))
    
    # Create results dictionary
    importance_results = {
        'feature_names': feature_names[:X.shape[1]],
        'importances_mean': np.array(importance_scores),
        'importances_std': np.array(importance_stds)
    }
    
    return importance_results

# Get feature names
feature_names = []
# Get numerical feature names
feature_names.extend(numerical_cols)
# Get one-hot encoded feature names
if categorical_cols:
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_features = ohe.get_feature_names_out(categorical_cols)
    feature_names.extend(encoded_features)

# Calculate feature importance
print("\nCalculating feature importance...")
importance_results = calculate_feature_importance(
    final_model, 
    X_test_preprocessed, 
    y_test, 
    feature_names, 
    n_repeats=5  # Reduced from 10 to speed up execution
)

# Plot feature importance
sorted_idx = importance_results['importances_mean'].argsort()[::-1]
plt.figure(figsize=(12, 8))
plt.barh(
    [importance_results['feature_names'][i] for i in sorted_idx][:15],
    importance_results['importances_mean'][sorted_idx][:15]
)
plt.xlabel('Mean Decrease in AUC')
plt.title('Feature Importance (Top 15)')
plt.tight_layout()
plt.savefig('tf_feature_importance.png')
plt.close()

print("\nTop 10 important features:")
for i in sorted_idx[:10]:
    print(f"{importance_results['feature_names'][i]}: "
            f"{importance_results['importances_mean'][i]:.4f} ± "
            f"{importance_results['importances_std'][i]:.4f}")

# Save the preprocessor for future use
import joblib
joblib.dump(preprocessor, 'preprocessor.joblib')
print("\nPreprocessor saved as 'preprocessor.joblib'")

# Function for using the model on new data
def predict_loan_approval(model, preprocessor, data):
    """
    Predict loan approval using the trained TensorFlow model
    
    Parameters:
    model: Trained TensorFlow model
    preprocessor: Fitted preprocessor
    data: DataFrame with loan application data
    
    Returns:
    DataFrame with predictions
    """
    # Preprocess the data
    X_processed = preprocessor.transform(data)
    
    # Make predictions
    probabilities = model.predict(X_processed)
    predictions = (probabilities > 0.5).astype(int)
    
    # Add predictions to the original data
    result = data.copy()
    result['default_probability'] = probabilities
    result['predicted_loan_status'] = predictions
    result['loan_approved'] = (predictions == 0).astype(int)  # 0 = no default = approved
    
    return result

print("\nModel training and evaluation completed.")
print("Check the following output files:")
print("- best_loan_model.h5: Saved TensorFlow model")
print("- preprocessor.joblib: Saved data preprocessor")
print("- tensorflow_predictions.csv: Predictions on test data")
print("- tf_confusion_matrix.png: Confusion matrix visualization")
print("- tf_roc_curve.png: ROC curve visualization")
print("- tf_training_history.png: Training history visualization")
print("- tf_feature_importance.png: Feature importance visualization")