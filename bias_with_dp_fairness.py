import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import random
import os
from typing import List
import matplotlib.pyplot as plt

# Import PrivacyEngine from opacus
try:
    from opacus import PrivacyEngine
except ImportError:
    from opacus.privacy_engine import PrivacyEngine


def set_all_seeds(seed):
    """Set all seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_run_seeds(base_seed: int, num_runs: int) -> List[int]:
    random.seed(base_seed)
    np.random.seed(base_seed)
    seeds = random.sample(range(1, 1000000), num_runs)
    return seeds


# --- Dataset Class ---
class AdultDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# --- Preprocessing ---
def preprocess_loan_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray, int):
    df = df.copy()

    numeric_features = [
        'person_age', 'person_income', 'person_emp_exp',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length', 'credit_score'
    ]

    categorical_features = [
        'person_gender', 'person_education', 'person_home_ownership',
        'loan_intent', 'previous_loan_defaults_on_file'
    ]

    # Handle missing values for numeric columns
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)

    # Fill missing values for categorical columns with the mode
    for col in categorical_features:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Scale numeric features
    scaler = MinMaxScaler()
    X_numeric = scaler.fit_transform(df[numeric_features])

    # One-hot encode categorical features
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_categorical = ohe.fit_transform(df[categorical_features])

    # Concatenate numeric and categorical features
    X = np.hstack([X_numeric, X_categorical])

    # Extract target (assumed to be in column 'loan_status')
    y = df['loan_status'].values.astype(np.int64)

    input_dim = X.shape[1]
    return X, y, input_dim


def load_adult_data(train_file: str, test_file: str, batch_size: int = 512):
    train_df = pd.read_csv(train_file, skipinitialspace=True)
    test_df = pd.read_csv(test_file, skipinitialspace=True)

    # Preprocess each dataset
    X_train, y_train, input_dim = preprocess_loan_data(train_df)
    X_test, y_test, _ = preprocess_loan_data(test_df)

    # Create PyTorch datasets
    train_dataset = AdultDataset(X_train, y_train)
    test_dataset = AdultDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader, len(train_dataset), input_dim


# --- Model ---
class ImprovedAdultNet(nn.Module):
    def __init__(self, input_dim: int):
        super(ImprovedAdultNet, self).__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.norm1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.norm3 = nn.LayerNorm(64)
        self.dropout3 = nn.Dropout(0.1)
        self.fc_out = nn.Linear(64, 2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_norm(x)
        x = torch.relu(self.fc1(x))
        x = self.norm1(x)
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.norm2(x)
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.norm3(x)
        x = self.dropout3(x)
        out = self.fc_out(x)
        return out


# --- Postprocessing for Fairness (Calibrated Equalized Odds) ---
def postprocess_equalized_odds(model, test_loader, device, sensitive_values):
    """
    Computes group-specific thresholds on predicted probability for the positive class
    so that each group's TPR is as close as possible to the overall average TPR.
    Returns the postprocessed binary predictions, thresholds per group, and the target TPR.
    """
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            probs = torch.softmax(model(data), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(target.cpu().numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_sens = np.array(sensitive_values)  # Sensitive values loaded from CSV

    groups = np.unique(all_sens)
    default_preds = (all_probs >= 0.5).astype(int)
    group_tprs = {}
    for g in groups:
        mask = (all_sens == g)
        if np.sum(all_labels[mask] == 1) > 0:
            tpr = np.sum((all_labels[mask] == 1) & (default_preds[mask] == 1)) / (np.sum(all_labels[mask] == 1) + 1e-6)
        else:
            tpr = 0.0
        group_tprs[g] = tpr

    target_TPR = np.mean(list(group_tprs.values()))
    thresholds = {}
    for g in groups:
        mask = (all_sens == g)
        probs_g = all_probs[mask]
        labels_g = all_labels[mask]
        best_thresh = 0.5
        best_diff = float('inf')
        for thresh in np.linspace(0, 1, 101):
            preds = (probs_g >= thresh).astype(int)
            if np.sum(labels_g == 1) > 0:
                tpr = np.sum((labels_g == 1) & (preds == 1)) / (np.sum(labels_g == 1) + 1e-6)
            else:
                tpr = 0.0
            diff = abs(tpr - target_TPR)
            if diff < best_diff:
                best_diff = diff
                best_thresh = thresh
        thresholds[g] = best_thresh

    postprocessed_preds = np.zeros_like(all_probs, dtype=int)
    for g in groups:
        mask = (all_sens == g)
        postprocessed_preds[mask] = (all_probs[mask] >= thresholds[g]).astype(int)
    return postprocessed_preds, thresholds, target_TPR


# --- Training Function ---
def train_private_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    privacy_engine,
    epochs: int = 50,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    target_delta=1e-5
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_model_state = None
    all_train_accs = []
    all_test_accs = []
    all_epsilons = []
    all_train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1

            _, predicted = torch.max(output, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

            if (batch_idx + 1) % 20 == 0:
                current_batch_loss = running_loss / (batch_idx + 1)
                current_train_acc = 100. * correct_train / total_train
                print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Avg Loss: {current_batch_loss:.3f}, Batch Train Acc: {current_train_acc:.2f}%')

        avg_epoch_loss = epoch_loss / num_batches
        all_train_losses.append(avg_epoch_loss)

        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()

        train_acc = 100. * correct_train / total_train
        test_acc = 100. * correct_test / total_test
        epsilon = privacy_engine.accountant.get_epsilon(delta=target_delta)

        print(f'Epoch: {epoch+1}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Loss: {avg_epoch_loss:.3f}, ε: {epsilon:.2f}')

        all_train_accs.append(train_acc)
        all_test_accs.append(test_acc)
        all_epsilons.append(epsilon)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = model.state_dict().copy()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return all_train_accs, all_test_accs, all_epsilons, all_train_losses


# --- Fairness Evaluation Function (Equal Opportunity) ---
def evaluate_equal_opportunity(model, test_loader, sensitive_values):
    """
    Computes the True Positive Rate (TPR) for the positive class (label 1)
    for each sensitive group (here, genders).
    """
    model.eval()
    all_preds = []
    all_true = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    sensitive_values = np.array(sensitive_values)
    results = {}
    for group in np.unique(sensitive_values):
        group_mask = (sensitive_values == group)
        group_true = all_true[group_mask]
        group_preds = all_preds[group_mask]
        TP = np.sum((group_true == 1) & (group_preds == 1))
        FN = np.sum((group_true == 1) & (group_preds != 1))
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        results[group] = TPR
    return results


# --- Plotting Function ---
def plot_results(train_accs, test_accs, epsilons, train_losses, epochs):
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(18, 5))

    # Plot Training and Test Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_accs, label='Training Accuracy')
    plt.plot(epochs_range, test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot Training Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_losses, label='Training Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot Privacy Budget (Epsilon)
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, epsilons, label='ε (Privacy Budget)', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('ε')
    plt.title('Privacy Budget over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# --- Main Function ---
def main():
    # Base configuration
    BASE_SEED = 42
    NUM_RUNS = 1
    run_seeds = generate_run_seeds(BASE_SEED, NUM_RUNS)
    print(f"Generated seeds for runs: {run_seeds}")

    # Hyperparameters
    BATCH_SIZE = 5000
    EPOCHS = 11
    LEARNING_RATE = 0.005
    NOISE_MULTIPLIER = 4
    MAX_GRAD_NORM = 1
    TARGET_DELTA = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_train_accuracies = []
    all_test_accuracies = []
    all_epsilons = []
    all_train_losses = []
    fairness_results_list = []

    for run_idx, run_seed in enumerate(run_seeds):
        print(f'\nRun {run_idx + 1}/{NUM_RUNS} (Seed: {run_seed})')
        set_all_seeds(run_seed)

        try:
            print("Loading and preprocessing data...")
            train_loader, test_loader, sample_size, input_dim = load_adult_data("train.csv", "test.csv", batch_size=BATCH_SIZE)

            print("Initializing model...")
            model = ImprovedAdultNet(input_dim).to(device)

            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

            print("Initializing privacy engine...")
            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=NOISE_MULTIPLIER,
                max_grad_norm=MAX_GRAD_NORM,
                poisson_sampling=False,
            )

            print("Starting DP training...")
            train_accs, test_accs, epsilons, train_losses = train_private_model(
                model,
                train_loader,
                test_loader,
                optimizer,
                privacy_engine,
                epochs=EPOCHS,
                device=device,
                target_delta=TARGET_DELTA
            )

            all_train_accuracies.append(train_accs)
            all_test_accuracies.append(test_accs)
            all_epsilons.append(epsilons)
            all_train_losses.append(train_losses)

            # --- Fairness Evaluation BEFORE Postprocessing ---
            test_df = pd.read_csv("test.csv", skipinitialspace=True)
            sensitive_values = test_df['person_gender'].values  # Sensitive attribute: gender

            fairness_results = evaluate_equal_opportunity(model, test_loader, sensitive_values)
            fairness_results_list.append(fairness_results)
            print("\nFairness Evaluation (Equal Opportunity) BEFORE Postprocessing:")
            for group, tpr in fairness_results.items():
                print(f"Group: {group}, TPR: {tpr:.2f}")
            if len(fairness_results) == 2:
                groups = list(fairness_results.keys())
                pre_gap = abs(fairness_results[groups[0]] - fairness_results[groups[1]])
                print(f"Equal Opportunity Gap BEFORE: {pre_gap:.2f}")
            print("----------------------------------------------------\n")

            # --- Fairness Postprocessing ---
            post_preds, thresholds, target_TPR = postprocess_equalized_odds(model, test_loader, device, sensitive_values)
            # Compute TPR for each group after postprocessing:
            post_fairness = {}
            # Reconstruct true labels from the test set:
            all_true = []
            for _, target in test_loader:
                all_true.extend(target.numpy())
            all_true = np.array(all_true)
            for group in np.unique(sensitive_values):
                mask = (sensitive_values == group)
                if np.sum(all_true[mask] == 1) > 0:
                    tpr = np.sum((all_true[mask] == 1) & (post_preds[mask] == 1)) / (np.sum(all_true[mask] == 1) + 1e-6)
                else:
                    tpr = 0.0
                post_fairness[group] = tpr
            print("\nFairness Evaluation (Equal Opportunity) AFTER Postprocessing:")
            for group, tpr in post_fairness.items():
                print(f"Group: {group}, TPR: {tpr:.2f}")
            if len(post_fairness) == 2:
                groups = list(post_fairness.keys())
                post_gap = abs(post_fairness[groups[0]] - post_fairness[groups[1]])
                print(f"Equal Opportunity Gap AFTER: {post_gap:.2f}")
            print("----------------------------------------------------\n")

            # Optionally, you can perform SHAP analysis here if feature names are available.
            # shap_analysis(model, test_loader, feature_names, device)

        except Exception as e:
            print(f"Error in run {run_idx + 1}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue

    avg_train_accs = np.mean(all_train_accuracies, axis=0)
    avg_test_accs = np.mean(all_test_accuracies, axis=0)
    avg_train_losses = np.mean(all_train_losses, axis=0)
    avg_epsilons = np.squeeze(np.mean(all_epsilons, axis=0))

    print("\nFinal Results:")
    print("Average Train Accuracy:", avg_train_accs)
    print("Average Test Accuracy:", avg_test_accs)
    print("Final Privacy Budget (ε):", avg_epsilons)

    if len(fairness_results_list) > 0:
        all_groups = fairness_results_list[0].keys()
        avg_fairness = {group: np.mean([run_results[group] for run_results in fairness_results_list])
                        for group in all_groups}
        print("\nAverage Fairness (Equal Opportunity) Metrics over Runs BEFORE Postprocessing:")
        for group, tpr in avg_fairness.items():
            print(f"Group: {group}, Average TPR: {tpr:.2f}")
        if len(avg_fairness) == 2:
            groups = list(avg_fairness.keys())
            pre_avg_gap = abs(avg_fairness[groups[0]] - avg_fairness[groups[1]])
            print(f"Average Equal Opportunity Gap BEFORE: {pre_avg_gap:.2f}")

    plot_results(avg_train_accs, avg_test_accs, avg_epsilons, avg_train_losses, EPOCHS)


if __name__ == "__main__":
    main()
